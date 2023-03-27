"""
All Rights Reserved
@author Morteza Emadi
This identifies on/off status of sensors in Z-Wave network
"""

import numpy as np
import pandas as pd
import csv
import easygui
import datetime
from datetime import datetime
from data_store import local_temp_path, HomeReadingStore
from data_preprocess import homeid


class submeter(object):

    def timedelta64_to_secs(self, timedelta):
        """Convert `timedelta` to seconds.

        Parameters
        ----------
        timedelta : np.timedelta64

        Returns
        -------
        float : seconds
        """
        if timedelta is None:
            return np.array([])
        else:
            return timedelta / np.timedelta64(1, 's')

    def removeDuplicates(self, eventDF):
        # Remove all but first in consecutive series of offs or ons in a dataframe of ons and offs
        # Could occur after removal of events under min length
        ########## TODO: ****AMLAN statechange ha harbar bas belaxe ham bashan,magar inke beghole lhodesh \n
        #####removale e yek evene zire min,baese tekrar 2ta on bshe ke inja miad chck mikone statechange ro diff
        ####migire va harja =1 bshe ro negah midare
        eventDF['change_duplicate'] = eventDF['state_change'].diff()
        eventDF = eventDF[eventDF['change_duplicate'] != 0]
        return eventDF

    def calculatePeriods(self, eventDF, endtime):
        # Calculate duration of on/off events. (Time of next event minus time of event.)
        eventDF['duration'] = -eventDF['time'].diff(-1)
        # Last event is a special case as there's no time of next event
        # Use end of sample time instead. If the event is very close to the end of the sameple
        # time, it will get discarded as too short when running the rule but will be picked up
        # as part of the overlap sample on the next run.
        lastDuration = endtime - eventDF.iloc[-1, eventDF.columns.get_loc('time')]
        eventDF.iloc[-1, eventDF.columns.get_loc('duration')] = lastDuration
        # Convert to seconds
        eventDF['duration'] = self.timedelta64_to_secs(eventDF['duration'])
        return eventDF



    def get_ons_and_offs(self, chunk, end, prevEvent, min_off_duration, min_on_duration,
                         max_on_duration, on_power_threshold, min_standby_duration, normal_on_power,acceptable_nan,whennormal,extra, sensorid=0,homeid=0,dx=False):
        # Convert Series of times and power values to DataFrame for manipulation.
        print(f"currently processing on_offs of sensor{sensorid}")
        eventDF = pd.DataFrame({'time': chunk.index, 'value': chunk['value']})
        #mostly for determining gaps
        eventDF=self.calculatePeriods(eventDF, eventDF.index[-1])
        eventDF=eventDF.reset_index(drop=True)
        if eventDF["value"].isna().sum() > 1:
            easygui.msgbox(f"an ZWaVe sensor= {sensorid} from home{homeid} have more than 1 NAN!(Nan/total: {eventDF['value'].isna().sum()}/{eventDF['value'].size}) (trouble like radiator is probable now!)",
                           title=f"NaN in ZWave! home {homeid} sensors={sensorid}")
        onchunk_mask=(((eventDF['duration'] < min_standby_duration) & \
                (eventDF['value'] >= on_power_threshold)) | (eventDF['value'] >= normal_on_power))
        eventDF.loc[onchunk_mask, 'onchunk'] = 1


        ### We ned to neglect any anamolies of NaN and accept them!(hasnt been accepted in onchunk masks yet) but being between some of the trusted onchunk nominiees!(so here we analyse the  ie, if power of them decreases even to zero in washing machine!
        on_nan_df = eventDF.loc[eventDF["onchunk"] == 1.0]
        ### whats the distance between those on nominiees? (na duratione evente khodeshun)
        on_nan_df['duration'] = -on_nan_df['time'].diff(-1)
        on_nan_df['duration'] = self.timedelta64_to_secs(on_nan_df['duration'])
        on_nan_df['index_diff'] = -on_nan_df.index.to_series().diff(-1)
        ###rows should NOT be sequencial(chon dar vaghe khorji inja onhaie bayad bashe ke baedeshun NaN oomade,va khob vaghti sequential ha ro miari eshteb select kardi) and duration should be less than MIN_ON_STANDBY/2 so as to accept those NAN as mistaken ones
        ### so many of the on_nans_df here drops from the df
        on_nan_df = on_nan_df.loc[(on_nan_df["duration"] <= (acceptable_nan)) &
                                              (on_nan_df["index_diff"] > 1)]

        ### Example of 584,,585,586 indexes should be selected as 582 is still true and
        for index, row in on_nan_df.iterrows():

            diff = int(row["index_diff"] - 1)
            ##az oon on_nan ha alan dorost hash munde,kafie tedadi az index haye baede oon ha be andaze (index-diff) =True(=1) bgirim
            eventDF.loc[index+1:index+diff+1, "onchunk"] = 1
#####################################################################################################
        ##we wanna find the on_nominees clusters, but the first step is to find their last row! because they are distinct 'coz of their index_diff value which is larger than1 !(if we calculate again their index_diff here)since their next index is not available in the on _candidate_df
        clusters = []     #  [(134,137),(456,5567),.....]
        on_candidate_df = eventDF.loc[eventDF["onchunk"] == 1]
        on_candidate_df['index_diff'] = -on_candidate_df.index.to_series().diff(-1)
        border_df = on_candidate_df.loc[on_candidate_df["index_diff"] > 1]
        ##cluster means a bunch of on_nominees together!//lets find their index like this=  [(134,137),(456,5567),.....]
        ####these indexes below(called BORDERs) are all the last index of each cluster
        borders = border_df.index.to_series()
        last_border = 0
        for value in borders:
            first_border = on_candidate_df.loc[(on_candidate_df.index > last_border) & (on_candidate_df.index <= value)].index.to_series().min()
            if np.isnan(first_border):
                first_border=0
            last_border = value
            clusters.append((first_border, value))
        error_dict = {}
        for cluster in clusters:
            ###for each cluster in the first minutes of cluster we should check is there a "normal_on_power" in the initial times,here endtime means where we obligate the onchunk to have its first onpower
            start_time = eventDF.loc[cluster[0], "time"]
            end_time = start_time + pd.Timedelta(seconds=whennormal)
            end_time=min([end_time,eventDF.loc[cluster[1],"time"]])
            ##deghat age dar duratione boland endtime ro haman 3barabare min_on_dur entekhab bshe va baed noraml_on_power dar oon bazeh dar khate zir peida nashe yaeni normal on power agar ham hast kheili dire va avayele oon cluster yahtamel bas hazf bshe
            condition_df = eventDF.loc[(eventDF["time"] >= start_time) &
                                       (eventDF["time"] <= end_time) &
                                       (eventDF["value"] >= normal_on_power)]
            if condition_df.empty:
                eventDF.loc[cluster[0]:cluster[1], "error_npower"] = 1
                eventDF.loc[cluster[0]:cluster[1], "onchunk"] = 0
                print(f"error_npower:{cluster[0]}_{cluster[1]}.")
                error_dict[f"{sensorid}"] = f"error_npower in cluster {cluster[0]}."

            ##calculating each cluster real period (not the normal power acceptable delay)
            if  (eventDF.loc[cluster[1],"duration"] > 1051): ##because in case of vacuum cleaner the whole sensor will be turned off after an on session
                cluster_dur = (eventDF.loc[cluster[1] , "time"] - eventDF.loc[cluster[0], "time"])
            else:
                cluster_dur = (eventDF.loc[cluster[1] + 1, "time"] - eventDF.loc[cluster[0], "time"])
            if not ((cluster_dur >= pd.Timedelta(seconds=min_on_duration)) \
                & (cluster_dur < pd.Timedelta(seconds=max_on_duration))):
                eventDF.loc[cluster[0]:cluster[1], "error_clusterdur"] = 1
                eventDF.loc[cluster[0]:cluster[1], "onchunk"] = 0
                print(f"error_clusterdur:{cluster[0]}_{cluster[1]}.")
                error_dict[f"{sensorid}"] = f"error_clusterdur in cluster {cluster[0]}."

        ##saving error dict in a csv
        with open(f'errordicts/home{homeid}_zwave_errors.csv', 'a') as f:
            w = csv.DictWriter(f, error_dict.keys())
            w.writeheader()
            w.writerow(error_dict)

        eventDF["onchunk"] = eventDF["onchunk"].fillna(value=0)
        # todo! => imp new:
        eventDF.loc[eventDF.value.isnull(), 'onchunk'] = np.nan

        if (eventDF.columns == 'error_npower').any():
            eventDF["error_npower"] = eventDF["error_npower"].fillna(value=0)
        if (eventDF.columns == 'error_clusterdur').any():
            eventDF["error_clusterdur"] = eventDF["error_clusterdur"].fillna(value=0)

        eventDFanalysor=eventDF.copy()
        #Todo!! eventDFanalysor.loc[eventDFanalysor["error_clusterdur"] == 1 ]

        eventDF=eventDF.set_index("time")
        on_off = eventDF["onchunk"]
        # on_off = on_off.loc[on_off.shift(1) != on_off]

        # ###Just for changing on periods to switches of ON and OFF events!
        # eventDF['state_change'] = eventDF['onchunk'].astype(np.int8).diff()


        # Just for changing on periods to switches of ON and OFF events!
        ##for better erecognizing nanS in results of this module
        eventDF["onchunk"] = eventDF["onchunk"].fillna(5)
        # eventDF['state_change'] = eventDF[eventDF['onchunk'].notnull()]["onchunk"].astype(np.int8).diff()
        eventDF["state_change"] = eventDF['onchunk'].astype(np.int8).diff()


        if (eventDF.iloc[0, eventDF.columns.get_loc('onchunk')] == 1):
            eventDF.iloc[0, eventDF.columns.get_loc('state_change')] = 1
        else:
            eventDF.iloc[0, eventDF.columns.get_loc('state_change')] = -1


        eventDF = eventDF[(eventDF.index == eventDF.index[0])|(eventDF.index == eventDF.index[-1]) | (eventDF['state_change'] == 1) | (eventDF['state_change'] == -1) | (eventDF['state_change'] == 4) | (eventDF['state_change'] == -4) | (eventDF['state_change'] == 5) | (eventDF['state_change'] == -5)]
        # eventDF = eventDF[(eventDF['state_change'] == 1) | (eventDF['state_change'] == -1) | (eventDF['value'].isna())]


        eventDF = self.removeDuplicates(eventDF)
        eventDF['state_change'] = eventDF['state_change'].map({1: 'on', -1: 'off', 5:np.nan, -5: "off", 4: np.nan, -4: "on"})
        if pd.isnull(eventDF['state_change'][-1]):
            eventDF.state_change[-1] = eventDF.state_change[-2]

        # eventDF['state_change'] = eventDF['state_change'].map({1: 'on', -1: 'off'})
        del eventDF['onchunk']
        del eventDF['duration']
        del eventDF['value']
        del eventDF['change_duplicate']
        try:
            del eventDF['error_npower']
        except:
            pass
        try:
            del eventDF['error_clusterdur']
        except:
            pass

        eventDF = eventDF.reset_index(level=0)
        print("finished")
        del eventDFanalysor
        return eventDF

def detector_zwave(homeid):

    import json
    homes = [homeid]

    with open("config\gt_rules_morteza__.json", "r") as rules_file:
        rule = json.load(rules_file)

    rules = {x["appliance"]:x for x in rule["rules"]}
    if 162 in homes:##in sensor maraz dare 20deghe sare 50watt pause mikone ama aadie vasash!(acceptable nan ro bas bala bebari ta  inharo candidate on chunk bokone)
        rules["washingmachine"]=rules["washingmachine162"]
    if 175 in homes:##in sensor maraz dare 20deghe sare 50watt pause mikone ama aadie vasash!(acceptable nan ro bas bala bebari ta  inharo candidate on chunk bokone)
        rules["washingmachine"]=rules["washingmachine175"]
        rules["dishwasher"] = rules["dishwasher175"]
    if 146 in homes:
        rules["washingmachine"]=rules["washingmachine146"]

    readings = pd.DataFrame()

    for homeid in homes:
        with HomeReadingStore() as s:
            readings = readings.merge(s.get_readings(homeid), left_index=True, right_index=True, how='outer',
                                      suffixes=('', homeid))
        readings_process = readings.rename(columns=lambda x: x.split("_")[0])
        sensor_dict = {x.split("_")[0]: x for x in readings.columns if "_" in x}


    sample_rate = 180
    apps = ["dehumidifier","toaster","vacuumcleaner","washingmachine","other","microwave","kettle","dishwasher"]

    # backup_reading = readings
    # readings = readings.rename(columns=lambda x: x.split("_")[0])


    result = pd.DataFrame()
    result2 = pd.DataFrame()

    for app in apps:
        if app not in readings_process:
            continue

        app_reading = readings_process[readings_process[app].notnull()].rename(columns={app : 'value'},errors="raise")
        app_reading = app_reading.replace(1234567.000001, np.nan)
        on_offs = submeter().get_ons_and_offs(app_reading,
                                              app_reading.index[-1], None,
                                              min_off_duration=float(rules[app]["min_off_duration"]),
                                              min_on_duration=float(rules[app]["min_on_duration"]),
                                              max_on_duration=float(rules[app]["max_on_duration"]),
                                              on_power_threshold=float(rules[app]["on_power_threshold"]),
                                              min_standby_duration=float(rules[app]["min_standby_duration"]),
                                              normal_on_power=float(rules[app]["normal_on_power"]),
                                              acceptable_nan=float(rules[app]["acceptable_nan"]),
                                              whennormal = float(rules[app]["whennormal"]),
                                              extra = float(rules[app]["extra"]),
                                              sensorid = sensor_dict[app],
                                              homeid = homeid)
        # outputresample = on_offs.set_index("time").asfreq(freq="3T", method="nearest").rename(columns={"state_change": app})
        on_offs = on_offs.set_index("time").rename(columns={"state_change": app})
        result = result.merge(on_offs, how="outer", left_index=True, right_index=True)

    ####just for doing filllna but not the first nan and the last nans==
    for app in apps:

        if app not in result:

            continue

        result_app = result[app]
        result_app = result_app.reset_index()
        minimum = result_app[result_app[app].notnull()].index.min()
        maximum = result_app[result_app[app].notnull()].index.max()

        first = result_app.loc[:minimum]
        second = result_app.loc[minimum+1:maximum+1].fillna(method="ffill")
        third = result_app.loc[maximum+2:]

        holder_app = pd.concat([first, second, third])
        result[app] = holder_app.set_index("time")


    result.to_csv(f"on_offs/home{homeid}_onoff_zwave_1401.csv")
    print("it's done")
