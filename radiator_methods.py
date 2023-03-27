"""
All Rights Reserved
@author Morteza Emadi
This is a crucial module with more than 5 pivotal methods that other scripts use in their executions
"""
from data_store import MetaDataStore
from data_store import local_temp_path, ReadingDataStore
from data_preprocess import homeid
import numpy as np
import easygui
import pandas as pd
from metadata import MetaData
with MetaDataStore() as s:
    metadata = MetaData(s)
####*** Important parametre==>
sample_rate = 180

def identify_io_room(roomid,homeid="optional"):
    """
    for the input room this stores all of the name of the couplesensors of radiators located in there
    :param roomid:
    :return:
    """
    result = []
    holder = metadata.sensorbox
    sensor_holder = metadata.sensor
    sensorbox_df = holder.loc[(holder["roomid"] == roomid) \
        & ((holder["function"] == "Radiator")
           | ((holder["clamp1pipe"]=="RadiatorInput")|(holder["clamp1pipe"]=="RadiatorOutput")))]
    if sensorbox_df.empty:
        print(f"there were no radiators in room{roomid} of home{homeid}")
    for index, box in sensorbox_df.iterrows():

        box_id = box["sensorboxid"]
        radiator_sensor_df = sensor_holder.loc[(sensor_holder["sensorboxid"] == box_id) &
                                               (sensor_holder["type"] == "clamp_temperature")]
        counter_info = radiator_sensor_df[["sensorid", "counter"]]
        clamp_info = box[["clamp1pipe", "clamp2pipe"]]

        result_dict = {}

        if radiator_sensor_df.shape[0] == 2:
            result_dict[clamp_info[counter_info.iloc[0]["counter"] -1]] = counter_info.iloc[0]["sensorid"]
            result_dict[clamp_info[counter_info.iloc[1]["counter"] -1]] = counter_info.iloc[1]["sensorid"]

        else:

            print(f"Sensorboxid: {box_id}, roomid: {roomid} in home{homeid} has 1 clamp sensor of Radiator instead of 2.")
        if result_dict:
            result.append(result_dict)

    return result

def nanpercent(df, threshold=65):
    """
    after preprocessing any of the DFs this will check if there are many NaNs in Percentage!
    returns null if less nan, else:returns which of the couple sensors?(0 or 1 or 2)
    """
    percent_missing = df.isnull().sum() * 100 / len(df)
    lst = []
    for a in df.columns:
        if percent_missing[a] >= threshold:
            lst.append(a)
            easygui.msgbox(f"sensors {a} from home{homeid} and rooms {roomid} have many NaNs+",
                           title=f"home {homeid} sensors={a}")

    return lst  ##returns null if less nan, else:returns which of the couple sensors?(0 or 1 or 2)


def identify_io_home(homeid):
    """
    this first finds all of the rooms of the homeid you input it,then uses the above method(identify io room) to
    store all the sesnor ids of radiators of each rooom in results(so all roooms of one home)
    :param homeid:
    :return:
    """
    rooms = metadata.room.loc[metadata.room["homeid"] == homeid]
    result = {}
    for index, room in rooms.iterrows():

        roomid = room["roomid"]
        intermediate_result = identify_io_room(roomid,homeid)

        if intermediate_result:
            result[roomid] = intermediate_result

    return result

def identify_io_enhanced():
    """
    this is just an input maker for the above func(identify_io_home since it give the homid of all enhanced homes as input to the above func
    :return:
    """
    holder = metadata.combinedhome
    homes = holder.loc[holder["install_type"] == "enhanced"]
    result = {}
    for index, home in homes.iterrows():

        homeid = home["homeid"]
        result[homeid] = identify_io_home(homeid)

    return result


def read_radiator_sensor(roomid,homeid="optional"):
    """
    get the room id and the result will give the dataframes each containing two column of input/output temp sesnors readings synced by time
    :param roomid:
    :return:
    """
    data = identify_io_room(roomid,homeid)

    for couple_sensor in data:

        output_id = couple_sensor["RadiatorOutput"]
        input_id = couple_sensor["RadiatorInput"]

        with ReadingDataStore(data_dir=local_temp_path) as s:

            readings_input = s.get_sensor_readings(input_id)
            readings_output = s.get_sensor_readings(output_id)
            readings = preprocess_radiators(readings_input, readings_output,homeid,roomid,input_id,output_id)
            result={}
            result[tuple(couple_sensor.values())] = readings

        return result


def on_off_radiator(couple_data,roomid,sesnorsid):
    """
    couple data(input here) is a dataframe containing 2 coulmn of input/output readings of each
    radiator and brings a df which tell the on/off timestamps in 1column for one radiator
    pay attention that 2nd arg(roomid) was added afterward for tuning the thresholds of on/off recognizer!
    :param couple_data:
    :param roomid:
    :return:
    """
    sensordids=f"radiator_{roomid}_{sesnorsid}"
    holder = couple_data
    holder["diff"] = holder["value_input"] - holder["value_output"]

    holdermean=pd.DataFrame()
    holdermean["mean"]=(holder["value_input"]+holder["value_output"] )/2


    holder["change_diff"] = -holder["diff"].diff(-1)
    holder["change_inp"] = -holder["value_input"].diff(-1)

    holder["slope"] = -holder["diff"].diff(-1)

    ###as we wanna compare the declining slope of consequative diff(in-out)-also for loop isnt effective in the case of large df-,and any non-negative slope should reset this cumulative slope! so we use nan to disturbe the cumulative process
    ##the zero version assumes all flatlines of slope as ZERO
    holder["zslope"] = holder["slope"].apply(lambda x: -1 if x < 0 else np.nan)
    zcumsum = holder["zslope"].cumsum().fillna(method="pad")
    reset = -zcumsum[holder["zslope"].isnull()].diff().fillna(zcumsum)
    holder["zcumslope"] = holder["zslope"].where(holder["zslope"].notnull(), reset).cumsum()
    holder["slope"] = holder["slope"].apply(lambda x: -1 if x <= 0 else np.nan)

    cumsum = holder["slope"].cumsum().fillna(method="pad")

    reset = -cumsum[holder["slope"].isnull()].diff().fillna(cumsum)
    ###khate zir mige sotune nahaie CumSlope hamun Slope hast ke ruyesh methode ".cumsum" mikhore ama harja null bashe ro
    # bjash az df ie be esme reset bzar
    holder["cumslope"] = holder["slope"].where(holder["slope"].notnull(), reset).cumsum()



    holder["in_slope"] = -holder["value_input"].diff(-1)

    ##as we wanna compare the declining slope of input value
    #  First the Zero version (flats will be reset to zero)
    holder["in_zslope"] = holder["in_slope"].apply(lambda x: -1 if x < 0 else np.nan)
    in_zcumsum = holder["in_zslope"].cumsum().fillna(method="pad")
    reset = -in_zcumsum[holder["in_zslope"].isnull()].diff().fillna(in_zcumsum)
    holder["in_zcumslope"] = holder["in_zslope"].where(holder["in_zslope"].notnull(), reset).cumsum()
    ###### here we will have non zero version!=> (flats will also decrease the neegtive slope)
    holder["in_slope"] = holder["in_slope"].apply(lambda x: -1 if x <= 0 else np.nan)
    in_cumsum = holder["in_slope"].cumsum().fillna(method="pad")
    reset = -in_cumsum[holder["in_slope"].isnull()].diff().fillna(in_cumsum)
    holder["in_cumslope"] = holder["in_slope"].where(holder["in_slope"].notnull(), reset).cumsum()
    del holder["slope"]
    del holder["in_slope"]
    del holder["zslope"]
    del holder["in_zslope"]

#######*****todo!!!!!! ==> for analysing the results evaluate the holder with "diff" column as the priotrizer,then write down all the "indexes" in the sciview bar,which their switch candidates in max diff is 0 or in zero diff is 1!

    ###identifiying "switch" on candidates
    ##Todo!!= ###sharte zir akharin viryesh va mohemtarin bude ke akharinbar vase in mavared javab dae bud==>roomid in [2251,2248,2258]:
    holder.loc[holder.loc[(((holder["change_diff"] >= 74) & (holder["diff"].shift(-1) >= 35) & (holder["value_input"] >= 201)) \
               | ((holder["change_diff"] >= 39) & (holder["change_inp"] >= 75) & (holder["diff"].shift(-1) >= 35)) \
                | ((holder["change_diff"] >= 46) & (holder["change_inp"] >= 48) & (holder["diff"].shift(-1) >= 35)) \
               | ((holder["diff"] >= 73) | (holder["value_input"] >= 320) ))].index \
                ,"switch_candidate"] = 1
    holder = holder.reset_index()
    ##identify all off candidates by using slope of the diff(diff of input and output) and slope of input
    ## for rromid=2255  sharte (holder["in_zcumslope"] <= -4)) ghablesh -3 bud! va zemnan (holder["zcumslope"]<=-4) ghablesh -3 bud,pas general doros kon ama sharte jadid tulanish kheili khube
    if roomid == 2243:
        holder.loc[holder.loc[((((holder["cumslope"] <= -4)|(holder['cumslope'].shift(1)<= -4 )|(holder['cumslope'].shift(-1)<= -4))& (holder["in_zcumslope"] <= -3)) \
                               | ((holder["zcumslope"]<=-3) & (holder["value_input"]<=holder["value_input"].shift(5)))\
                               |((holder["cumslope"]<=-4) & (holder["in_zcumslope"]<=-4)))].index, "switch_candidate"] = 0
    ##ToDo!: DeGhaT! ghablan sharte zir akharin tuninge man bud ke faghat vase in roomha(roomid in [2255,2251,2248,2258]:)javab midad,deghat va check kon baedan!
    ## masan khune 242 va kole 162 ba chand khate zir bud
    else:###nokte dar khane 146 didam ke ba sharte diff<-32(le adade khubie chon range ziadi darim) ya sharte input<320 trigere nahai mishe(yaeni baghie shart ha mesle cum<-4 kheili zud triger shodan be ghalat!)
        holder.loc[holder.loc[(((holder["in_zcumslope"] <= -4) &((holder["cumslope"] <= -4)|(holder['cumslope'].shift(1)<= -4 )|(holder['cumslope'].shift(-1)<= -4))&(holder["diff"]<=29)& (holder["value_input"] <= 300)) \
                               | ((holder["value_input"]<=holder["value_input"].shift(5)) & (holder["zcumslope"]<=-4) & (holder["diff"]<=29)& (holder["value_input"]<=303)) \
                            |(((holder["in_cumslope"]<=-24) | ((holder["in_cumslope"]<=-18) & (holder["cumslope"]<=-12))) & ((holder["diff"]<=29))& (holder["value_input"]<=303) )\
                               | (((holder["cumslope"]<=-5) & (holder["in_zcumslope"]<=-5)) &(holder["diff"]<=28) & (holder["value_input"]<=300) ) \
                   | ((holder["diff"].isnull())))].index \
                             , "switch_candidate"] = 0 ##imp line,every row with any of the in/out=nan will be off


    # b =  a.loc[(a["function"] == "Radiator") & (a["type"] == "clamp_temperature")]
    holder["switch_candidate"] = holder["switch_candidate"].fillna(method="pad")
    #####for the rows before the first switch on and also the main gaps between on/off switches!
    holder["switch_candidate"] = holder["switch_candidate"].fillna(value=0)


    holder = holder.set_index("time")
    on_off = holder["switch_candidate"]
    on_off = on_off.loc[on_off.shift(1) != on_off]
    on_off = on_off.replace([0, 1], ["off", "on"])
    on_off = on_off.reset_index(level=0).rename({"switch_candidate": "on/off"})

    return on_off
    pass


def on_off_room(roomid,homeid="optional"):
    ##next line finds at first 2 sensorids then brings their readings as a
    #..a Dict,containing any couple Radiators readings DF in each value,so their keys are the ids of those sensors in format of a tuple!
    data = read_radiator_sensor(roomid,homeid)

    result = {}
    ###here we may have multiple radiators=keys(each has 2 reading ids in the df in the value) in a dict! so
    #..so loop for each key,and send their couple readings to the main on_off detector of radiators!
    for k in data.keys():

        datum=None
        datum = data[k]
        ####Todo: do this below trick for all other clean data gatherings! so if there is any problem make the df empty!s here it wont congatulat
        if datum is not None:
            result[k] = on_off_radiator(datum,roomid,k)

    return result

def meanvalue_radiators(roomid,homeid="not set"):
    """
    get the room id and the result will give the dataframes each containing two column of input/output temp sesnors readings synced by time
    :param roomid:
    :return:
    """
    data = identify_io_room(roomid,homeid)
    result = {}
    readings= None
    if data:
        for couple_sensor in data:

            output_id = couple_sensor["RadiatorOutput"]
            input_id = couple_sensor["RadiatorInput"]

            with ReadingDataStore(data_dir=local_temp_path) as s:

                readings_input = s.get_sensor_readings(input_id)
                readings_output = s.get_sensor_readings(output_id)
                readings = preprocess_radiators(readings_input,readings_output,homeid,roomid,input_id,output_id)
            if readings is not None:
                readings = readings.reset_index()
                # readings = readings_input.merge(readings_output, on="time", suffixes=("_input", "_output"))
                readings.dropna(inplace=True)

                readings = readings.set_index("time")
                readings["value"] = readings[['value_input', 'value_output']].mean(axis=1)
                readings.drop(columns=['value_input','value_output'],inplace=True)
                result[tuple(couple_sensor.values())] = readings

            ##DEGHAT INJA ahamiat dasht *result* ro khareje in if avordam chon result besurate tple dare meghdar migire baraye otagh hai ke chand ta joft radiator dashte bashan
            return result

    else:
        return result


def preprocess_radiators(readings_input, readings_output,homeid="not set",roomid="not set",input_id="not set",output_id="not set"):
    result = {}
    if ((len(readings_input) - len(readings_output)) / len(readings_input)) < 0.1 and \
            abs(readings_output.time[0] - readings_input.time[0]) < np.timedelta64(1, 'D'):
        readings = readings_input.merge(readings_output, on="time", suffixes=("_input", "_output"))
        readings = readings.resample(str(sample_rate / 60) + 'T', on='time').mean()

        ####### preprocess
        for i in range(2):
            colnames = ["value_input", "value_output"]
            df = readings.drop(columns=colnames[i])
            s = df[colnames[abs(1 - i)]].notnull()  # ye bolean mask mide harja NOT Null bashe TRUE mide
            s = s.ne(s.shift()).cumsum()
            m = df.groupby([s, df[colnames[abs(1 - i)]].isnull()])[colnames[abs(1 - i)]].transform('size').where(
                df[colnames[abs(1 - i)]].isnull())
            df = df.interpolate(limit_area='inside', method='linear').mask(
                m >= 7)
            df = df[df.isna() & (~df.shift().isna())] = df.ffill(limit=3)
            df = df[df.isna() & (~df.shift(-1).isna())] = df.bfill(limit=3)
            readings[colnames[abs(1 - i)]] = df
        ####Here i added a zero value before any NAN Chunks! 'coz in energy calculation
        # those nan chunk which has got ON and High value prioir to them will make an
        # abnormality for the NAN chunk(which is an off chunk)in energy calculation!
        #next line=for convinience temporarily use mean of two columns for fininding nans
        mean_read=readings[['value_input', 'value_output']].mean(axis=1)
        tmp = mean_read.isna()
        # tmp = readings[pd.isna(readings["value_input"]) or pd.isna(readings["value_output"])]
        #### IMP! u can check the below df for the philosphy of mechanism
        df_s = pd.DataFrame({"mean_read": mean_read, "na_cumsum": tmp.cumsum(), "diff": tmp.cumsum().diff(), "diff2": tmp.cumsum().diff().diff()})
        df_first_nans = readings[df_s["diff2"] == 1]
        df_first_nans = df_first_nans.replace({np.nan: 0})
        df_first_nans.index = df_first_nans.index - pd.Timedelta(seconds=0.5)
        readings = readings.append(df_first_nans).sort_index()
##########So imp! put 70% threshold for nan percent ha!
        nanlist = nanpercent(readings, 70)
        if nanlist:  ### hata age yeki az 2taye inp/out empty bashan inja result ro por nmikone
            print(
                f"sensors {nanlist} from home{homeid} and rooms {roomid} have many NaNs+ ***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n***************************************************************\n")
        else:
            return readings

    else:
        easygui.msgbox(f"sensors {input_id}_{output_id} from home{homeid} and rooms {roomid} arent in accordance",
                       title=f"not in accordance temp series! home {homeid} room={roomid}")


def calculate_heat_output(reading_radiator, room_id, rooms,status=None):

    data_org = pd.DataFrame(reading_radiator)
    boxes = metadata.sensorbox.loc[(metadata.sensorbox["roomid"] == room_id) &
                                         (metadata.sensorbox["sensorbox_type"] == "room")]["sensorboxid"]

    sensors = metadata.sensor.loc[(metadata.sensor["sensorboxid"].isin(boxes)) &
                                  (metadata.sensor["type"] == "temperature")]["sensorid"]

    rooms.remove(room_id)

    #####dar in loop e zir mibinim age oon otagh tempreture nadare ya age dare chand tast mire donbale
    if sensors.shape[0] != 1:

        for room in rooms:

            boxes = metadata.sensorbox.loc[(metadata.sensorbox["roomid"] == room) &
                                           (metadata.sensorbox["sensorbox_type"] == "room")]["sensorboxid"]

            sensors = metadata.sensor.loc[(metadata.sensor["sensorboxid"].isin(boxes)) &
                                          (metadata.sensor["type"] == "temperature")]["sensorid"]

            if sensors.shape[0] == 1:

                easygui.msgbox(f"room{room_id} from home{homeid} had NO singular room temp in it but we put a naighbours room temp for it! ",
                               title=f"ALternative Temp: room{room_id} home {homeid}")
                break

    if sensors.shape[0] == 1:

        adr = r"C:\Users\MortezaEm\Desktop\debugging tracing files\running_path\mylocaldircoding_JUST ROOM TEMP"

        with ReadingDataStore(data_dir=adr) as s:

            reading_temp = s.get_sensor_readings(sensors.iloc[0])

            # reading_temp = reading_temp.set_index("time").asfreq(freq=str(sample_rate / 60) +'T', method="nearest")
            reading_temp = reading_temp.resample(str(sample_rate / 60) + 'T', on='time').mean()
            # readings = readings.set_index("time")
    else:
        ###in dar halatie ke baghie otagh ha ham hata hich kodum tak TEMP nadshte bashan!!
        reading_temp = pd.DataFrame(pd.Series(175, index=np.arange(data_org.shape[0])))
        easygui.msgbox(f"room{room_id} from home{homeid} had NO singular room temp in it and neighbour rooms,so "
                       f"we put 175 as alternative",
                       title=f"NO TEMP:room{room_id} home {homeid} ")


    radiator_name = data_org.columns[0]

    merged_data = data_org.merge(reading_temp, how="left", right_index=True, left_index=True).set_axis([
        "mean", "roomtemp"
    ], axis=1, inplace=False)

    merged_data["roomtemp"] = merged_data["roomtemp"].fillna(method="pad")
    merged_data = merged_data.loc[data_org.index, :]
    #here "/200" is "rated delta T" in the formula which is the standard diff of room temp and radiator surface temp
    result_series = ((merged_data["mean"] - merged_data["roomtemp"]) / 200) ** 1.3
    err_index = result_series.loc[result_series < 0].index
    if len(err_index)>0:
        result_series.loc[err_index] = 0
        easygui.msgbox(f"room{room_id} from home ??:mean temp of Radiator in {len(err_index)} points is less than Mean temp!(diff=Neg!),so we put zero in those points(energy=0)",
                       title=f"#{len(err_index)} negative temp diffs: room{room_id} from home")

    room_coef_dict = {'kitchen': 3,
                       'livingroom': 5,
                       'bathroom': 3,
                       'bedroom': 4,
                       'hall': 5,
                       'outside': 3,
                       'utility': 3,
                       'cupboard': 3,
                       'other': 3,
                       'playroom': 4,
                       'study': 4,
                       'kitchenliving': 5,
                       'diningroom': 5,
                       'conservatory': 5.5}

    room_type = metadata.room.loc[metadata.room["roomid"] == room_id]["type"]
    roomheight=metadata.room.loc[metadata.room["roomid"] == room_id]["height"]
    roomarea=metadata.room.loc[metadata.room["roomid"] == room_id]["floorarea"]
    roomtype_coef = room_coef_dict[room_type.iloc[0]]
    ## below 1.09 is for north orientation or windows or.../(i transfere the action of next sentence to ann module)deviding3.41 turn btu to watt but I also turned it to KW so i wrote /3410 instead of /3.410
    rated_heat_output = np.round((1.09 * roomtype_coef * (roomheight.values * roomarea.values * 0.0355))/3.41, 2)

    result_df = pd.DataFrame(result_series * rated_heat_output).set_axis([radiator_name], axis=1)
#########
    return result_df[radiator_name]

with MetaDataStore() as s:
    metadata = MetaData(s)

def all_enhn_rads():
    """
    ######### For finding all "Radiator IDs of all enhanced homes"
    """
#### a = on_off_room(roomid,homeid)
    a = identify_io_enhanced()

    id_list = []

    for k in a.keys():
        for j in a[k].keys():
            for z in a[k][j]:
                id_list += list(z.values())

    pd.DataFrame(id_list).to_csv("radiator_list.csv")
    return pd.DataFrame(id_list)


def all_enhn_rooms():
    """
    ######### For finding all "ROOMs IDs of all enhanced homes"
    """
    a = identify_io_enhanced()

    id_list = []

    for k in a.keys():

        id_list += a[k].keys()

    pd.DataFrame(id_list).to_csv("room_list.csv")
    return pd.DataFrame(id_list)


def room_temp_sensors():
    """
    ######### For finding all "sensor IDs of all enhanced homes"
    """
    ####for alll enhanced homes==
    homes = metadata.combinedhome[metadata.combinedhome["install_type"] == "enhanced"]["homeid"]
    rooms = metadata.room.loc[metadata.room["homeid"].isin(homes)]["roomid"]

    boxes = metadata.sensorbox.loc[(metadata.sensorbox["roomid"].isin(rooms))
                                   & (metadata.sensorbox["sensorbox_type"] == "room")]["sensorboxid"]
    sensors = metadata.sensor.loc[(metadata.sensor["sensorboxid"].isin(boxes))
                                  & (metadata.sensor["type"] == "temperature")]
    sensors["sensorid"].to_csv(f"room_temp_sensors.csv")
    return sensors["sensorid"]