from radiator_methods import identify_io_enhanced, on_off_room
import pandas as pd
from data_preprocess import homeid

def detector_radiator(homeid):

    all_enhanced = identify_io_enhanced()
    homeid = homeid
    home_rdata = all_enhanced[homeid]

    zwave_onoffs = pd.read_csv(f"on_offs\home{homeid}_onoff_zwave_1401.csv")
    zwave_onoffs = zwave_onoffs.set_index(pd.to_datetime(zwave_onoffs["time"], format="%Y-%m-%d %H:%M:%S"))
    zwave_onoffs = zwave_onoffs.drop(columns="time")
    oem_onoffs = pd.read_csv(f"on_offs\home{homeid}_onoff_oem_1401.csv")
    oem_onoffs = oem_onoffs.set_index(pd.to_datetime(oem_onoffs["time"], format="%Y-%m-%d %H:%M:%S"))
    oem_onoffs = oem_onoffs.drop(columns="time")

    elec_onoffs = zwave_onoffs.merge(oem_onoffs, how = "outer", left_index = True, right_index = True)

    # home_data = home_data.set_index("time")

    for room_id in home_rdata.keys():

        room_data = on_off_room(room_id,homeid)

        for sensor_couple in room_data.keys():

            sensor_data = room_data[sensor_couple]
            col_name = f"radiator_{room_id}_{sensor_couple[0]}"

            radiat_onoff = sensor_data.set_index("time").rename(
                columns={"switch_candidate": col_name})

            ####Todo: do this below trick for all other clean data gatherings! so if there is any problem make the df empty!s here it wont congatulate
            if not radiat_onoff.empty:
                elec_onoffs = elec_onoffs.merge(radiat_onoff, how="outer", left_index=True, right_index=True)
                print(f"on_offs of the {col_name} now extarcted and has been merged to the elec_onoff & the first time index was= {radiat_onoff.index[0]} and the lastindex of it={radiat_onoff.index[-1]}")



    # home_onoff=elec_onoffs.fillna("off")

    for app in elec_onoffs.columns:


        home_onoff = elec_onoffs[app]
        home_onoff = home_onoff.reset_index()
        minimum = home_onoff[home_onoff[app].notnull()].index.min()
        maximum = home_onoff[home_onoff[app].notnull()].index.max()


        first = home_onoff.loc[:minimum]
        second = home_onoff.loc[minimum + 1:maximum+1].fillna(method="ffill")
        third = home_onoff.loc[maximum + 2:]

        holder_onoff = pd.concat([first, second, third])
        elec_onoffs[app] = holder_onoff.set_index("time")
    # home_onoff=home_onoff.replace({'on': 1, 'off': 0})
    elec_onoffs.to_csv(f"spark_input/onoffs_{homeid}_1401.csv")






