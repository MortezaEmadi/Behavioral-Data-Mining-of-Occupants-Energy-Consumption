"""
after running generatecleaned nilm data
"""
import pandas as pd
import easygui
from metadata import MetaData
from data_store import MetaDataStore
from radiator_methods import identify_io_room,meanvalue_radiators
from data_preprocess import homeid

with MetaDataStore() as s:
    metadata = MetaData(s)
####Todo! Important parametres==>
homeid= homeid
sample_rate = 180




def merge_values(homeid):
    """
    this first finds all of the rooms of the homeid you input it,then uses the above method(identify io room) to
    store all the sesnor ids of radiators of each rooom in results(so all roooms of one home)
    :param homeid:
    :return:
    """
    rooms = metadata.room.loc[metadata.room["homeid"] == homeid]
    elec_valuedf = pd.read_csv(f"ann/value_elecreadings_home{homeid}_1401.csv")
    elec_valuedf = elec_valuedf.set_index(pd.to_datetime(elec_valuedf["time"], format="%Y-%m-%d %H:%M:%S"))
    elec_valuedf = elec_valuedf.drop(columns="time")
    roomnum=0
    result = pd.DataFrame()
    for index, room in rooms.iterrows():
        roomnum += 1
        room_id = room["roomid"]
        data = meanvalue_radiators(room_id,homeid)
        for k in data.keys():
            datum = data[k]
            datum=datum.rename(columns={"value":f"radiator_{room_id}_{k[0]}"})

            result = result.merge(datum, how = "outer", left_index = True, right_index = True)
    value_df = elec_valuedf.merge(result, how="outer", left_index=True, right_index=True)
    print(f"there were {roomnum}rooms in home{homeid} and {len(value_df.columns)-len(elec_valuedf.columns)}radiators were added")
    return value_df
        # col_name = f"radiator_{room_id}_{sensor_couple[0]}"


if __name__ == '__main__':
    from detect_zwave import detector_zwave
    from detect_oem import detector_oem
    from detect_radiator import detector_radiator
    value_df=merge_values(homeid)
    value_df.to_csv(f"data/ann/values_home{homeid}_1401.csv")

    detector_zwave(homeid)
    detector_oem(homeid)
    detector_radiator(homeid)
