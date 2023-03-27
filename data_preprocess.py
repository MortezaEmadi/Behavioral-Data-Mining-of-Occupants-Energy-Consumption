"""
All Rights Reserved
@author Morteza Emadi
This script creates a cleaned dataset of electrical appliances
by aid of many other components/modules the project
"""
import argparse
import datetime as dt
import numpy as np
from metadata import MetaData
from data_store import MetaDataStore, HomeReadingStore
from electric_preprocess import ElecPreprocessor
homeid= 146  #next=136
if __name__ == "__main__" :
    homeid = homeid
    def make_df(name):
        name=pd.DataFrame()
        return df
    def valid_date(s):
        try:
            return dt.datetime.strptime(s, "%Y-%m-%d")
        except ValueError:
            msg = "Not a valid date: '{0}'.".format(s)
            raise argparse.ArgumentTypeError(msg)

    # get arguments
    parser = argparse.ArgumentParser(description='Clean electrical sensor readings and merge .')
    parser.add_argument('--home',type=int, default=[homeid], help='home to process, default all')
    #####*********************** TODO: deghat dar khate bala age run migiram hatman be 3ta manzele madenazar kahesh dahid,default ash -1 kole khane ha ast
    parser.add_argument('--enddate', type=valid_date, default=None, help='use only readings before this date')
    args = parser.parse_args()


    end_date = args.enddate

    with MetaDataStore() as s:
        metadata = MetaData(s)

    home_reading_store = HomeReadingStore()
    preprocessor = ElecPreprocessor()

    if isinstance(args.home, list):
        homeids = args.home
    elif isinstance(args.home, int) and args.home != -1:
        homeids = [args.home]
    else:
        homeids = metadata.gold_homes()


    for homeid in homeids:
        print('homeid: {0}'.format(homeid))
        ###mains_readings bayad az inja ham mains reading ra pas migerefti ha kolan ctrl+shift+F ro bzan ta harche mains mibini pak koni!
        zwave_readings,oem_readings = preprocessor.get_home_readings(homeid)
        if end_date is not None:
            zwave_readings = zwave_readings[zwave_readings.index < end_date]
            oem_readings = oem_readings[oem_readings.index < end_date]
            # mains_readings = mains_readings[mains_readings.index < end_date]
        # zwave_readings
        # oem_readings
        home_reading_store.set_readings(homeid, zwave_readings)
        home_reading_store.set_readings((int(str(8)+str(homeid))), oem_readings)

        # Since real nan of each device were ignored,we put an arbitrary value and now for values df
        # we need to restore them:
        zwave_readings = zwave_readings.replace(1234567.000001, np.nan)
        oem_readings = oem_readings.replace(1234567.000001, np.nan)
        readingsvalue_zwave = zwave_readings.merge(oem_readings, how="outer", left_index=True, right_index=True)
        readingsvalue_zwave.to_csv(f"ann/value_elecreadings_home{homeid}_1401.csv")

        # home_reading_store.set_readings((int(str(88)+str(homeid))), mains_readings)
        ############## ToDo!: please note all home-readings file with 8 as 1st num is for oem and double 88 for Mains!!
