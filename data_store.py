"""
All Rights Reserved
@author Morteza Emadi
used to help other modules for storing data/metadata
"""
import pandas as pd
import os

# LOCAL_DATA_DIR = os.environ['IDEAL_DATA_DIR']
#changed for github
local_temp_path = r'data\local_temp'
dataset_path = r'data\dataset'

class MetaDataStore(pd.HDFStore):
    """used to store metadata tables
        store = MetaDataStore()
        sensor = store['sensor']
    """
    FILE_NAME = 'metadata.h5'
####################*************$$$$$$$$$$$$$$$$$$$******************* Todo: inja pycharm error mide!!
    def __init__(self, data_dir=local_temp_path):
        """
        :param data_dir: directory where local data is stored
        :type data_dir: str
        """
        #######******************************************** todo: hala inja pycharm erore python ended mide
        super().__init__(os.path.join(data_dir, self.FILE_NAME))


class ReadingDataStore(pd.HDFStore):
    """Used to store readings from sensors. For example, Sensor readings for
     sensorid=1662 are stored under the key 'sensorid_1662'.
    """
    FILE_NAME = 'reading.h5'

    def __init__(self, data_dir=local_temp_path):
        super().__init__(os.path.join(data_dir, self.FILE_NAME))

    @staticmethod
    def sensorid2index(sensorid):
        return '/sensorid_' + str(sensorid)

    @staticmethod
    def index2sensorid(index):
        if "c" in index:
            return (index.split('_')[1])
        else:
            return int(index.split('_')[1])

    def get_sensorids(self):
        return [self.index2sensorid(index) for index in self.keys()]

    def get_sensor_readings(self, sensorid):
        """Get the readings for one sensor
        :param sensorid: sensor id
        :type sensorid: int
        :return: readings from sensor with sensorid
        :rtype: pandas.Dataframe
        """
        return self[self.sensorid2index(sensorid)]

    def set_sensor_readings(self, sensorid: object, readings: object) -> object:
        """Set the sensor readings for one sensor

        :param sensorid: sensor id
        :type sensorid: int
        :param readings: Datafraود/me containing sensor readings
        :type readings: pandas.Dataframe
        """
        self[self.sensorid2index(sensorid)] = readings

class HomeReadingStore(pd.HDFStore):
    """ Used to store ""processed"" electrical readings for each home to be used for
    dissagregation.
    """
    FILE_NAME = 'home-reading.h5'

    def  __init__(self, data_dir=local_temp_path):
        super().__init__(os.path.join(data_dir, self.FILE_NAME))

    @staticmethod
    def id2index(id):
        return 'homeid_' + str(id)

    @staticmethod
    def index2id(index):
        return int(index.split('_')[1])

    def get_ids(self):
        return [self.index2id(index) for index in self.keys()]

    def get_readings(self, id):
        """Get the readings for one home
        :param id: home id
        :type sensorid: int
        :return: readings from home with homeid
        :rtype: pandas.Dataframe
        """
        return self[self.id2index(id)]

    def set_readings(self, id, readings):
        """Set the readings for one home

        :param sensorid: home id
        :type sensorid: int
        :param readings: Dataframe containing home readings
        :type readings: pandas.Dataframe
        """
        self[self.id2index(id)] = readings