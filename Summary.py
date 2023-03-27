"""
All Rights Reserved
@author Morteza Emadi
This module assists you in calculating the necessary parameters
 for use in the ranking process, which relies on the
 entropy-based TOPSIS method. (e.g, PerCapita Energy Usage, Mean Indoor/Outdoor Temp, etc.)

# Pay attention: if you want to see the summary results of all homes or homes in a
particular class you can change the 'homes' list for this purpose.
"""
from metadata import MetaData
from data_store import MetaDataStore, ReadingDataStore,dataset_path, local_temp_path
from data_preprocess import homeid
import pandas as pd
import glob

with MetaDataStore() as s:
    metadata = MetaData(s)

residents = pd.read_csv(r"data\clustering\residentsdata.csv").set_index('Unnamed: 0')
persons = residents.set_index('homes_number_order').T.to_dict('list')

allhomes = list(residents["homes_number_order"].values)

homes = [homeid]
all_processed = 500

exact_folder = r"household_sensors_main\sensordata\\"

for home_id in homes:
    gas_processed = {home_id: all_processed / 2}
    electric_processed = {home_id: all_processed / 2}

    #value/gas_processed[home_id] * 100

    # electric mains
    filenames = glob.glob(dataset_path + exact_folder + f'home{home_id}*electric-combined*.csv.gz')

    readings = pd.read_csv(filenames[0], compression='gzip', parse_dates=[0], names=['time', 'value'])

    end_date = readings.iloc[-1].time

    start_date = readings.iloc[0].time

    value = readings.loc[(readings["time"] >= start_date) & (readings["time"] <= end_date), "value"].mean()

    electric_value = value * ((end_date - start_date).total_seconds() / 3600000)

    #gas
    gas_file = glob.glob(dataset_path + exact_folder + f'home{home_id}*gas-pulse*.csv.gz')
    gas = pd.read_csv(gas_file[0], compression='gzip', parse_dates=[0], names=['time', 'value'])
    gas = gas.set_index("time", drop=True)
    gas = gas["value"]
    num=gas[(gas.index >= pd.to_datetime('2017-06-01 00:00:00', format='%Y-%m-%d %H:%M:%S')) & (
            gas.index <= pd.to_datetime('2017-06-30 23:59:59', format='%Y-%m-%d %H:%M:%S'))].sum()


    end_date = readings.iloc[-1].time

    start_date = readings.iloc[0].time

    value = readings.loc[(readings["time"] >= start_date) & (readings["time"] <= end_date), "value"].sum()

    gas_value = value / 1000


    # per capita usage

    per_person_value = (electric_value + gas_value) / (persons[home_id][0])
    per_person_electric = (electric_value ) / (persons[home_id][0])
    per_person_gas = ( gas_value) / (persons[home_id][0])
    absoulute_total_saving = (electric_processed[home_id] + gas_processed[home_id])


    # temp diff

    rooms = metadata.room.loc[metadata.room["homeid"] == home_id, "roomid"]

    boxes = metadata.sensorbox.loc[(metadata.sensorbox["roomid"].isin(rooms)) &
                                   (metadata.sensorbox["onMainThermostat"] == 1), "sensorboxid"]

    in_temp_sensors = metadata.sensor[(metadata.sensor["type"] == "temperature") &
                                      (metadata.sensor["sensorboxid"].isin(boxes))]

    if in_temp_sensors.empty:

        boxes = metadata.sensorbox.loc[(metadata.sensorbox["roomid"].isin(rooms)) &
                                       (metadata.sensorbox["sensorbox_type"] == "room"), "sensorboxid"]

        all_temp_sensors = metadata.sensor[(metadata.sensor["type"] == "temperature") &
                                          (metadata.sensor["sensorboxid"].isin(boxes))]["sensorid"]

        cumulative_sum = 0
        number_of_sensors = all_temp_sensors.size

        for sensor_id in all_temp_sensors:
            if sensor_id == 16569:
                continue
            with ReadingDataStore(data_dir=local_temp_path) as s:

                readings = s.get_sensor_readings(sensor_id)
            if readings.value.mean() >= 100:
                cumulative_sum += readings.value.mean()
                number_of_sensors -= 1

        in_temp_value = cumulative_sum / number_of_sensors

    else:

        sensor_id = in_temp_sensors.sensorid.iloc[0]

        with ReadingDataStore(data_dir=local_temp_path) as s:

            readings = s.get_sensor_readings(sensor_id)

        in_temp_value = readings.value.mean()

    filenames = glob.glob("mcdmprocessing\\" + f'GRA2*.csv')

    readings = pd.read_csv(filenames[0])

    value = readings.loc[readings["homeid"] == home_id]["temperature1_cold_term"]

    if value.empty:

        print(f"Home {home_id} doesn't have out temp value.")
        value = 0

    else:

        value = value.iloc[0]


    diff_temp =( in_temp_value - value)/10

    # print(f"_________________ \nfor home={home_id} \n gasperperson={per_person_gas} \n ElecperPerson={per_person_electric} \n diff_temp={diff_temp} \n residents={persons[home_id][0]} ")
    print(f"_________________________________________________________________ \n{home_id}\n{per_person_gas}\n{per_person_electric}\n{persons[home_id][0]}\n{diff_temp}")

b = 1

