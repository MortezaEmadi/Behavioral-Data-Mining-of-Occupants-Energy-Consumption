"""
This script converts all of the sensor/radiator data from the timestamp format
to a df suitbale for Spark DataFrame format,and then uses FP_Growth  method from
pyspark library to generate Association Rules

"""

import pandas as pd
from radiator_methods import identify_io_enhanced
from data_preprocess import homeid
import findspark

findspark.init()
from pyspark.ml.fpm import FPGrowth
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import round, col

homeid = homeid
file_path = f"spark_input/onoffs_{homeid}_1401.csv"

all_enhanced = identify_io_enhanced()
homeid = homeid
home_rdata = all_enhanced[homeid]

df = pd.read_csv(file_path).set_index("time")


apps = {"kettle": 1, "dishwasher": 2,	"microwave": 3 ,"socket": 4, "toaster": 5, "electriccooker": 6,
        "electricshower": 7, "vacuumcleaner": 8, "dehumidifier": 9, "washingmachine": 10}

counter = 1

for rad_name in df.columns:
    if rad_name.split("_")[0] == "radiator":
        apps[rad_name] = int(str(homeid) + str(counter))
        counter += 1
    else:
        continue


#Todo!! Dropping column which have insufficient values in the values file!(if in prev modules it's skipped by mistake!)
### This one for home 162=
# df = df.drop("radiator_1621_15126",1)

radiator_number = len([x for x in apps.keys() if x.startswith("radiator")])
##shomare hameye radiator ha albate ++plus being ON! i mean it's 1 in the right!
radiators = [apps[x]*10+1 for x in apps.keys() if x.startswith("radiator")]

result = list()
indexed_result = dict()
indexed_result["time"] = []
indexed_result["data_index"] = []

data_index = 0

for index, row in df.iterrows():

    current_list = []

    for i, r in row.iteritems():

        if r == "on":

            current_list.append(int(str(apps[i])+"1"))
        if r == "off":

            current_list.append(int(str(apps[i])+"0"))
        ###Chon baraye antirule ye ghanuni ke consequntesh faghat 1 bude va ants hash hame 0,zamanhai ke ham vasayel 0 bashe mishe antirules esh
    bye="bye"
    if len([x for x in current_list if (x % 10) == 1]) == 0 :
        continue
    result.append((data_index, current_list))
    indexed_result["time"].append(index)
    indexed_result["data_index"].append(data_index)
    data_index += 1


index_df = pd.DataFrame(indexed_result)
index_df.to_csv(r"data\ann\index_rel_spark_home{}_1401.csv".format(homeid))



sc = SparkContext('local')
spark = SparkSession(sc)



df = spark.createDataFrame(result, ["id", "items"])

fpGrowth = FPGrowth(itemsCol="items",minSupport=0.01, minConfidence=0.08)
model = fpGrowth.fit(df)


rules = model.associationRules
rules2 = rules.where(rules.lift>'1')
rules2.count()


rules2 = rules2.select("*", round(col('confidence'),5))
rules2 = rules2.select("*", round(col('support'),5))
rules2 = rules2.select("*", round(col('lift'),5))
rules2 = rules2.drop('confidence')
rules2 = rules2.drop('support')
rules2 = rules2.drop('lift')
rules2 = rules2.withColumnRenamed("round(support, 5)","support")
rules2 = rules2.withColumnRenamed("round(confidence, 5)","confidence")
rules2 = rules2.withColumnRenamed("round(lift, 5)","lift")

rules_df = rules2.select("*").toPandas()

rules_df.to_csv(f'data/ann/home{homeid}_rules_1401_2.csv')