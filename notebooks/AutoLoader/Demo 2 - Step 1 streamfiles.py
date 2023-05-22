# Databricks notebook source
display(dbutils.fs.ls("dbfs:/databricks-datasets/iot-stream/data-device/"))
# lake_data_root_path = "abfss://datasets@storagemh1westeu.dfs.core.windows.net"
# dbutils.fs.cp("dbfs:/databricks-datasets/iot-stream/data-device/part-00000.json.gz", f"{lake_data_root_path}/temp/json")

# COMMAND ----------

import time

def insert_new_files(path):
    path = path if path.endswith("/") else path + "/"
        
    # Lop to copy existing 20 files we have to target (autolodaer inbox)
    for i in range(1, 20):
        i = str(i).zfill(2)
        target_path = str(f"{path}part-000{i}.json.gz")
        print(f"Inserted new file: {target_path}")
        dbutils.fs.cp(f"dbfs:/databricks-datasets/iot-stream/data-device/part-000{i}.json.gz", target_path)
        time.sleep(3)
        
    #Simulae we have more files by copying the same file as new target names over and over again
    for i in range(21, 71):
        i = str(i).zfill(2)
        target_path = str(f"{path}part-000{i}.json.gz")
        print(f"Inserted new file: {target_path}")
        dbutils.fs.cp("dbfs:/databricks-datasets/iot-stream/data-device/part-00019.json.gz", target_path)
        time.sleep(7)

# COMMAND ----------

insert_new_files(path="/tmp/iot_stream/")
