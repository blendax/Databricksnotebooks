# Databricks notebook source
1+1

# COMMAND ----------

from pyspark.sql.functions import lit,col,concat, substring
NO_OF_ROWS = 100000
dfIds = spark.range(NO_OF_ROWS).withColumn("IDAsString", concat(lit("Str="), col("id")))
display(dfIds.limit(5))

# COMMAND ----------

path = "/tmp/parquet/10MIDs"
dfIds.write.mode("Overwrite").parquet(path)

# COMMAND ----------

dfBeforeAppend = spark.read.parquet(path)
print("No Rows before append:", dfBeforeAppend.count())

# COMMAND ----------

# MAGIC %sql
# MAGIC -- DROP IF EXIST
# MAGIC DROP TABLE IF EXISTS IDS;
# MAGIC -- Skapa en tabell som pekar på PARQUET
# MAGIC CREATE TABLE IDS
# MAGIC     USING parquet
# MAGIC     OPTIONS (
# MAGIC       path "/tmp/parquet/10MIDs"
# MAGIC     )

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM IDS

# COMMAND ----------

from pyspark.sql import functions as f
# Append Rows
NO_APPEEND_NEW_ROWS = 10000
dfAppendRows = spark.range(NO_OF_ROWS + NO_APPEEND_NEW_ROWS).filter("id >= {rows}".format(rows=NO_OF_ROWS)).withColumn("IDAsString", f.concat(f.lit("Str="), f.col("id")))
dfAppendRows.write.mode("Append").parquet(path)
dfAfterAppend = spark.read.parquet(path)
print("NoROws after", dfAfterAppend.count())

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM IDS

# COMMAND ----------

# MAGIC %sql
# MAGIC REFRESH TABLE IDS

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM IDS

# COMMAND ----------

# MAGIC %md
# MAGIC # Append data to partitions using parquet

# COMMAND ----------

# Create a col to partition on
# Just take last digit in id as partition id
dfId = spark.read.parquet(path)
dfId3 = dfId.withColumn("PartitionId", f.substring(f.col("id"), -1, 1))
# substring(column, -1, 1)
display(dfId3.limit(7))

# COMMAND ----------

# Write IDS partitioned on last digit
pathPar = "/tmp/parquet/IDSPartitioned/"
dfId3.write.partitionBy("PartitionId").mode("Overwrite").parquet(pathPar)

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS IDPARTITIONED;
# MAGIC -- Create table on top of partitioned data
# MAGIC CREATE TABLE IDPARTITIONED (id long, IDAsString string, PartitionId int)
# MAGIC     USING parquet
# MAGIC     OPTIONS (
# MAGIC       path "/tmp/parquet/IDSPartitioned/"
# MAGIC     )
# MAGIC     partitioned by (PartitionId)
# MAGIC     -- When the table schema is not provided, schema and partition columns will be inferred

# COMMAND ----------

# MAGIC %sql
# MAGIC MSCK REPAIR TABLE IDPARTITIONED

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*), PartitionId from IDPARTITIONED group by PartitionId order by PartitionId limit 7

# COMMAND ----------

# MAGIC %fs
# MAGIC ls /mnt/datasetsneugen2/parquet/IDSPartitioned/

# COMMAND ----------

dfReadPar = spark.read.parquet(pathPar)
display(dfReadPar.groupBy("PartitionId").count())

# COMMAND ----------

# filter out ids with partition 0 and 1 and append to original parquet
dfFilter01 = dfId3.filter("PartitionId == 0 or PartitionId == 1")
# append these rows to original parquet
dfFilter01.write.partitionBy("PartitionId").mode("Append").parquet(pathPar)
# Test result
dfReadPar = spark.read.parquet(pathPar)
display(dfReadPar.groupBy("PartitionId").count())

# COMMAND ----------

# MAGIC %sql
# MAGIC -- check table
# MAGIC select count(*), PartitionId from IDPARTITIONED group by PartitionId order by PartitionId

# COMMAND ----------

# MAGIC %sql
# MAGIC MSCK REPAIR TABLE IDPARTITIONED;
# MAGIC select count(*), PartitionId from IDPARTITIONED group by PartitionId order by PartitionId
# MAGIC -- We should now be able to see the changes

# COMMAND ----------

# MAGIC %md
# MAGIC #What if we tried to update specific rows?
# MAGIC Parquet does not allow updates. That means we would have some options:
# MAGIC - Rewritee all data
# MAGIC - Read exisitng data and create a new updated version of the data and rewrite all data
# MAGIC - FIgure out in what partition we want to change and do the above for only the partition
# MAGIC 
# MAGIC We have our data partitioned on the Partition ID.

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from IDPARTITIONED where id < 30 limit 7

# COMMAND ----------

# What if we would like to update the row where id = 50013 or id = 50033 and we want to add a string fixed to the IDAsString
# Both are in partition 3 so we can read the partion 3 only and write the update back

dfPartition = spark.sql("select * from IDPARTITIONED where PartitionId = 3")
dfPartition.show(5)

# COMMAND ----------

# Some logic for update
dfChange = dfPartition.filter("id = 50013 or id = 50033")
dfChange.show()

dfFixed = dfChange.withColumn("IDAsString", f.concat(f.col("IDAsString"), f.lit("_fixed")))
dfFixed.show()

# COMMAND ----------

# Take original data and put chnages in and write back
# Here dropping rows that should be updated and then adding by union. (Remember there is no update - a DF is immuatble)
# Get a new DF with the rows that should be updated
dfFiltered = dfPartition.filter("id != 50033 and id != 50013")
# Add the updated rows to a new DF
dfUpdatedDF = dfFiltered.union(dfFixed)
display(dfUpdatedDF.filter("id > 50012").orderBy("id").limit(7))

# COMMAND ----------

# Overwrite the correct parition with our new DF containing all data fro partion 3
dfUpdatedDF.write.mode("overwrite").parquet(pathPar + "PartitionId={partion_name}/".format(partion_name="3"))

# COMMAND ----------

# The rows in partitio 3 are updated in our final table.
display(spark.read.parquet(pathPar).filter("id < 50015 and id > 50011 or id < 50034 and id > 50032").orderBy("id"))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Note - It´s easier to append data as you can write append to a specific partition as we showed above. The updates are harder as you are forces to rewrite at least the full partition.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Note 2 - DELTA - By using the delta file format you can do upserts into different partitions much easier and not risking ovwerwriting your data by using time travel and ACID transactions.
# MAGIC Also table meta-data will be updated and in sync when using delta.
