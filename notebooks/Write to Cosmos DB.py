# Databricks notebook source
# MAGIC %md
# MAGIC # Write to Cosmos DB from Spark and Databricks
# MAGIC - Simple types
# MAGIC - Nested struct types

# COMMAND ----------

# Write to Comsos DB from a Spark dataframe
# See: https://docs.microsoft.com/en-us/azure/cosmos-db/sql/create-sql-api-spark

# COMMAND ----------

# MAGIC %md ### Cosmos connection config
# MAGIC Use keyvault not to expose any secrets

# COMMAND ----------

# Cosmos DB config

# Endpoint URI exmaple: https://REPLACEME.documents.azure.com:443/
cosmosEndpoint = dbutils.secrets.get("databricks", "cosmosdbmhne-endpoint")
# Key to conect to Cosmos DB
cosmosMasterKey = dbutils.secrets.get("databricks", "cosmosdbmhne-key")
# Cosmos DB name
cosmosDatabaseName = "sparktest"
# Cosmos DB container name
cosmosContainerName = "sparkwrite"

cfg = {
  "spark.cosmos.accountEndpoint" : cosmosEndpoint,
  "spark.cosmos.accountKey" : cosmosMasterKey,
  "spark.cosmos.database" : cosmosDatabaseName,
  "spark.cosmos.container" : cosmosContainerName,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Simple types to Cosmos DB

# COMMAND ----------

import pandas as pd
csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
iris = pd.read_csv(csv_url, header = None)
df_iris = spark.createDataFrame(iris)

# COMMAND ----------

# Use iris with added id column for test (or any other data you have)
from pyspark.sql.functions import monotonically_increasing_id, row_number, col, max
from pyspark.sql.window import Window

df_iris_with_increasing_id = df_iris.withColumn("monotonically_increasing_id", monotonically_increasing_id())
window = Window.orderBy(col('monotonically_increasing_id'))
df_iris_with_consecutive_increasing_id = df_iris_with_increasing_id.withColumn('increasing_id', row_number().over(window))
df_iris_with_id = df_iris_with_consecutive_increasing_id.select(col("increasing_id").alias("id"), *df_iris_with_consecutive_increasing_id.columns).drop("increasing_id, monotonically_increasing_id")

# COMMAND ----------

# id needs to be a string
df_iris_id_str = df_iris_id.withColumn("id", col("id").cast("string"))

# COMMAND ----------

# Select a range of the data (or all if you would like) that shall be written to Cosmos DB
df_range = df_iris_id_str.filter("id > 10 and id < 21")

# COMMAND ----------

# Write to Cosmos DB
(df_range.write.format("cosmos.oltp")
   .options(**cfg)
   .mode("APPEND")
   .save())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write nested type to Cosmos DB

# COMMAND ----------

from pyspark.sql.types import ArrayType, IntegerType

def create_list():
  # Create a list in a range of 10-20
  my_list = [*range(1, 10, 1)]
  return my_list

createlist_udf = udf(create_list, ArrayType(IntegerType()))

df_range_with_list = df_range.withColumn("listofids", createlist_udf())

display(df_range_with_list)

# COMMAND ----------

(df_range_with_list.write.format("cosmos.oltp")
   .options(**cfg)
   .mode("APPEND")
   .save())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Nested Struct (1 to many) to Cosmos DB
# MAGIC Generate nested sample test data

# COMMAND ----------

# Example of writing one to many relationship to Cosmos DB as nested dataframe 

# Data structure

# client1
#    |
#     -- person1 
#     -- person2
# client2
#    |
#    -- person3
# ...


# Test data
# client is top entity
client = sc.parallelize([["1", "Computer Company", "Ottawa", "Nice client"], 
                         ["2", "Cosmetics Company", "Düsseldorf", "large"], 
                         ["3", "Manufacturing Company", "Bengtsfors", "small efficient"]]).toDF(("client_id", "client_name", "client_city", "client_ note"))

# person is child entity with 0-∞ persones per client
person = sc.parallelize([["pid1", "Otto", "New York", "Otto is fine", "1"], 
                         ["pid2", "Bella", "Paris", "Bella is angry", "1"], 
                         ["pid3", "Signe", "London", "Signe is sleepy", "2"]]).toDF(("person_id", "person_name", "city", "note", "client_id"))

from pyspark.sql.functions import collect_list, struct
# Group the person data by client_id and create a list of remaining columns that you want in the child record of the person
grouped_person = person.groupBy("client_id").agg(collect_list(struct("person_id", "person_name", "city", "note")).alias("persons"))

# Join the client and groupedPerson Data
nested_df = client.join(grouped_person, "client_id", "left")

# cosmos needs id column of type string
nested_df = nested_df.withColumn("id", nested_df.client_id)

display(nested_df)

# COMMAND ----------

# DBTITLE 1,Write (append) to Cosmos DB
# Write nested dataframe to a Cosmos DB container "sparknested" (create before write)
# For all config options see: https://github.com/Azure/azure-sdk-for-java/blob/main/sdk/cosmos/azure-cosmos-spark_3_2-12/docs/configuration-reference.md#write-config
cfg_nested = {
  "spark.cosmos.accountEndpoint" : cosmosEndpoint,
  "spark.cosmos.accountKey" : cosmosMasterKey,
  "spark.cosmos.database" : cosmosDatabaseName,
  "spark.cosmos.container" : "sparknested",
}

(nested_df.write.format("cosmos.oltp")
   .options(**cfg_nested)
   .mode("APPEND")
   .save())

nested_df.createOrReplaceTempView("tempperson")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Write data to storage for other testing

# COMMAND ----------

# MAGIC %sql
# MAGIC -- select to_json(map(1, 'a', 2, 'b', 3, DATE '2021-01-01'));
# MAGIC select id, client_id, client_name, client_city, `client_ note` as client_note, to_json(persons) as person from tempperson

# COMMAND ----------

df_person_as_str = spark.sql("select id, client_id, client_name, client_city, `client_ note` as client_note, to_json(persons) as person from tempperson")

# COMMAND ----------

# Write parquet
path = "dbfs:/temp/person/parquet"
# Small sample data so make sure we have only 1 file
df_person_as_str.coalesce(1).write.parquet(path)
# download using CLI with:
# databricks fs cp -r dbfs:/temp/person/parquet . --profile X

# COMMAND ----------

# write csv
path = "dbfs:/temp/person/csv"
df_person_as_str.coalesce(1).write.csv(path, sep='|')
# download using CLI with (where X is your profile if not default then you can skip -profile): 
# databricks fs cp -r dbfs:/temp/person/csv . --profile X

# COMMAND ----------

# MAGIC %md #### appendix

# COMMAND ----------

# Using nested tuples to nest enteties when creating a Spark Dataframe
# Similar to Case Classes in Scala
from collections import namedtuple
Child = namedtuple("child", ["id_c", "name", "age"])
Parent = namedtuple("parent", ["id", "name", "age", "child_rec"])
child = Child(2, "Stine", 4)
parent = Parent(1, "Otto", 45, child) 
df_from_nested_tuple = spark.createDataFrame([parent])
display(df_from_nested_tuple)
