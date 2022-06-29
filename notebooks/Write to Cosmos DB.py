# Databricks notebook source
# Write to Comsos DB from a Spark dataframe
# See: https://docs.microsoft.com/en-us/azure/cosmos-db/sql/create-sql-api-spark

# COMMAND ----------

# Cosmos DB config

# Endpoint URI: https://REPLACEME.documents.azure.com:443/
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

# Use iris with added id column for test (or any other data you have)
from pyspark.sql.functions import monotonically_increasing_id, row_number, col, max
from pyspark.sql.window import Window
df_iris = spark.sql('select * from iris100x')
df_iris_with_increasing_id = df_iris.withColumn("monotonically_increasing_id", monotonically_increasing_id())
window = Window.orderBy(col('monotonically_increasing_id'))
df_iris_with_consecutive_increasing_id = df_iris_with_increasing_id.withColumn('increasing_id', row_number().over(window))
df_iris_with_id = df_iris_with_consecutive_increasing_id.select(col("increasing_id").alias("id"), *df_iris_with_consecutive_increasing_id.columns).drop("increasing_id, monotonically_increasing_id")

# COMMAND ----------

df_iris_id = spark.sql('select * from rawurlsdb237.iris_interference_100x_with_id')

# COMMAND ----------

# id needs to be a string
df_iris_id_str = df_iris_id.withColumn("id", col("id").cast("string"))

# COMMAND ----------

# Select a rande of the data (or all if you would like) that shall be written to Cosmos DB
df_range = df_iris_id_str.filter("id > 10 and id < 21")

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

# COMMAND ----------

#import com.microsoft.azure.cosmosdb.spark.schema._
#import com.microsoft.azure.cosmosdb.spark.CosmosDBSpark
# import com.microsoft.azure.cosmosdb.spark.config.Config
# Generate a simple dataset containing five values and
# write the dataset to Cosmos DB.
# val df = spark.range(5).select(col("id").cast("string").as("value"))

# 4000
(df_range.write.format("cosmos.oltp")
   .options(**cfg)
   .mode("APPEND")
   .save())

# COMMAND ----------

#import java.time.LocalTime;
#import java.time.temporal.ChronoUnit;
#import org.apache.spark.sql.functions.{col, udf, typedLit, lit, explode, explode_outer}
#import org.apache.spark.sql.types.IntegerType
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
