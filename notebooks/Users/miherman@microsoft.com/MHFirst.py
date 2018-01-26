# Databricks notebook source
1+1

# COMMAND ----------

myDataFrame = sc.parallelize([('a', 1), ('b', 2), ('c', 3)]).toDF()
display(myDataFrame)

# COMMAND ----------

firstDataFrame = sqlContext.range(1000000)

# The code for 2.X is
# spark.range(1000000)
print firstDataFrame

# COMMAND ----------

# An example of a transformation
# select the ID column values and multiply them by 2
secondDataFrame = firstDataFrame.selectExpr("(id * 2) as value")

# COMMAND ----------

# an example of an action
# take the first 5 values that we have in our firstDataFrame
print firstDataFrame.take(5)
# take the first 5 values that we have in our secondDataFrame
print secondDataFrame.take(5)

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/Rdatasets/data-001/datasets.csv

# COMMAND ----------

dataPath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"
diamonds = spark.read.format("com.databricks.spark.csv")\
  .option("header","true")\
  .option("inferSchema", "true")\
  .load(dataPath)
  
# inferSchema means we will automatically figure out column types 
# at a cost of reading the data more than once

# COMMAND ----------

display(diamonds)

# COMMAND ----------

df1 = diamonds.groupBy("cut", "color").avg("price") # a simple grouping

df2 = df1\
  .join(diamonds, on='color', how='inner')\
  .select("`avg(price)`", "carat")
# a simple join and selecting some columns

# COMMAND ----------

df2.explain()

# COMMAND ----------

df2.count()

# COMMAND ----------

df2.cache()

# COMMAND ----------

df2.count()

# COMMAND ----------

df2.count()

# COMMAND ----------

