-- Databricks notebook source
-- MAGIC %md
-- MAGIC 
-- MAGIC ### Creating Delta Lake Table using SQL
-- MAGIC For official docs see: https://docs.microsoft.com/en-us/azure/databricks/spark/latest/spark-sql/language-manual/sql-ref-syntax-ddl-create-table-datasource

-- COMMAND ----------

-- DROP DB and TABLE if the Exist to have a working demo
DROP TABLE IF EXISTS demodb.events;
DROP DATABASE IF EXISTS demodb;

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Set the location in the data lake where the delta table will be stored
-- MAGIC table_location = "/mnt/datasetsneugen2/delta/eventsexmaple"

-- COMMAND ----------

-- MAGIC %python
-- MAGIC # Also delete the directory where the files are located for the table
-- MAGIC # WARNING it´s recursive so make sure you really use the correct path for the table_location for the delete
-- MAGIC #dbutils.fs.rm(table_location, True)

-- COMMAND ----------

-- DBTITLE 1,List input data to read
-- MAGIC %fs
-- MAGIC ls /databricks-datasets/structured-streaming/events/

-- COMMAND ----------

-- DBTITLE 1,List input data to read option 2
-- MAGIC %python
-- MAGIC display(dbutils.fs.ls("/databricks-datasets/structured-streaming/events/"))

-- COMMAND ----------

-- DBTITLE 1,Take a look at one of the source data json files with shell
-- MAGIC %sh
-- MAGIC tail -n 10 /dbfs/databricks-datasets/structured-streaming/events/file-0.json

-- COMMAND ----------

-- DBTITLE 1,Create a database to be used
create database demodb

-- COMMAND ----------

-- DBTITLE 1,Read Databricks switch action dataset (many json files)
CREATE TABLE demodb.events
-- The format that we want to use for our stored table after we have created it
USING delta
-- Where in our data lake do we want to store our table
LOCATION '/mnt/datasetsneugen2/delta/eventsexmaple'
PARTITIONED BY (date)
AS
-- The source data for our table that will be copied and wtitte to the delta location
SELECT action, from_unixtime(time, 'yyyy-MM-dd') as date
FROM json.`/databricks-datasets/structured-streaming/events/`


-- COMMAND ----------

SELECT * FROM demodb.events

-- COMMAND ----------

-- DBTITLE 1,Query the table
SELECT count(*) FROM demodb.events

-- COMMAND ----------

-- DBTITLE 1,Visualize data
SELECT date, action, count(action) AS action_count FROM demodb.events GROUP BY action, date ORDER BY date, action

-- COMMAND ----------

-- DBTITLE 1,Generate historical data - original data shifted backwards 2 days
INSERT INTO demodb.events
SELECT action, from_unixtime(time-172800, 'yyyy-MM-dd') as date
FROM json.`/databricks-datasets/structured-streaming/events/`;

-- COMMAND ----------

-- DBTITLE 1,Count rows
SELECT count(*) FROM demodb.events

-- COMMAND ----------

-- DBTITLE 1,Visualize final data
SELECT date, action, count(action) AS action_count FROM demodb.events GROUP BY action, date ORDER BY date, action

-- COMMAND ----------

DESCRIBE EXTENDED demodb.events PARTITION (date='2016-07-25')

-- COMMAND ----------

OPTIMIZE demodb.events

-- COMMAND ----------

-- DBTITLE 1,Show table history
DESCRIBE HISTORY demodb.events

-- COMMAND ----------

-- DBTITLE 1,Show table details
DESCRIBE DETAIL demodb.events

-- COMMAND ----------

-- DBTITLE 1,Show the table format
DESCRIBE FORMATTED demodb.events
