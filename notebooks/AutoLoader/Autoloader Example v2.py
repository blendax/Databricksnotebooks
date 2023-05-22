# Databricks notebook source
# MAGIC %md
# MAGIC ## The Problem
# MAGIC <br>
# MAGIC A very common case is to process files in a folder. When processing files we only want to process the same file once. There are different ways to solve this:
# MAGIC <br>
# MAGIC
# MAGIC - Process and then move/delete if successfull.
# MAGIC - Keeping a list of metadata of all processed files and other ways.
# MAGIC - React to file system events when a new file arrives and put the event on a queue that we consume
# MAGIC <br>
# MAGIC
# MAGIC Autoloader can use the last approach mentioned above combined with streaming and checkpoints to make things more easy.<br>
# MAGIC <br>
# MAGIC
# MAGIC **This notebook will show how you can use Autoloader and how to use the two different modes of Autoloader in Azure.**

# COMMAND ----------

# MAGIC %md
# MAGIC ###Auto Loader
# MAGIC Incrementally and efficiently processes new data files as they arrive in Azure Blob storage or Azure Data Lake Storage Gen2 without any additional setup. Auto Loader provides a new Structured Streaming source called *cloudFiles*. Given an input directory path on the cloud file storage, the cloudFiles source automatically processes new files as they arrive, with the option of also processing existing files in that directory.
# MAGIC
# MAGIC Docs
# MAGIC
# MAGIC
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/ingestion/auto-loader/<br>
# MAGIC https://databricks.com/blog/2020/02/24/introducing-databricks-ingest-easy-data-ingestion-into-delta-lake.html

# COMMAND ----------

# MAGIC %md
# MAGIC There are two types of file listening:
# MAGIC * Directory listening -  (for few files every day) - simple to get started with
# MAGIC * File notification - (For many files and directories) - using event grid and storage queues (requires extra permission setup)
# MAGIC
# MAGIC Specify:
# MAGIC
# MAGIC <code>.option("cloudFiles.useNotifications", "true")</code>
# MAGIC
# MAGIC to use File notification listening.
# MAGIC
# MAGIC You can change mode when you restart the stream. For example, you may want to switch to file notification mode when the directory listing is getting too slow due to the increase in input directory size. For both modes, Auto Loader internally keeps tracks of what files have been processed to provide exactly-once semantics, so you do not need to manage any state information yourself.

# COMMAND ----------

# MAGIC %md
# MAGIC Example synax:<br>
# MAGIC *If you have data coming only once every few hours, you can still leverage auto loader in a scheduled job using Structured Streaming’s Trigger.Once mode.*
# MAGIC ```python
# MAGIC df = spark.readStream.format("cloudFiles")
# MAGIC   .option(<cloudFiles-option>, <option-value>)
# MAGIC   .schema(<schema>)
# MAGIC   .load(<input-path>)
# MAGIC
# MAGIC df.writeStream.format("delta")
# MAGIC   .option("checkpointLocation", <checkpoint-path>)
# MAGIC   .start(<output-path>)
# MAGIC
# MAGIC # Example
# MAGIC val df = spark.readStream.format("cloudFiles")
# MAGIC      .option("cloudFiles.format", "json")
# MAGIC          .load("/input/path")
# MAGIC
# MAGIC df.writeStream.trigger(Trigger.Once)
# MAGIC          .format(“delta”)
# MAGIC          .start(“/output/path”)
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC For <code>cloudFiles-option</code> see:
# MAGIC https://docs.microsoft.com/en-us/azure/databricks/spark/latest/structured-streaming/auto-loader#configuration for options.

# COMMAND ----------

# MAGIC %md
# MAGIC Below exmaple is based on having an inbox folder with files in parquet format.
# MAGIC We will also write to a new location as we consume new files in the inbox. We will append data.

# COMMAND ----------

# Setup
base_path = "abfss://datasets@storagemh1westeu.dfs.core.windows.net/autoloader"
pathToInbox = f"{base_path}/inbox"
pathToOutputAppend = f"{base_path}/processed"
pathToSchemForPaqrquet = f"{base_path}/schema"
pathToCheckpoint = f"{base_path}/checkpoints"
spark.sql("SET spark.databricks.cloudFiles.schemaInference.enabled=true")

# COMMAND ----------

dbutils.fs.ls(pathToCheckpoint)

# COMMAND ----------

# Cleanup - workspace folders
dbutils.fs.rm(pathToInbox, True)
dbutils.fs.rm(pathToOutputAppend, True)
dbutils.fs.mkdirs(pathToOutputAppend)
# Only delete checkpoint first time (if needed)
dbutils.fs.rm(pathToCheckpoint, True)


# COMMAND ----------

from pyspark.sql.functions import lit, col, concat, substring

# Utility function to create new files in inbox
def createParquetFile(path, noRows = 10000):
  dfIds = spark.range(noRows) #.withColumn("IDAsString", concat(lit("Str="), col("id")))
  dfIds.coalesce(1).write.mode("append").parquet(path)

# COMMAND ----------

# Create a first file
createParquetFile(pathToInbox)

# COMMAND ----------

# Define source schema
# read schema from existing parquet files in data lake
dfSchema = spark.read.parquet(pathToInbox)
schemaParquet = dfSchema.schema
# or you can define you own schema by hand. (Faster to read schema from file if we have it though)

# COMMAND ----------

# Or manual schema
#from pyspark.sql.types import StructType, StructField, LongType

# schemaParquet = StructType([
#   StructField("id", LongType(), True)
#])
# schemaParquet


# COMMAND ----------

# MAGIC %md
# MAGIC cloudFiles.includeExistingFiles
# MAGIC
# MAGIC Type: Boolean
# MAGIC
# MAGIC Whether to include existing files in the stream processing input path or to only process new files arriving after initial setup. This option is evaluated only when you start a stream for the first time. Changing this option after restarting the stream has no effect.
# MAGIC Default value: true

# COMMAND ----------

# Start test with directory listening without extra permission setup
# Good for few files and folders on a regular basis
# We are monitoring folder pathToInbox for new files and folders

df = spark.readStream.format("cloudFiles")\
.option("cloudFiles.format", "parquet")\
.option("cloudFiles.includeExistingFiles", "true")\
.option("cloudFiles.useNotifications", "false")\
.schema(schemaParquet)\
.load(pathToInbox)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Spark is lazy

# COMMAND ----------

# The key here is to use .trigger(once=True) so that you can use the capabilities with streaming using checkpoints,
# but run the streaming in batches whenever you want based on a schedule.
# We are writing new data to pathToOutputAppend
df.writeStream\
.trigger(once=True)\
.format("delta")\
.outputMode("append")\
.option("checkpointLocation", pathToCheckpoint)\
.start(pathToOutputAppend)

# COMMAND ----------

display(dbutils.fs.ls(pathToOutputAppend))

# COMMAND ----------

spark.read.format("delta").load(pathToOutputAppend).count()

# COMMAND ----------

# Drop a file or 2 in the "inbox" folder and run the code again above (Trigger once)
# or run script below to copy

# COMMAND ----------

createParquetFile(pathToInbox)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Change to File notification
# MAGIC
# MAGIC Setp a Service Principal with the right permissions as explained here:
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/ingestion/auto-loader/file-notification-mode#--required-permissions-for-configuring-file-notification-for-adls-gen2-and-azure-blob-storage
# MAGIC
# MAGIC
# MAGIC When you need to scale up you can change to File Notification mode instead.<br>
# MAGIC Databricks will make the change easy and remember what files we already have read when using the Directory listening.<br>
# MAGIC
# MAGIC We now change from directory listening to File Notification mode.
# MAGIC This mode will use Event Grid to subscribe to event in the folder we chose. A Storage queue will be used to store the events.
# MAGIC This solution is more scalable, but requires some extra setup in Azure.
# MAGIC
# MAGIC You must provide the following authentication options only if you choose file notification mode (cloudFiles.useNotifications = true):
# MAGIC
# MAGIC |Authentication Option|Type|Default|Description|
# MAGIC |---------------------|----|-------|-----------|
# MAGIC |cloudFiles.connectionString|String|None|The connection string for the storage account, based on either account access key or shared access signature (SAS)|
# MAGIC |cloudFiles.resourceGroup|String|None|The Azure Resource Group under which the storage account is created|
# MAGIC |cloudFiles.subscriptionId|String|None|The Azure Subscription ID under which the resource group is created|
# MAGIC |cloudFiles.tenantId|String|None|The Azure Tenant ID under which the service principal is created|
# MAGIC |cloudFiles.clientId|String|None|The client ID or application ID, of the service principal|
# MAGIC |cloudFiles.clientSecret|String|None|The client secret of the service principal|
# MAGIC |cloudFiles.queueName|String|None|The URL of the Azure queue. If provided, the cloud files source directly consumes events from this queue instead of setting up its own Azure Event Grid and Queue Storage services. In that case, your cloudFiles.connectionString requires only read permissions on the queue.|
# MAGIC
# MAGIC If you don´t want to setup you need different Azure permissions listed here: https://learn.microsoft.com/en-us/azure/databricks/ingestion/auto-loader/file-notification-mode#permissions-azure

# COMMAND ----------



# COMMAND ----------

# SP/App ID from keyvault
appId = dbutils.secrets.get(scope = "databricks", key = "sp-autoloader-id")
# SP/app secret from key vault
appSecret = dbutils.secrets.get(scope = "databricks", key = "sp-autoloader-secret")

# COMMAND ----------

# NOTE IMPORTANT - Permissions
# In the portal:
# Add the SP via its name (in my case databricksneu) as "EventGrid EventSubscription Contributor" (on IAM on the subscription)
# Add the SP via its name (in my case databricksneu) as "Contributor" for the storage account where we want to listen for new files

# Databricks will create a "Event Grid System Topics" with a storage que as the endpoint (our account)

# Note: If you want to create our own topic and endpoint queue yourself you can give the name of the storage account and the queue instead of letting databricks create this for you.

# COMMAND ----------

createParquetFile(pathToInbox)

# COMMAND ----------

# MAGIC %md
# MAGIC See Azure specific settings here:
# MAGIC https://learn.microsoft.com/en-us/azure/databricks/ingestion/auto-loader/options#azure-specific-options
# MAGIC
# MAGIC Also make sure Event grid is registred as resource provider

# COMMAND ----------

# Get connection string from Key Vault for contributor right on storage
# conStr = dbutils.secrets.get(scope = "databricks", key = "datasetsneugen2-contributor-con-str")
subscriptionId = dbutils.secrets.get(scope = "databricks", key = "subscriptionid")
tenantId = dbutils.secrets.get(scope = "databricks", key = "tenantId")

df = (spark.readStream.format("cloudFiles")
    .option("cloudFiles.clientId", appId)
    .option("cloudFiles.clientSecret", appSecret)
    .option("cloudFiles.resourceGroup", "RG_storage_we")
    .option("cloudFiles.subscriptionId", subscriptionId)
    .option("cloudFiles.tenantId", tenantId)
    .option("cloudFiles.format", "parquet")
    .option("cloudFiles.includeExistingFiles", "true")
    .option("cloudFiles.useNotifications", "true")
    .schema(schemaParquet)
    .load(pathToInbox))

#.option("cloudFiles.connectionString", conStr)\

#df.writeStream.trigger(Trigger.Once)
df.writeStream\
.trigger(once=True)\
.format("delta")\
.outputMode("append")\
.option("checkpointLocation", pathToCheckpoint)\
.start(pathToOutputAppend)

# COMMAND ----------

# MAGIC %md
# MAGIC You can look at the back end of this in the portal:<br>
# MAGIC https://ms.portal.azure.com/#blade/HubsExtension/BrowseResource/resourceType/Microsoft.EventGrid%2FsystemTopics<br>
# MAGIC https://portal.azure.com/#view/Microsoft_Azure_EventGrid/EventGridCentralMenuBlade/~/system_topics<br>
# MAGIC too see that an "Event Grid System Topic" is created that will route the event to a StorageQueue. The names of the "Event Grid System Topic" and the StorageQueue are prefixed with databricks- so you can find them.
# MAGIC
# MAGIC **Event grid system topic**<br>
# MAGIC ![Event Grid System Topic](https://publicstoragemh.z16.web.core.windows.net/images/EventGridSystemTopic.png)
# MAGIC <br>
# MAGIC ---
# MAGIC **and the details of the Event Grid System Topic:**
# MAGIC ![Event Grid System Topic Details](https://publicstoragemh.z16.web.core.windows.net/images/EventGridSystemTopicDetails.png)
# MAGIC <br>
# MAGIC This topic is created by Databricks automatically using the contributors right on the Eventgrid. As it says in the docs above:<br>
# MAGIC - You can create your on Storage Queue and create your own topic. In that case Databricks only needs to have read access to the queue.

# COMMAND ----------

# Test to drop some new files in our inbox foler and re-run the above cell.

# COMMAND ----------

createParquetFile(pathToInbox, 500)

# COMMAND ----------

# Check if we got more data
spark.read.format("delta").load(pathToOutputAppend).count()
