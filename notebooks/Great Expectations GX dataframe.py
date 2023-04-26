# Databricks notebook source
# MAGIC %md
# MAGIC Follow this tutorial (the Spark Dataframe option)
# MAGIC https://docs.greatexpectations.io/docs/deployment_patterns/how_to_use_great_expectations_in_databricks

# COMMAND ----------

# MAGIC %md
# MAGIC Install the following libs in this notebook, if they are not installed on the cluster already
# MAGIC - great_expectations (mandatory)
# MAGIC - azure-storage-blob azure-identity azure-keyvault-secrets (optional - needed if we publish to static web)

# COMMAND ----------

# MAGIC %pip install great_expectations azure-storage-blob azure-identity azure-keyvault-secrets

# COMMAND ----------

# MAGIC %md
# MAGIC ### Parameters needed for this notebook

# COMMAND ----------

# A conection string to storage account hosting our static web
con_str_storage = dbutils.secrets.get(scope="databricks", key="storage-con-str-static-web-weu")

# Meta-data root directory in DBFS or mount
root_directory = "/dbfs/great_expectations_mh/"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Imports

# COMMAND ----------

# MAGIC %md
# MAGIC Take care of some imports that will be used later. (Dataframe usage in this case)<br>
# MAGIC I have choosen to use dataframes because I want GE to be independent of storage and files.<br>
# MAGIC We will use Databrick to load any filea nd then we will use GE to check the loaded data frame.<br>
# MAGIC This also enables us to use GE with Unity Catalopg.

# COMMAND ----------

import datetime

import pandas as pd

from great_expectations.core.batch import RuntimeBatchRequest
from great_expectations.util import get_context
from great_expectations.data_context.types.base import (
    DataContextConfig,
    FilesystemStoreBackendDefaults,
)

from great_expectations.core.yaml_handler import YAMLHandler
yaml = YAMLHandler()

# COMMAND ----------

# MAGIC %md
# MAGIC When you don't have easy access to a file system, instead of defining your Data Context via great_expectations.yml you can do so by instantiating a BaseDataContext with a config. Take a look at our how-to guide to learn more: How to instantiate a Data Context without a yml file. In Databricks, you can do either since you have access to a filesystem - we've simply shown the in code version here for simplicity.

# COMMAND ----------

# MAGIC %md
# MAGIC The `root_directory` here refers to the directory that will hold the data for your Metadata Stores (e.g. Expectations Store, Validations Store, Data Docs Store). We are using the FilesystemStoreBackendDefaults since DBFS acts sufficiently like a filesystem that we can simplify our configuration with these defaults. These are all more configurable than is shown in this simple guide, so for other options please see our "Metadata Stores" and "Data Docs" sections in the "How to Guides" for "Setting up Great Expectations."

# COMMAND ----------

# MAGIC %md
# MAGIC Data context is the container that holds everything together we git it the root_dir to store things
# MAGIC ![Data Context](https://docs.greatexpectations.io/assets/images/data_context_does_for_you-df2eca32d0152ead16cccd5d3d226abb.png)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Data Context
# MAGIC Let's create our Data Context:

# COMMAND ----------

data_context_config = DataContextConfig(
    store_backend_defaults=FilesystemStoreBackendDefaults(
        root_directory=root_directory
    ),
)
context = get_context(project_config=data_context_config)

# COMMAND ----------

data_context_config = DataContextConfig(
    ## Local storage backend
    store_backend_defaults=FilesystemStoreBackendDefaults(
        root_directory=root_directory
    ),
    ## Data docs site storage
    data_docs_sites={
        "az_site": {
            "class_name": "SiteBuilder",
            "store_backend": {
                "class_name": "TupleAzureBlobStoreBackend",
                "container":  "\$web",
                "connection_string":  con_str_storage,
            },
            "site_index_builder": {
                "class_name": "DefaultSiteIndexBuilder",
                "show_cta_footer": True,
            },
        }
    },
)
context = get_context(project_config=data_context_config)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Read data to dataframe
# MAGIC Now we read our Taxi 201901 CSV data into a spark dataframe<br>
# MAGIC To optimize future reads we rewrite the CSV to delta the first time.

# COMMAND ----------

import os
deltadir = "gxtaxidemodata201901"
if not os.path.exists(f"/dbfs/{deltadir}"):
    print("reading data from csv first time")
    df = spark.read.format("csv")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .load("/databricks-datasets/nyctaxi/tripdata/yellow/yellow_tripdata_2019-01.csv.gz")
    print("writing data to delta - first time")
    df.write.format("delta").save(f"dbfs:/{deltadir}", mode="overwrite")

print(f"Reading from delta: {deltadir}")
df = spark.read.format("delta").load(f"dbfs:/{deltadir}")

# COMMAND ----------

# MAGIC %md
# MAGIC We now want GX to be able to access our spark dataframe df

# COMMAND ----------

# MAGIC %md
# MAGIC First we define a new datasource using a config.<br>
# MAGIC We tell the DS to use Sparkb´<br>
# MAGIC **Note:** This is a `RuntimeDataConnector` which means that the data source does not define any specific assets.<br>
# MAGIC We will supply assets (a loaded spark data frame) when we run a batch request `RuntimeBatchRequest`
# MAGIC <br>**MH:** I think that we could have generic spark data source as we give the data frame later.

# COMMAND ----------

my_spark_datasource_config = {
    "name": "ds_spark_df",
    "class_name": "Datasource",
    "execution_engine": {"class_name": "SparkDFExecutionEngine"},
    "data_connectors": {
        "data_connector_spark_df": {
            "module_name": "great_expectations.datasource.data_connector",
            "class_name": "RuntimeDataConnector",
            "batch_identifiers": [
                "some_key_maybe_pipeline_stage",
                "some_other_key_maybe_run_id",
            ],
        }
    },
}

# COMMAND ----------

# MAGIC %md
# MAGIC We can check our defined Datasource with:

# COMMAND ----------

context.test_yaml_config(yaml.dump(my_spark_datasource_config))

# COMMAND ----------

# MAGIC %md
# MAGIC We add our defined spark data source `ds_spark_df` to our context so we can use it

# COMMAND ----------

context.add_datasource(**my_spark_datasource_config)

# COMMAND ----------

# MAGIC %md
# MAGIC To connect GX to our dataframe we need to create a RuntimeBatchRequest that connects the spark data frame to use. We will point to the real data frame `df` here.

# COMMAND ----------

batch_request = RuntimeBatchRequest(
    datasource_name="ds_spark_df",
    data_connector_name="data_connector_spark_df",
    data_asset_name="taxi_201901",  # This can be anything that identifies this data_asset for you
    batch_identifiers={
        "some_key_maybe_pipeline_stage": "prod",
        "some_other_key_maybe_run_id": f"my_run_name_{datetime.date.today().strftime('%Y%m%d')}",
    },
    runtime_parameters={"batch_data": df},  # Our dataframe goes here
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validator and expectations suite
# MAGIC First we create the suite and get a Validator:

# COMMAND ----------

expectation_suite_name = "taxi_expectations_suite"
context.add_or_update_expectation_suite(expectation_suite_name=expectation_suite_name)
# We need to give the batchrequest that point to the data so that the validator knows what data to use
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite_name=expectation_suite_name,
)

print(validator.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Expectations
# MAGIC Then we use the Validator to add a few Expectations:

# COMMAND ----------

validator.expect_column_values_to_not_be_null(column="passenger_count")

validator.expect_column_values_to_be_between(
    column="congestion_surcharge", min_value=0, max_value=1000
)


# COMMAND ----------

# MAGIC %md
# MAGIC Let's validate that the data is from 201901 as the CSV says so

# COMMAND ----------

# Change to 0.96 for success
validator.expect_column_values_to_be_between(column="tpep_pickup_datetime", min_value="2019-01-01", max_value="2019-01-31", mostly=0.97)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Save Expectation Suite
# MAGIC Finally we save our Expectation Suite (all of the unique Expectation Configurations from each run of validator.expect_*) to our Expectation Store:

# COMMAND ----------

validator.save_expectation_suite(discard_failed_expectations=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Check that our expectations suite is saved

# COMMAND ----------

# MAGIC %%bash
# MAGIC ls /dbfs/great_expectations_mh/expectations/

# COMMAND ----------

# MAGIC %md
# MAGIC Look at our Expectatios suite `taxi_expectations_suite.json`<br>
# MAGIC We can see that the validations we defines are stored in our suite.

# COMMAND ----------

# MAGIC %%bash
# MAGIC less /dbfs/great_expectations_mh/expectations/taxi_expectations_suite.json

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checkpoints

# COMMAND ----------

# MAGIC %md
# MAGIC To save validation results and take actions we need to create Checkpoints
# MAGIC ![Checkpiont picture](https://docs.greatexpectations.io/assets/images/how_a_checkpoint_works-10e7fda2c9013d98a36c1d8526036764.png)

# COMMAND ----------

yaml_config_ceckpoint = f"""
name: my_checkpoint  # This is populated by the CLI.
config_version: 1
class_name: SimpleCheckpoint
validations:
  - batch_request:
    expectation_suite_name: {expectation_suite_name}  # your already defined and stored expectation suite
"""

# data_asset_name: taxi_201901  # Update this value.
# data_connector_name: data_connector_spark_df  # Update this value.
# datasource_name: ds_spark_df  # Update this value.
# batch_identifiers:
        # some_key_maybe_pipeline_stage: prod
        # some_other_key_maybe_run_id: my_run_name_20230425

# COMMAND ----------

# Sanity check of our checkppoint config above
context.test_yaml_config(yaml_config=yaml_config_ceckpoint)

# COMMAND ----------

# This will store our checkpoint to our root dir for our data context
context.add_checkpoint(**yaml.load(yaml_config_ceckpoint))

# COMMAND ----------

# MAGIC %%bash
# MAGIC # look at our stored checkpoint
# MAGIC less /dbfs/great_expectations_mh/checkpoints/my_checkpoint.yml

# COMMAND ----------

# MAGIC %md
# MAGIC #### Run the checkpoint 
# MAGIC and look at the result

# COMMAND ----------

from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult
result: CheckpointResult = context.run_checkpoint(run_name="taxi_check", checkpoint_name="my_checkpoint", batch_request=batch_request)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Alternative checkpoint
# MAGIC We can just create a simple checkpoint that does not tell what expectation suite to run<br>
# MAGIC We give the expectation suite later when we run the checkpoint<br>

# COMMAND ----------

# We only give the checkpoint a name here (my_checkpoint_new)
yaml_config_ceckpoint_new = f"""
name: my_checkpoint_new  # This is populated by the CLI.
config_version: 1
class_name: SimpleCheckpoint
"""

# COMMAND ----------

# Sanity check of our checkppoint config above
context.test_yaml_config(yaml_config=yaml_config_ceckpoint_new)

# COMMAND ----------

# This will store our checkpoint to our root dir for our data context
context.add_checkpoint(**yaml.load(yaml_config_ceckpoint_new))

# COMMAND ----------



# COMMAND ----------

from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult
result: CheckpointResult = context.run_checkpoint(run_name="taxi_check", checkpoint_name="my_checkpoint_new", batch_request=batch_request, expectation_suite_name=expectation_suite_name)

# COMMAND ----------

# MAGIC %md
# MAGIC #### CheckpointResult

# COMMAND ----------

print(result)

# COMMAND ----------

# todo: write result to our general delta table

# COMMAND ----------

# Success?
if not result.success:
    # Do further check
    if result.list_validation_results()[0].get("results")[2].get("result").get("unexpected_percent") > 3.0:
        print("aj aj aj mer än 3% fel datum inte bra")

# COMMAND ----------

# List all checkpoints
context.list_checkpoints()

# COMMAND ----------

# Load checkpoint
context.get_checkpoint('my_checkpoint_new')

# COMMAND ----------


