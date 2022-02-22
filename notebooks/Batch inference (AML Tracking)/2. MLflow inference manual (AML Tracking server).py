# Databricks notebook source
# MAGIC %md # MLflow batch inference when using Azure Machine Learning (AML) as tracking server
# MAGIC 
# MAGIC This notebook shows how to load a model previously logged via MLflow and use it to make predictions on data in different formats. The notebook includes three examples of applying the model in batch:
# MAGIC * as a scikit-learn model to a pandas DataFrame
# MAGIC * as mlflow pyfunc UDF
# MAGIC * as a classic PySpark UDF to a Spark DataFrame
# MAGIC * as a chached Pandas UDF using iterator
# MAGIC   
# MAGIC ## Requirements
# MAGIC * If you are using a cluster running Databricks Runtime, you must install MLflow. Use 'mlflow' python lib with %pip install mlflow or install mflow as lib on the cluster.
# MAGIC * If you are using a cluster running Databricks Runtime ML, MLflow is already installed.
# MAGIC * Create an Azure ML workspace and a Service Principal to use for access.
# MAGIC 
# MAGIC ## Prerequsite
# MAGIC * This notebook uses the ElasticNet models from "1. MLflow train and log (AML Tracking server)"

# COMMAND ----------

# Connect to AML WS using SP
import os
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

global ws
ws = None

def get_aml_ws(tenant_id, sp_id, sp_secret, subscription_id, resource_group, aml_workspace_name, force: bool = False) -> Workspace:
  global ws
  if(force or not ws):
    svc_pr = ServicePrincipalAuthentication(
       tenant_id=tenant_id,
       service_principal_id=sp_id,
       service_principal_password=sp_secret)

    ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=aml_workspace_name, auth=svc_pr)
    return ws
  else:
    return ws

# COMMAND ----------

import mlflow

# Config
sp_id = dbutils.secrets.get(scope="databricks", key="azureml-ws-databricks-sp-appid")
sp_secret = dbutils.secrets.get(scope="databricks", key="azureml-ws-databricks-sp-secret")
tenant_id = dbutils.secrets.get(scope="databricks", key="tenant-id-mh-sub")
subscription_id = dbutils.secrets.get(scope="databricks", key="subscriptionId")
resource_group = "Databricks"
aml_workspace_name = "azureml-ws-databricks"

# Authenticate AML WS via SP
ws = get_aml_ws(tenant_id=tenant_id, sp_id=sp_id, sp_secret=sp_secret, subscription_id=subscription_id, resource_group=resource_group, aml_workspace_name=aml_workspace_name)
ws.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())
print("Using AML workspace {} at location {}".format(ws.name, ws.location))

# COMMAND ----------

# MAGIC %md ## Find and copy the run ID of the run that created the model
# MAGIC 
# MAGIC Find and copy a run ID associated with an ElasticNet training run from the '1. MLflow train and log (AML Tracking server)' notebook. The run ID can be found by searching the experiment or look in the AML WS UI.  
# MAGIC 
# MAGIC Reach you AML workspaces list see: ([List AML WS in portal](https://ms.portal.azure.com/#blade/HubsExtension/BrowseResource/resourceType/Microsoft.MachineLearningServices%2Fworkspaces)).

# COMMAND ----------

run_id1 = "0babbc78-6ba6-4ff1-833e-2cc9ad5c6b65" # replace with your run_id
input_run_id = dbutils.widgets.get("run_id")
if(input_run_id):
  run_id1 = input_run_id
model_uri = "runs:/" + run_id1 + "/model"
print("Will use model uri:", model_uri)

# COMMAND ----------

mlflow.get_tracking_uri()

# COMMAND ----------

# MAGIC %md ## Load the model as a scikit-learn model by RunId
# MAGIC Use the MLflow API to load the model from the MLflow server (AML) that was created by the run. After loading the model, you can use just like you would any scikit-learn model. 

# COMMAND ----------

# We can fetch and load the model by providing the run_id only
import mlflow.sklearn
model = mlflow.sklearn.load_model(model_uri=model_uri)
model.coef_

# COMMAND ----------

# download model from run to /dbfs/ temp dir (in case you want to install dependencies via %pip)
from mlflow.tracking.client import MlflowClient
dbutils.fs.mkdirs("dbfs:/downloads/temp")
MlflowClient().download_artifacts(run_id=run_id1, path="model", dst_path="/dbfs/downloads/temp")
dbutils.fs.ls("dbfs:/downloads/temp/model")

# COMMAND ----------

# Import required libraries
from sklearn import datasets
import numpy as np
import pandas as pd

# Load diabetes datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame for sklearn ElasticNet linear_model
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

# COMMAND ----------

# For the purposes of this example, create a small Spark DataFrame. This is the original pandas DataFrame without the label column.
dataframe = spark.createDataFrame(data.drop(["progression"], axis=1))

# COMMAND ----------

# Get a prediction for a row of the dataset
model.predict(data[0:1].drop(["progression"], axis=1))

# COMMAND ----------

# MAGIC %md ## Create a mlflow pyfunc PySpark UDF and use it for batch inference
# MAGIC In this section, you use the MLflow API to create a PySpark UDF from the model you saved to MLflow. For more information, see [Export a python_function model as an Apache Spark UDF](https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf).  
# MAGIC 
# MAGIC Saving the model as a PySpark UDF allows you to run the model to make predictions on a Spark DataFrame. 

# COMMAND ----------

# Create the PySpark UDF
import mlflow.pyfunc
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md Use the Spark function `withColumn()` to apply the PySpark UDF to the DataFrame and return a new DataFrame with a `prediction` column. 

# COMMAND ----------

from pyspark.sql.functions import struct

# OK with AML tracking server
predicted_df = dataframe.withColumn("prediction", pyfunc_udf(struct('age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6')))
display(predicted_df)

# COMMAND ----------

# MAGIC %md #### Using "classic" UDF

# COMMAND ----------

import mlflow.sklearn

# Load model before UDF to make sure it is cached
uri_runid_aml = run_id1 # replace with your run_id

model_uri = "runs:/" + uri_runid_aml + "/model"
model = mlflow.sklearn.load_model(model_uri)
model_bc = spark.sparkContext.broadcast(model)

# Or you can create your own prediction and customize fully with custom UDF
def my_cust_udf(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):  
  inputvals = [[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]]
  model = model_bc.value
  prediction = model.predict(inputvals)
  return prediction.tolist()[0]
  
# Local call test of function
pred = my_cust_udf(-0.103593093156339, 0.0506801187398187, 0.0616962065186885, 0.0218723549949558,-0.0442234984244464, -0.0348207628376986, -0.0434008456520269, -0.00259226199818282, 0.0199084208763183, -0.0176461251598052)
print(pred)

# COMMAND ----------

# wrap in UDF
from pyspark.sql.functions import udf
udf_cust_pred = udf(my_cust_udf)

# COMMAND ----------

# Use classic UDF
predicted_cust_df = dataframe.withColumn("prediction_cust", udf_cust_pred('age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'))
display(predicted_cust_df)

# COMMAND ----------

# MAGIC %md #### Using Pandas UDF with Iterator[pd.Series]
# MAGIC Will only have to load model once for all series

# COMMAND ----------

from typing import Iterator, Tuple
from pyspark.sql.functions import pandas_udf
import pandas as pd
import mlflow

# Load model before UDF to make sure it is chached
uri_runid_aml = "0babbc78-6ba6-4ff1-833e-2cc9ad5c6b65"
# uri_runid_dbx = "634e2a5bdc0a4ed584c9a0ff0efc994b"
model_uri = "runs:/" + uri_runid_aml + "/model"
model = mlflow.sklearn.load_model(model_uri)
# broadcast model
model_bc = spark.sparkContext.broadcast(model)
  
@pandas_udf("double")
def predict_custom_cache(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
  model_in_udf = model_bc.value
  for features in iterator:
    pdf = pd.concat(features, axis=1)
    yield pd.Series(model_in_udf.predict(pdf))

prediction_df = dataframe.withColumn("prediction", predict_custom_cache(*dataframe.columns))
display(prediction_df)

# COMMAND ----------

# MAGIC %md #### Test with more data

# COMMAND ----------

df_iris_large_interference = spark.read.format("delta").load("abfss://datasets@datasetsneugen2.dfs.core.windows.net/parquet/iris_interference_100x")

# COMMAND ----------

# MAGIC %md #### Pandas UDF 353k rows

# COMMAND ----------

prediction_df = df_iris_large_interference.withColumn("prediction", predict_custom_cache(*dataframe.columns))
display(prediction_df)

# COMMAND ----------

# MAGIC %md #### Classic UDF 353k rows

# COMMAND ----------

predicted_cust_df = df_iris_large_interference.withColumn("prediction_cust", udf_cust_pred('age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'))
display(predicted_cust_df)
