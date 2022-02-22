# Databricks notebook source
# MAGIC %md # MLflow quickstart: inference
# MAGIC 
# MAGIC This notebook shows how to load a model previously logged to MLflow and use it to make predictions on data in different formats. The notebook includes two examples of applying the model:
# MAGIC * as a scikit-learn model to a pandas DataFrame
# MAGIC * as a PySpark UDF to a Spark DataFrame
# MAGIC   
# MAGIC ## Requirements
# MAGIC * This notebook requires Databricks Runtime 6.4 or above, or Databricks Runtime 6.4 ML or above. You can also use a Python 3 cluster running Databricks Runtime 5.5 LTS or Databricks Runtime 5.5 LTS ML.
# MAGIC * If you are using a cluster running Databricks Runtime, you must install MLflow. See "Install a library on a cluster" ([AWS](https://docs.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster)|[Azure](https://docs.microsoft.com/azure/databricks/libraries/cluster-libraries#--install-a-library-on-a-cluster)|[GCP](https://docs.gcp.databricks.com/libraries/cluster-libraries.html#install-a-library-on-a-cluster)). Select **Library Source** PyPI and enter `mlflow` in the **Package** field.
# MAGIC * If you are using a cluster running Databricks Runtime ML, MLflow is already installed.  
# MAGIC 
# MAGIC ## Prerequsite
# MAGIC * This notebook uses the ElasticNet models from MLflow quickstart part 1: training and logging ([AWS](https://docs.databricks.com/applications/mlflow/tracking-ex-scikit.html#training-quickstart)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking-ex-scikit#--training-quickstart)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking-ex-scikit.html#training-quickstart)).

# COMMAND ----------

# if not using cluster lib for mlflow run: %pip install mlflow

# COMMAND ----------

# MAGIC %md ## Find and copy the run ID of the run that created the model
# MAGIC 
# MAGIC Find and copy a run ID associated with an ElasticNet training run from the MLflow quickstart part 1: training and logging notebook. The run ID appears on the run details page; it is a 32-character alphanumeric string shown after the label "**Run**".  
# MAGIC 
# MAGIC To navigate to the run details page for the MLflow quickstart part 1: training and logging notebook, open that notebook and click **Experiment** in the upper right corner. The Experiments sidebar displays. Do one of the following:
# MAGIC 
# MAGIC * In the Experiments sidebar, click the icon at the far right of the date and time of the run. The run details page appears in a new tab. 
# MAGIC 
# MAGIC * Click the square icon with the arrow to the right of **Experiment Runs**. The Experiment page displays in a new tab. This page lists all of the runs associated with this notebook. To display the run details page for a particular run, click the link in the **Start Time** umn for that run. 
# MAGIC 
# MAGIC For more information, see "View notebook experiment" ([AWS](https://docs.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)|[Azure](https://docs.microsoft.com/azure/databricks/applications/mlflow/tracking#view-notebook-experiment)|[GCP](https://docs.gcp.databricks.com/applications/mlflow/tracking.html#view-notebook-experiment)).

# COMMAND ----------

run_id1 = "634e2a5bdc0a4ed584c9a0ff0efc994b" # replace with your own run_id
input_run_id = dbutils.widgets.get("run_id")
if(input_run_id):
  run_id1 = input_run_id
model_uri = "runs:/" + run_id1 + "/model"
print("Will use model uri:", model_uri)

# COMMAND ----------

import mlflow
# check tracking server is databricks
mlflow.get_tracking_uri()

# COMMAND ----------

# MAGIC %md ## Load the model as a scikit-learn model by RunId
# MAGIC Use the MLflow API to load the model from the MLflow server that was created by the run. After loading the model, you can use just like you would any scikit-learn model. 

# COMMAND ----------

# We can fetch and load the model by providing the run_id only
import mlflow.sklearn
model = mlflow.sklearn.load_model(model_uri=model_uri)
model.coef_

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

# MAGIC %md ## Create a PySpark UDF and use it for batch inference
# MAGIC In this section, you use the MLflow API to create a PySpark UDF from the model you saved to MLflow. For more information, see [Export a python_function model as an Apache Spark UDF](https://mlflow.org/docs/latest/models.html#export-a-python-function-model-as-an-apache-spark-udf).  
# MAGIC 
# MAGIC Saving the model as a PySpark UDF allows you to run the model to make predictions on a Spark DataFrame. 

# COMMAND ----------

# Create the PySpark UDF
import mlflow.pyfunc
pyfunc_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri)

# COMMAND ----------

# MAGIC %md Use the Spark function `withColumn()` to apply the PySpark UDF to the DataFrame and return a new DataFrame with a `prediction` column. 

# COMMAND ----------

from pyspark.sql.functions import struct

predicted_df = dataframe.withColumn("prediction", pyfunc_udf(struct('age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6')))
display(predicted_df)

# COMMAND ----------

# MAGIC %md #### Using "classic" UDF

# COMMAND ----------

# Can be optimized to load model less times
import mlflow.sklearn
# Or you can create your own prediction and customize fully with custom UDF
run_id1 = "634e2a5bdc0a4ed584c9a0ff0efc994b"
model_uri = "runs:/" + run_id1 + "/model"
model = mlflow.sklearn.load_model(model_uri=model_uri)

def my_cust_udf(age, sex, bmi, bp, s1, s2, s3, s4, s5, s6):  
  # We can fetch and load the model by providing the run_id only
  inputvals = [[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]]
  prediction = model.predict(inputvals)
  return prediction.tolist()[0]
  
pred = my_cust_udf(-0.103593093156339, 0.0506801187398187, 0.0616962065186885, 0.0218723549949558,-0.0442234984244464, -0.0348207628376986, -0.0434008456520269, -0.00259226199818282, 0.0199084208763183, -0.0176461251598052)
print(pred)

# COMMAND ----------

udf_cust_pred = udf(my_cust_udf)

# COMMAND ----------

predicted_cust_df = dataframe.withColumn("prediction_cust", udf_cust_pred('age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'))
display(predicted_cust_df.select('prediction_cust', 'age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6'))

# COMMAND ----------

# MAGIC %md #### Using Pandas UDF

# COMMAND ----------

from typing import Iterator, Tuple
from pyspark.sql.functions import pandas_udf

run_id1 = "634e2a5bdc0a4ed584c9a0ff0efc994b"
model_uri = "runs:/" + run_id1 + "/model"
model = mlflow.sklearn.load_model(model_uri=model_uri)

@pandas_udf("double")
# lite magic att spark vet vilen DF den jobbar på
# tänk på SQL functioner som bara tar en sträng som säger vilken kolumn den skall använda
# tänk att vi får kolumner istället för en kolumn (alltså namnen på dessa som features)
def predict_custom_cache(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
  for features in iterator:
    pdf = pd.concat(features, axis=1)
    yield pd.Series(model.predict(pdf))

prediction_df = dataframe.withColumn("prediction", predict_custom_cache(*dataframe.columns))
display(prediction_df)

# COMMAND ----------

# MAGIC %md #### Test with more data

# COMMAND ----------

df_iris_large_interference = spark.read.format("delta").load("abfss://datasets@datasetsneugen2.dfs.core.windows.net/parquet/iris_interference_100x")
df_iris_large_interference.count()

# COMMAND ----------

# MAGIC %md #### Pandas UDF with more data data

# COMMAND ----------

prediction_df = df_iris_large_interference.withColumn("prediction", predict_custom_cache(*dataframe.columns))
display(prediction_df)

# COMMAND ----------

# MAGIC %md #### Classic UDF more data

# COMMAND ----------

predicted_cust_df = df_iris_large_interference.withColumn("prediction", udf_cust_pred(*dataframe.columns))
display(predicted_cust_df)
