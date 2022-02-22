# Databricks notebook source
# MAGIC %md
# MAGIC ## Experiment, train and log with MLFLow using Azure Machine Learning as tracking server
# MAGIC - Install dependencies with pip
# MAGIC - Create Experiment
# MAGIC   - Set name
# MAGIC   - Set locstion for artifact
# MAGIC   - Set location for experiment in WS
# MAGIC - Train
# MAGIC   - Log parameters and metrics
# MAGIC   - Log images
# MAGIC   - Log model in experiment
# MAGIC   - Save model to custom location (not needed really as we have model in experiment)

# COMMAND ----------

# Connect to AML WS using SP
import os
from azureml.core import Workspace
from azureml.core.authentication import ServicePrincipalAuthentication

global ws
ws = None
# Config
sp_id = dbutils.secrets.get(scope="databricks", key="azureml-ws-databricks-sp-appid")
sp_secret = dbutils.secrets.get(scope="databricks", key="azureml-ws-databricks-sp-secret")
tenant_id = dbutils.secrets.get(scope="databricks", key="tenant-id-mh-sub")
subscription_id = dbutils.secrets.get(scope="databricks", key="subscriptionId")
resource_group = "Databricks"
aml_workspace_name = "azureml-ws-databricks"

def get_aml_ws(force: bool = False) -> Workspace:
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

ws = get_aml_ws()
print("Using AML workspace {} at location {}".format(ws.name, ws.location))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Set MLFlow tracking_uri to AML

# COMMAND ----------

import mlflow
ws.get_mlflow_tracking_uri()
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# COMMAND ----------

# MAGIC %md ### MLflow quickstart: training and logging  
# MAGIC 
# MAGIC This tutorial is based on the MLflow [ElasticNet Diabetes example](https://github.com/mlflow/mlflow/tree/master/examples/sklearn_elasticnet_diabetes). It illustrates how to use MLflow to track the model training process, including logging model parameters, metrics, the model itself, and other artifacts like plots. It also includes instructions for viewing the logged results in the MLflow tracking UI.    
# MAGIC 
# MAGIC This notebook uses the scikit-learn `diabetes` dataset and predicts the progression metric (a quantitative measure of disease progression after one year) based on BMI, blood pressure, and other measurements. It uses the scikit-learn ElasticNet linear regression model, varying the `alpha` and `l1_ratio` parameters for tuning. For more information on ElasticNet, refer to:
# MAGIC   * [Elastic net regularization](https://en.wikipedia.org/wiki/Elastic_net_regularization)
# MAGIC   * [Regularization and Variable Selection via the Elastic Net](https://web.stanford.edu/~hastie/TALKS/enet_talk.pdf)
# MAGIC   
# MAGIC ## Requirements
# MAGIC * If you are using a cluster running Databricks Runtime, you must install MLflow. Use 'mlflow' python lib with %pip install mlflow or install mflow as lib on the cluster.
# MAGIC * If you are using a cluster running Databricks Runtime ML, MLflow is already installed.
# MAGIC * Create an Azure ML workspace and a Service Principal to use for access.

# COMMAND ----------

# MAGIC %md #### Load dataset

# COMMAND ----------

# Import required libraries
import os
import warnings
import sys

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import lasso_path, enet_path
from sklearn import datasets

# Import mlflow
import mlflow
import mlflow.sklearn

# Load diabetes dataset
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame 
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

# COMMAND ----------

# MAGIC %md ## Create function to plot ElasticNet descent path
# MAGIC The `plot_enet_descent_path()` function:
# MAGIC * Creates and saves a plot of the [ElasticNet Descent Path](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html) for the ElasticNet model for the specified *l1_ratio*.
# MAGIC * Returns an image that can be displayed in the notebook using `display()`
# MAGIC * Saves the figure `ElasticNet-paths.png` to the cluster driver node

# COMMAND ----------

def plot_enet_descent_path(X, y, l1_ratio):
    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    # Reference the global image variable
    global image
    
    print("Computing regularization path using ElasticNet.")
    alphas_enet, coefs_enet, _ = enet_path(X, y, eps=eps, l1_ratio=l1_ratio, fit_intercept=False)

    # Display results
    fig = plt.figure(1)
    ax = plt.gca()

    colors = cycle(['b', 'r', 'g', 'c', 'k'])
    neg_log_alphas_enet = -np.log10(alphas_enet)
    for coef_e, c in zip(coefs_enet, colors):
        l1 = plt.plot(neg_log_alphas_enet, coef_e, linestyle='--', c=c)

    plt.xlabel('-Log(alpha)')
    plt.ylabel('coefficients')
    title = 'ElasticNet Path by alpha for l1_ratio = ' + str(l1_ratio)
    plt.title(title)
    plt.axis('tight')

    # Display images
    image = fig
    
    # Save figure
    fig.savefig("ElasticNet-paths.png")

    # Close plot
    plt.close(fig)

    # Return images
    return image    

# COMMAND ----------

# MAGIC %md ## Name the experiment in AML WS

# COMMAND ----------

# Selects what EXPERIMENT to use for logging
experiment_name = "diabetes_20220216"
if(not mlflow.get_experiment_by_name(experiment_name)):
  mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# COMMAND ----------

MlflowClient().list_experiments()

# COMMAND ----------

# MAGIC %md ## Train the diabetes model
# MAGIC The `train_diabetes()` function trains ElasticNet linear regression based on the input parameters *in_alpha* and *in_l1_ratio*.
# MAGIC 
# MAGIC The function uses MLflow Tracking to record the following:
# MAGIC * parameters
# MAGIC * metrics
# MAGIC * model
# MAGIC * the image created by the `plot_enet_descent_path()` function defined previously.
# MAGIC 
# MAGIC **Tip:** Databricks recommends using `with mlflow.start_run:` to create a new MLflow run. The `with` context closes the MLflow run regardless of whether the code completes successfully or exits with an error, and you do not have to call `mlflow.end_run`.

# COMMAND ----------

# train_diabetes
#   Uses the sklearn Diabetes dataset to predict diabetes progression using ElasticNet
#       The predicted "progression" column is a quantitative measure of disease progression one year after baseline
#       http://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html
def train_diabetes(data, in_alpha, in_l1_ratio):
  
  # Evaluate metrics
  def eval_metrics(actual, pred):
      rmse = np.sqrt(mean_squared_error(actual, pred))
      mae = mean_absolute_error(actual, pred)
      r2 = r2_score(actual, pred)
      return rmse, mae, r2

  warnings.filterwarnings("ignore")
  np.random.seed(40)

  # Split the data into training and test sets. (0.75, 0.25) split.
  train, test = train_test_split(data)

  # The predicted column is "progression" which is a quantitative measure of disease progression one year after baseline
  train_x = train.drop(["progression"], axis=1)
  test_x = test.drop(["progression"], axis=1)
  train_y = train[["progression"]]
  test_y = test[["progression"]]

  if float(in_alpha) is None:
    alpha = 0.05
  else:
    alpha = float(in_alpha)
    
  if float(in_l1_ratio) is None:
    l1_ratio = 0.05
  else:
    l1_ratio = float(in_l1_ratio)
  
  # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
  with mlflow.start_run():
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # Print out ElasticNet model metrics
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # Log mlflow attributes for mlflow UI
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    # You can log the model in the experiment (normally done) and the the model will be logged to the location of the experiment. 
    # The experiemtn always has a location on the dbfs or similar.
    mlflow.sklearn.log_model(lr, "model")
    # We can do a manual save to a custom location
    # modelpath = "/dbfs/modeltrackingmh/diabetes20220215/manualsave/model-%f-%f" % (alpha, l1_ratio)
    # mlflow.sklearn.save_model(lr, modelpath)
    
    # Call plot_enet_descent_path
    image = plot_enet_descent_path(X, y, l1_ratio)
    
    # Log artifacts (output files)
    mlflow.log_artifact("ElasticNet-paths.png")

# COMMAND ----------

# MAGIC %md ## Experiment with different parameters
# MAGIC 
# MAGIC Call `train_diabetes` with different parameters. You can visualize all these runs in the MLflow experiment.

# COMMAND ----------

# %fs rm -r /dbfs/modeltrackingmh/diabetes20220215/manualsave/

# COMMAND ----------

# alpha and l1_ratio values of 0.01, 0.01
train_diabetes(data, 0.01, 0.01)

# COMMAND ----------

display(image)

# COMMAND ----------

# alpha and l1_ratio values of 0.01, 0.75
train_diabetes(data, 0.01, 0.75)

# COMMAND ----------

display(image)

# COMMAND ----------

# alpha and l1_ratio values of 0.01, .5
train_diabetes(data, 0.01, .5)

# COMMAND ----------

display(image)

# COMMAND ----------

# alpha and l1_ratio values of 0.01, 1
train_diabetes(data, 0.01, 1)

# COMMAND ----------

display(image)

# COMMAND ----------

# MAGIC %md ## View the experiment in the AML workspace

# COMMAND ----------

# You can also use the MlflowClient vs AML as AML is just an alternative tracking server. 
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

MlflowClient().get_run(run_id = '0babbc78-6ba6-4ff1-833e-2cc9ad5c6b65')

# Sorting does not work
runs = MlflowClient().search_runs(experiment_ids=['ffcc76d2-b687-4740-920c-9d2da4008cc1'], filter_string="metrics.rmse < 70", run_view_type=ViewType.ACTIVE_ONLY, max_results=100, order_by="rmse")
#order_by="metrics.Max(rmse) ASC")

print(0, runs[0].data.metrics['rmse'])
print(1, runs[1].data.metrics['rmse'])
print()
for run in runs:
  print(run.data.metrics['rmse'])

# COMMAND ----------

# MAGIC %sh
# MAGIC less /dbfs/modeltrackingmh/diabetes20220215/db6ad503fcb844d9ba1d302ab2ddc1a6/artifacts/model/requirements.txt

# COMMAND ----------

# MAGIC %sh
# MAGIC less /dbfs/modeltrackingmh/diabetes20220215/db6ad503fcb844d9ba1d302ab2ddc1a6/artifacts/model/conda.yaml
