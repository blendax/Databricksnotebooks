# Databricks notebook source
# MAGIC %md ## Setup

# COMMAND ----------

print("cluster_running", True)

# COMMAND ----------

# MAGIC %pip install mlflow scikit-learn matplotlib azureml-mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ####Make sure to set the tracking_uri of MLFlow to AML

# COMMAND ----------

# MAGIC %md
# MAGIC #### Connect to AML WS using SP (or manully if you prefer)

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

ws.get_mlflow_tracking_uri()

# COMMAND ----------

import mlflow
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# COMMAND ----------

# MAGIC %md #### Write Your ML Code Based on the`train_diabetes.py` Code
# MAGIC This tutorial is based on the MLflow's [train_diabetes.py](https://github.com/mlflow/mlflow/blob/master/examples/sklearn_elasticnet_diabetes/train_diabetes.py) example, which uses the `sklearn.diabetes` built-in dataset to predict disease progression based on various factors.

# COMMAND ----------

# Import various libraries including matplotlib, sklearn, mlflow
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

# Load Diabetes datasets
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

# Create pandas DataFrame for sklearn ElasticNet linear_model
Y = np.array([y]).transpose()
d = np.concatenate((X, Y), axis=1)
cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
data = pd.DataFrame(d, columns=cols)

# COMMAND ----------

# MAGIC %md #### Plot the ElasticNet Descent Path
# MAGIC As an example of recording arbitrary output files in MLflow, we'll plot the [ElasticNet Descent Path](http://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_coordinate_descent_path.html) for the ElasticNet model by *alpha* for the specified *l1_ratio*.
# MAGIC 
# MAGIC The `plot_enet_descent_path` function below:
# MAGIC * Returns an image that can be displayed in our Databricks notebook via `display`
# MAGIC * As well as saves the figure `ElasticNet-paths.png` to the Databricks cluster's driver node
# MAGIC * This file is then uploaded to MLflow using the `log_artifact` within `train_diabetes`

# COMMAND ----------

def plot_enet_descent_path(X, y, l1_ratio):
    # Compute paths
    eps = 5e-3  # the smaller it is the longer is the path

    # Reference the global image variable
    global image
    
    print("Computing regularization path using the elastic net.")
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

# MAGIC %md #### Organize MLflow Runs into Experiments
# MAGIC 
# MAGIC As you start using your MLflow server for more tasks, you may want to separate them out. MLflow allows you to create [experiments](https://mlflow.org/docs/latest/tracking.html#organizing-runs-in-experiments) to organize your runs. To report your run to a specific experiment, pass an experiment name to `mlflow.set_experiment`.

# COMMAND ----------

# Selects what EXPERIMENT to use for logging
experiment_name = "Diabetes_MLflow_DBX_with_aml_as_tracking_uri_20211208_af"

if(not mlflow.get_experiment_by_name(experiment_name)):
  mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

# COMMAND ----------

mlflow.get_experiment_by_name(experiment_name)

# COMMAND ----------

# MAGIC %md #### Train the Diabetes Model
# MAGIC The next function trains Elastic-Net linear regression based on the input parameters of `alpha (in_alpha)` and `l1_ratio (in_l1_ratio)`.
# MAGIC 
# MAGIC In addition, this function uses MLflow Tracking to record its
# MAGIC * parameters
# MAGIC * metrics
# MAGIC * model
# MAGIC * arbitrary files, namely the above noted Lasso Descent Path plot.
# MAGIC 
# MAGIC **Tip:** We use `with mlflow.start_run:` in the Python code to create a new MLflow run. This is the recommended way to use MLflow in notebook cells. Whether your code completes or exits with an error, the `with` context will make sure that we close the MLflow run, so you don't have to call `mlflow.end_run` later in the code.

# COMMAND ----------

# Add extra pip req for model serving azureml-defaults just to try an extra requirement (not needed until you server with Azure ML only)

# COMMAND ----------

# MAGIC %%writefile ./extra_pip_requirements.txt
# MAGIC azureml-defaults

# COMMAND ----------

# MAGIC %sh less extra_pip_requirements.txt

# COMMAND ----------

import joblib
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
  
  # -----------------
  # ML FLOW
  # -----------------
  # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
  with mlflow.start_run() as latest_run:
    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    # -----------
    # METRICS
    # -----------
    # Print out ElasticNet model metrics
    print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
    print("  RMSE: %s" % rmse)
    print("  MAE: %s" % mae)
    print("  R2: %s" % r2)

    # -----------
    # PARAMS and more metrics, MODEL, ARTIFACT
    # -----------
    # Log mlflow attributes for mlflow UI
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mae", mae)
    
    # We can use the mlflow.sklearn logger to det some things for free but also loose control slightly
    mlflow.sklearn.log_model(sk_model=lr, 
                             artifact_path="foldername/in/outputlogs",
                             extra_pip_requirements="extra_pip_requirements.txt", 
                             input_example = test_x.iloc[0:2])
    
    # Alternative is to create a binary ourselves and upload to output of experiment (depends on model lib what we want to do)
    # Save the trained model
    model_file = 'lr-model-binary.pkl'
    joblib.dump(value=lr, filename=model_file)
    mlflow.log_artifact(local_path="./lr-model-binary.pkl", artifact_path="manualmodelupload")

    
    # Call plot_enet_descent_path
    image = plot_enet_descent_path(X, y, l1_ratio)
    
    # Log artifacts (output files)
    mlflow.log_artifact("ElasticNet-paths.png")
    return latest_run

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 0.01
run = train_diabetes(data, 0.03, 0.04)

# COMMAND ----------

from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType

# Sorting does not work
runs = MlflowClient().search_runs(experiment_ids=['ce64fe0d-1abd-41b6-92e3-dee03a04c7dc'], filter_string="metrics.rmse < 70", run_view_type=ViewType.ACTIVE_ONLY, max_results=3, order_by="metrics.Max(rmse) ASC")

print(0, runs[0].data.metrics['rmse'])
print(1, runs[1].data.metrics['rmse'])
print()
for run in runs:
  print(run.data.metrics['rmse'])

# COMMAND ----------

runs[0]

# COMMAND ----------

MlflowClient().get_run

# COMMAND ----------

# Now when we have developed out model and logged it we have the knowledge how to use it

# COMMAND ----------

# MAGIC %md
# MAGIC ![](https://docs.databricks.com/_static/images/mlflow/elasticnet-paths-by-alpha-per-l1-ratio.png)

# COMMAND ----------

# MAGIC %md #### Experiment with Different Parameters
# MAGIC 
# MAGIC Now that we have a `train_diabetes` function that records MLflow runs, we can simply call it with different parameters to explore them. Later, we'll be able to visualize all these runs on our MLflow tracking server.

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 0.01
train_diabetes(data, 0.01, 0.01)

# COMMAND ----------

display(image)

# COMMAND ----------

# Start with alpha and l1_ratio values of 0.01, 1
train_diabetes(data, 0.01, 1)

# COMMAND ----------

display(image)

# COMMAND ----------

# MAGIC %md ## Review the experiment
# MAGIC 
# MAGIC #### USe AML GUI in AZURE instead now as we track there (nothing in the WS)
# MAGIC 1. Open the experiment `/Shared/experiments/DiabetesModel` in the workspace.
# MAGIC 1. Click a date to view a run.

# COMMAND ----------

# MAGIC %md
# MAGIC The experiment should look something similar to the animated GIF below. Inside the experiment, you can:
# MAGIC * View runs
# MAGIC * Review the parameters and metrics on each run
# MAGIC * Click each run for a detailed view to see the the model, images, and other artifacts produced.
# MAGIC 
# MAGIC <img src="https://docs.databricks.com/_static/images/mlflow/mlflow-ui.gif"/>

# COMMAND ----------

# Input to ML WS
myjson = data[["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"]][:2].to_json(orient='split')
myjson

# COMMAND ----------

# MAGIC %md
# MAGIC Works as input to test the the WebSservice in Azure ML serving:<br>
# MAGIC ```
# MAGIC {
# MAGIC 	"input_data": {"columns":["age","sex","bmi","bp","s1","s2","s3","s4","s5","s6"],
# MAGIC     "index":[0,1],
# MAGIC     "data":[[0.0380759064,0.0506801187,0.0616962065,0.021872355,-0.0442234984,-0.0348207628,-0.0434008457,-0.002592262,0.0199084209,-0.0176461252],
# MAGIC     [-0.0018820165,-0.0446416365,-0.0514740612,-0.0263278347,-0.0084487241,-0.0191633397,0.0744115641,-0.0394933829,-0.0683297436,-0.0922040496]]}
# MAGIC }
# MAGIC ```
