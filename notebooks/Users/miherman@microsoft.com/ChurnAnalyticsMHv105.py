# Databricks notebook source
# MAGIC %md ## Churn Prediction using AMLWorkbench
# MAGIC 
# MAGIC This notebook will introduce the use of the churn dataset to create churn prediction models. The dataset used to ingest is from SIDKDD 2009 competition. The dataset consists of heterogeneous noisy data (numerical/categorical variables) from French Telecom company Orange and is anonymized.
# MAGIC 
# MAGIC We will use the .dprep file created from the datasource wizard.

# COMMAND ----------

1+2

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key.datasetsnortheu.blob.core.windows.net",
  "NA0Tk+AT2oDB39cAGhW0TMLfYWOX+5ftNsROurNbUsVzQ6l1JQ1QdNryAOa6CbDsM75dr1iK39uYSxN0gH2sFQ==")

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier# DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
# from pyspark.mllib.linalg import Vectors
from pyspark.ml.linalg import Vectors 
# import nympy as np
# from azureml.logging import get_azureml_logger

# initialize logger
#run_logger = get_azureml_logger() 

# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('MH-ML-Spark-WB').getOrCreate()

# data = spark.read.format("libsvm").load("data/mllib/sample_libsvm_data.txt")
# Load the data stored in a blob linked to the HDI Cluster
# Use on HDI Cluster
# df = spark.read.csv('wasb://datasets@datasetsnortheu.blob.core.windows.net/csv/iris.csv', header=False, inferSchema=False).toDF("M1", "M2", "M3", "M4", "flower")
# Use on local docker spark
# df = spark.read.csv('./data/iris.csv', header=False, inferSchema=False).toDF("M1", "M2", "M3", "M4", "flower")
df = spark.read.csv('wasb://datasets@datasetsnortheu.blob.core.windows.net/csv/iris/iris.csv',
                         header=False, inferSchema=False).toDF("M1", "M2", "M3", "M4", "flower")

# Takes last column as label (ensure that the last column is the label before)
# Takes all columns before last as features
vectorized_CV_data = df.rdd.map(lambda r: [r[-1], Vectors.dense(r[:-1])]).toDF(['label','features'])

# Vectorize data - we need the data in a special form - only two columns features and label column
# features should be all features in a Vector - one Vector for each row of features
# label should also be a vector
def vectorizeData(dataFrame):
    return dataFrame.rdd.map(lambda r: [r[-1], Vectors.dense(r[:-1])]).toDF(['label','features'])

# vectorized_CV_data = vectorizeData(df)
data = vectorized_CV_data

# Index labels, adding metadata to the label column.
# Fit on whole dataset to include all labels in index.
# this will give each possible lable an index - in the Iris case index 0-2 as there are 3 categories
labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(data)

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
# in the iris case all features are continious so no new categorial features are created
featureIndexer =\
    VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(data)

# Split the data into training and test sets (30% held out for testing)
trainingDataShare = 0.60
#for trainingDataShare in np.arange(0.0, 0.9, 0.1):
testingDataShare = 1 - trainingDataShare
(trainingData, testData) = data.randomSplit([trainingDataShare, testingDataShare])

# Train a DecisionTree model.
dt = RandomForestClassifier(labelCol="indexedLabel", featuresCol="indexedFeatures", probabilityCol="probs")

# Chain indexers and tree in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, featureIndexer, dt])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# probs = predictions.select('probability').take(2)

#predicted = predictions.select("predictions")
#actual = predictions.select("indexedLabel")

# Select example rows to display.
predictions.select("prediction", "indexedLabel", "features", "probs").show(10)
errors = predictions.select("prediction", "indexedLabel", "features", "probs").filter(predictions.prediction != predictions.indexedLabel)
print("NoOf Errors:", errors.count(), errors.show(10))
print(errors.toPandas())

# Select (prediction, true label) and compute test error
evaluatorAcc = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")

evaluatorrec = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="weightedRecall")

# Select (prediction, true label) and compute f1 value
evaluatorF1 = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="f1")

evaluatorprec = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="weightedPrecision")

# run_logger.log('trainingDataShare', trainingDataShare)

accuracy = evaluatorAcc.evaluate(predictions)
testError = 1.0 - accuracy
print("Test Error = %g " % (testError))
#run_logger.log('Test Error:', testError)
recall = evaluatorrec.evaluate(predictions)
print("weightedRecall:", recall)

f1 = evaluatorF1.evaluate(predictions)
print("f1:", f1)
# run_logger.log('f1:', f1)

prec = evaluatorprec.evaluate(predictions)
print("weightedPrecision:", prec)

treeModel = model.stages[2]
# summary only
print(treeModel)

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark
import os
import urllib
import sys

from pyspark.sql.functions import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *

#from azureml.logging import get_azureml_logger

# initialize logger
#run_logger = get_azureml_logger() 

# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('MH-ML-Spark-WB').getOrCreate()

# 'wasb:///example/data/gutenberg/ulysses.txt'

# print runtime versions
print ('****************')
print ('Python version: {}'.format(sys.version))
print ('Spark version: {}'.format(spark.version))
print ('****************')

# val df = spark.read.parquet("wasbs://{YOUR CONTAINER NAME}@{YOUR STORAGE ACCOUNT NAME}.blob.core.windows.net/{YOUR DIRECTORY NAME}")
# dbutils.fs.ls("wasbs://{YOUR CONTAINER NAME}@{YOUR STORAGE ACCOUNT NAME}.blob.core.windows.net/{YOUR DIRECTORY NAME}")


##dbutils.fs.mount(
#  source = "wasbs://datasets@datasetsnortheu.blob.core.windows.net/csv",
#  mount_point = "/mnt/iris")#,
#  #extra_configs = {"{confKey}": "{confValue}"})

#ok
csvFile = spark.read.csv('wasb://datasets@datasetsnortheu.blob.core.windows.net/csv/iris/iris.csv',
                         header=False, inferSchema=False).toDF("M1", "M2", "M3", "M4", "flower")
# csvFile.head(2)

csvFile.columns
csvFile.head(5)
csvFile.printSchema()

# COMMAND ----------

import numpy as np
import pandas as pd
import pyspark
import os
import urllib
import sys

from pyspark.sql.functions import *
from pyspark.ml.classification import *
from pyspark.ml.evaluation import *
from pyspark.ml.feature import *

# from azureml.logging import get_azureml_logger

# initialize logger
# run_logger = get_azureml_logger() 

# start Spark session
spark = pyspark.sql.SparkSession.builder.appName('Iris').getOrCreate()

# 'wasb:///example/data/gutenberg/ulysses.txt'

# print runtime versions
print ('****************')
print ('Python version: {}'.format(sys.version))
print ('Spark version: {}'.format(spark.version))
print ('****************')

# load iris.csv into Spark dataframe
# ok data = spark.createDataFrame(pd.read_csv('./data/iris.csv', header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']))
# ej ok då pd.read inte kan läsa från wasb
#data = spark.createDataFrame(pd.read_csv('wasb://datasets@datasetsnortheu.blob.core.windows.net/datasets/csv/iris.csv', header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']))

#ok
data = spark.read.csv('wasb://datasets@datasetsnortheu.blob.core.windows.net/csv/iris/iris.csv', header=False, inferSchema=False)
#ok
# csvFile = spark.read.csv('wasb://nifitestmh-2017-11-01t14-01-18-626z@datasetsnortheu.blob.core.windows.net/example/data/sample.log', header=False, inferSchema=False)

#data = spark.createDataFrame(pd.read_csv('wasb:///user/sshuser/CustomerChurn_1513191596822/data/iris.csv', header=None, names=['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']))
# 'wasb:///user/sshuser/CustomerChurn_1513191596822/data/iris.csv'
# wasb[s]://<containername>@<accountname>.blob.core.windows.net/<path>
# wasb://datasets@datasetsnortheu.blob.core.windows.net/datasets/csv/iris.csv
print("First 10 rows of Iris dataset:")
data.show(10)

# vectorize all numerical columns into a single feature column
feature_cols = data.columns[:-1]
assembler = pyspark.ml.feature.VectorAssembler(inputCols=feature_cols, outputCol='features')
data = assembler.transform(data)

# convert text labels into indices
data = data.select(['features', 'class'])
label_indexer = pyspark.ml.feature.StringIndexer(inputCol='class', outputCol='label').fit(data)
data = label_indexer.transform(data)

# only select the features and label column
data = data.select(['features', 'label'])
print("Reading for machine learning")
data.show(10)

# change regularization rate and you will likely get a different accuracy.
reg = 0.01
# load regularization rate from argument if present
if len(sys.argv) > 1:
    reg = float(sys.argv[1])

# log regularization rate
# run_logger.log("Regularization Rate", reg)

# use Logistic Regression to train on the training set
train, test = data.randomSplit([0.70, 0.30])
lr = pyspark.ml.classification.LogisticRegression(regParam=reg)
model = lr.fit(train)

# predict on the test set
prediction = model.transform(test)
print("Prediction")
prediction.show(10)

# evaluate the accuracy of the model using the test set
evaluator = pyspark.ml.evaluation.MulticlassClassificationEvaluator(metricName='accuracy')
accuracy = evaluator.evaluate(prediction)

print()
print('#####################################')
print('Regularization rate is {}'.format(reg))
print("Accuracy is {}".format(accuracy))
print('#####################################')
print()

# log accuracy
# run_logger.log('Accuracy', accuracy)


# COMMAND ----------

import dataprep
from dataprep.Package import Package
import pickle

from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pandas as pd
#import numpy as np
import csv
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder

# COMMAND ----------

# MAGIC %matplotlib inline

# COMMAND ----------

# MAGIC %md ## Read Data
# MAGIC 
# MAGIC We first retrieve the data as a data frame using .dprep that we created using the datasource wizard. Print the top few lines using head()

# COMMAND ----------

#Local via package
#with Package.open_package('CATelcoCustomerChurnTrainingSample.dprep') as pkg:
#    df = pkg.dataflows[0].get_dataframe()

# Blob via package
#with Package.open_package('CATelcoCustomerChurnTrainingBlobSample.dprep') as pkg:
#    df = pkg.dataflows[0].get_dataframe()
    
# if this one fails check to see that you are NOT exporting file to local/blob
# storage because it will fail to overwrite

# Local via CSV file
df = pd.read_csv("./data/CATelcoCustomerChurnTrainingSample.csv")
# df = pd.read_csv("./data/CATelcoCustomerChurnTrainingCleaned.csv")

df.head(5)

# COMMAND ----------

df.shape

# COMMAND ----------

# What is the index for the churn column
df.columns.get_loc("churn")

# COMMAND ----------

# MAGIC %md ## Encode Columns
# MAGIC 
# MAGIC Convert categorical variable into dummy/indicator variables using pandas.get_dummies. In addition, we will need to change the column names to ensure there are no multiple columns with the same name 

# COMMAND ----------

# Pick columns of the type category (object represents strings)
columns_to_encode = list(df.select_dtypes(include=['category','object']))
print(columns_to_encode)
# Create new columns for each value in the chosen columns. Use the the indicator value 1 for a value set. 
for column_to_encode in columns_to_encode:
    # Create a new column for each unique value a column has
    # The value becomes a column name
    # See: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html
    dummies = pd.get_dummies(df[column_to_encode])
    # Save column names in this variable
    one_hot_col_names = []
    for col_name in list(dummies.columns):
        # Use the original column name followed by the value of the category
        one_hot_col_names.append(column_to_encode + '_' + col_name)
    # Assign new column names
    dummies.columns = one_hot_col_names
    # drop the original columns that are now one-hot-encoded (axis=1 means to drop columns)
    df = df.drop(column_to_encode, axis=1)
    # merge the original data frame with the newly constructed columns
    df = df.join(dummies)

print("Encoded columns:")
print(df.columns)

# COMMAND ----------

# Lets look at our newly created columns (one-hot-encoded)
df

# COMMAND ----------

# MAGIC %md ## Modeling
# MAGIC 
# MAGIC First, we will build a Gaussian Naive Bayes model using GaussianNB for churn classification. Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes theorem with the “naive” assumption of independence between every pair of features.
# MAGIC 
# MAGIC In addition, we will also build a decision tree classifier for comparison:
# MAGIC 
# MAGIC - min_samples_split=20 requires 20 samples in a node for it to be split
# MAGIC - random_state=99 to seed the random number generator

# COMMAND ----------

# MAGIC %md ## Investigate important features
# MAGIC Not all features might be important. Try to find the features that influence the most bu using the TreeClassifier.

# COMMAND ----------

from sklearn.ensemble import ExtraTreesClassifier

train, test = train_test_split(df, test_size = 0.3)
Y = train['churn'].values
X = train.drop('churn', 1)
X = X.values

model_feature_test = ExtraTreesClassifier()
model_feature_test.fit(X, Y)
features_ranked = model_feature_test.feature_importances_
print("Feature importance (the number indicates the importance of each column in the data frame:\n" + str(features_ranked))

# Pick the index for the top ranked features (20)
ind = np.argpartition(features_ranked, -20)[-20:]
print("Top Feature Index:\n" + str(ind))
print("Top feature values\n")
print(features_ranked[ind])
top_feature_names = train.columns[ind]
print("Top feature names:\n" + str(top_feature_names))

# COMMAND ----------

# Index of the churn column
df.columns.get_loc("churn")

# COMMAND ----------

# Number of columns we now have
len(df.columns)

# COMMAND ----------

# MAGIC %md ## Train on only top features

# COMMAND ----------

top_feature_names

# COMMAND ----------

# MAGIC %md #### Decision tree classifier

# COMMAND ----------

model = DecisionTreeClassifier(max_depth=20, max_features=20, random_state=98)

# top_feature_names.append(['churn'])
dfChurn = df[['churn']]
#dfFeatures = df[top_feature_names]
#dfTopFeaturesChurn = dfFeatures.join(dfChurn)

train, test = train_test_split(df, test_size = 0.3, random_state=98)
# Does top features only improve the classifier?
# train, test = train_test_split(dfTopFeaturesChurn, test_size = 0.3, random_state=98)
target = train['churn'].values
train = train.drop('churn', 1)
train = train.values
model.fit(train, target)

Y_test_val = test['churn'].values
X_test_val = test.drop('churn', 1).values
predicted = model.predict(X_test_val)
print("Decision Tree Classification Accuracy", accuracy_score(Y_test_val, predicted))


# COMMAND ----------

# Calculate confusion matrix
exp_inp = pd.Categorical(Y_test_val, categories=[0,1])
pred_inp = pd.Categorical(predicted, categories=[0,1])
print(pd.crosstab(exp_inp, pred_inp, colnames=["Predicted"], rownames=['Actual']))

# COMMAND ----------

print("#Expected churn: " + str(sum(Y_test_val[Y_test_val[:] > 0])))
print("#Predcited churn: " + str(sum(predicted[predicted[:] > 0])))

# COMMAND ----------

# MAGIC %md |Predicted|     0 |  1 |
# MAGIC |---------|-------|----|
# MAGIC |Actual   |       |    |             
# MAGIC |0        |   2665|  49|
# MAGIC |1        |    282|   4|
# MAGIC 
# MAGIC Accuracy example: acc = correct predicted / all predictions<br>
# MAGIC (2665 + 4)/ (2665 + 49 + 282 + 4) = 0.8897

# COMMAND ----------

# MAGIC %md #### ROC curve

# COMMAND ----------

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
predicted_prob = model.predict_proba(X_test_val)
# TASK: Rita ROC curve
# confidence_nb = predicted_prob.max(axis=1)
fpr, tpr, thresholds = roc_curve(Y_test_val, predicted_prob[:, 1])
roc_auc_nb = roc_auc_score(Y_test_val, predicted_prob[:, 1])
print("AUC:" + str(roc_auc_nb))

plt.figure(figsize=(10,10))
plt.title('Receiver operating characteristic example')
plt.plot(fpr, tpr, color='darkorange', lw = 2, label='ROC curve (area = %0.2f)' % roc_auc_nb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# COMMAND ----------

# MAGIC %md ### Precision and recall
# MAGIC High recall means that we manage to pick most of the positive examples (churn)
# MAGIC Low precision means that we also pick a lot of false positive for the churn which
# MAGIC means that a high recall and a low precision catches most of the true churn at the
# MAGIC cost of getting many false churn at the same time. Low precision = not so precise
# MAGIC 
# MAGIC See: https://medium.com/@klintcho/explaining-precision-and-recall-c770eb9c69e9 for a good explanation.
# MAGIC 
# MAGIC The chart below calculates the precision and recall for the different thresholds of the probability of our predicions.

# COMMAND ----------

precision, recall, thresholds_pr = precision_recall_curve(Y_test_val, predicted_prob[:,1])
plt.figure(figsize=(10,10))
plt.title('Precision Recall curve')
plt.plot(precision, recall, color='darkorange', lw = 2, label='ROC curve (area = %0.2f)' % roc_auc_nb)
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('precision')
plt.ylabel('recall')
plt.show()
print("Precision:\n" + str(precision.round(4)))
print("Recall:\n" + str(recall.round(4)))

#print("\nIf we want to catch every churn the precision would be only " + "{:.2%}".format(precision[0]))
#print("I.e. we would catch a huge number of false positive at the same time.")
#print("The other way around. If we want only True positive churn (no false) we would get a recall of " + "{:.2%}".format(recall[6]))

# COMMAND ----------

# MAGIC %md ### ExtraTrees Classifier

# COMMAND ----------

help(ExtraTreesClassifier)

# COMMAND ----------

modelxt = ExtraTreesClassifier(max_depth=25, n_estimators=15, max_features=20, random_state=99)
# modelxt = RandomForestClassifier()

# top_feature_names.append(['churn'])
dfChurn = df[['churn']]
# dfFeatures = df[top_feature_names]
# dfTopFeaturesChurn = dfFeatures.join(dfChurn)

train, test = train_test_split(df, test_size = 0.3, random_state=98)

target = train['churn'].values
train = train.drop('churn', 1)
train = train.values
modelxt.fit(train, target)

Y_test_val = test['churn'].values
X_test_val = test.drop('churn', 1).values
predicted = modelxt.predict(X_test_val)
print("ExtraTreesClassifier acc:", accuracy_score(Y_test_val, predicted))

# COMMAND ----------

# MAGIC %md #### Confusion matrix

# COMMAND ----------

print("#Actual churn: " + str(sum(Y_test_val[Y_test_val[:] > 0])))
print("#Predcited churn: " + str(sum(predicted[predicted[:] > 0])))

# COMMAND ----------

exp_inp = pd.Categorical(Y_test_val, categories=[0,1])
pred_inp = pd.Categorical(predicted, categories=[0,1])
print(pd.crosstab(exp_inp, pred_inp, colnames=["Predicted"], rownames=['Actual']))

# COMMAND ----------

# MAGIC %md ####  Confusion matrix results example
# MAGIC     Predicted     0    1
# MAGIC     Actual              
# MAGIC     0          2708    4
# MAGIC     1            75  213
# MAGIC 
# MAGIC We predicted 2708+75 as non-churn<br>
# MAGIC We predicted 4 non-churn wrongly as churn<br>
# MAGIC We predicted 217 as churn (4 was non-churn of these)<br>
# MAGIC We did not catch 75 churn that was predcited as non-churn<br>

# COMMAND ----------

# MAGIC %md ### ROC Curve ExtraTreesClassifier

# COMMAND ----------

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
predicted_prob = modelxt.predict_proba(X_test_val)
# TASK: Rita ROC curve
# confidence_nb = predicted_prob.max(axis=1)
fpr, tpr, thresholds = roc_curve(Y_test_val, predicted_prob[:, 1])
roc_auc_nb = roc_auc_score(Y_test_val, predicted_prob[:, 1])
print("AUC:" + str(roc_auc_nb))

plt.figure(figsize=(10,10))
plt.title('Receiver operating characteristic example')
plt.plot(fpr, tpr, color='darkorange', lw = 2, label='ROC curve (area = %0.2f)' % roc_auc_nb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()

# COMMAND ----------

# MAGIC %md ### Precision and recall
# MAGIC High recall means that we manage to pick most of the positive examples (churn)
# MAGIC Low precision means that we also pick a lot of false positive for the churn which
# MAGIC means that a high recall and a low precision catches most of the true churn at the
# MAGIC cost of getting many false churn at the same time. Low precision = not so precise
# MAGIC 
# MAGIC See: https://medium.com/@klintcho/explaining-precision-and-recall-c770eb9c69e9 for a good explanation.
# MAGIC 
# MAGIC The chart below calculates the precision and recall for the different thresholds of the probability of our predicions.

# COMMAND ----------

precision, recall, thresholds_pr = precision_recall_curve(Y_test_val, predicted_prob[:,1])
plt.figure(figsize=(10,10))
plt.title('Precision Recall curve')
plt.plot(precision, recall, color='darkorange', lw = 2, label='ROC curve (area = %0.2f)' % roc_auc_nb)
plt.legend(loc="lower right")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('precision')
plt.ylabel('recall')
plt.show()
print("Precision:\n" + str(precision))
print("Recall:\n" + str(recall))

print("\nIf we want to catch every churn the precision would be only " + "{:.2%}".format(precision[0]))
print("I.e. we would catch a huge number of false positive at the same time.")
print("The other way around. If we want only True positive churn (no false) we would get a recall of X? %")
# print("The other way around. If we want only True positive churn (no false) we would get a recall of " + "{:.2%}".format(recall[5]))



# COMMAND ----------

# MAGIC %md ### Multi model training

# COMMAND ----------

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn import model_selection


# prepare data
# top_feature_names.append(['churn'])
#dfChurn = df[['churn']]
#dfFeatures = df[top_feature_names]
#dfTopFeaturesChurn = dfFeatures.join(dfChurn)

train, test = train_test_split(df.head(25000), test_size = 0.3, random_state=98)
Y_train = train['churn'].values
X_train = train.drop('churn', 1)
seed = 7

# prepare models
models = [('DecisionTreeClassifier', DecisionTreeClassifier(max_depth=20, max_features=20, random_state=98)),
         ('RandomForestClassifier', RandomForestClassifier(max_depth=20, max_features=20, random_state=98)),
         ('ExtraTreesClassifier', ExtraTreesClassifier(max_depth=20, max_features=20, random_state=98)),
         ('ExtraTreesClassifier tuned', ExtraTreesClassifier(max_depth=30, max_features=30, n_estimators=20, random_state=198))]
predicts = []
confusion_matrixes = []
accuracies = []
predicted_probs = []

dfRes = pd.DataFrame()
results = []
names = []
scoring = 'f1'
for name, model in models:
    names.append(name)
    model.fit(X_train, Y_train)
    Y_test_val = test['churn'].values
    X_test_val = test.drop('churn', 1).values
    
    # model selection with kfold
    kfold = model_selection.KFold(n_splits=3, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    
    predicted =  model.predict(X_test_val)
    predicts.append((name, predicted))
    acc = accuracy_score(Y_test_val, predicted)
    accuracies.append((name, acc))
    # Confusion matrix
    exp_inp = pd.Categorical(Y_test_val, categories=[0,1])
    pred_inp = pd.Categorical(predicted, categories=[0,1])
    confusion_matrixes.append((name, pd.crosstab(exp_inp, pred_inp, colnames=["Predicted"], rownames=['Actual'])))
    predicted_prob = model.predict_proba(X_test_val)
    predicted_probs.append((name, predicted_prob))

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison: ' + scoring)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels([name[0:3] + name[-2:] for name in names])
ax.autoscale_view
plt.show()

for pred in predicts:
    print(pred)
    
for acc in accuracies:
    print('Accuracy:', acc)
    
for name, conf_mat in confusion_matrixes:
    print('\n')
    print(name, conf_mat, sep='\n')
    
# Draw ROC Curves
plt.figure(figsize=(10,10))
plt.title('Receiver operating characteristic example')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
n=len(predicted_probs)
# color=iter(plt.cm.rainbow(np.linspace(0,1,n)))
colArr = []
for k in range(n):
    colArr.append(k/n)

print(colArr)

color=iter(plt.cm.rainbow(colArr))
print(color)
           
# Plot all results from all models
for name, predicted_prob in predicted_probs:
    c=next(color)
    #predicted_prob = model.predict_proba(X_test_val)
    # TASK: Rita ROC curve
    # confidence_nb = predicted_prob.max(axis=1)
    fpr, tpr, thresholds = roc_curve(Y_test_val, predicted_prob[:, 1])
    roc_auc_nb = roc_auc_score(Y_test_val, predicted_prob[:, 1])
    print("AUC:", name, str(roc_auc_nb))
    plt.plot(fpr, tpr, color=c, lw = 2, label=name + '(area = %0.2f)' % roc_auc_nb)
    
    plt.legend(loc="lower right")
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
plt.show()



# COMMAND ----------

names = ['kalle', 'otto', 'pelle', 'nisselong']

[name[0:3]+name[-2:] for name in names] 