# Databricks notebook source
# DBTITLE 1,Imports nice to have
# Imports
from pyspark.sql.window import Window
from pyspark.sql import functions as F
from pyspark.sql.functions import col, asc, datediff, to_date, lit, coalesce, explode
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import GBTRegressor, RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from datetime import datetime
from pyspark.ml.feature import VectorAssembler, VectorIndexer, StringIndexer
import time

# COMMAND ----------

# DBTITLE 1,Get notebook name (python)
def get_current_notebook():
  notebook_name_with_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
  return notebook_name_with_path

print(get_current_notebook())

# COMMAND ----------

# DBTITLE 1,Get notebook name (Scala)
# MAGIC %scala
# MAGIC val notebookPath: String = dbutils.notebook.getContext.notebookPath.getOrElse("No notebook found")
# MAGIC println(notebookPath)

# COMMAND ----------

# DBTITLE 1,Log to file (blob ADLS) (Note: only works on old runtimes <6.5)
# Log to file in blob storage/ADLS 
logTimeFile = "/dbfs/tmp/timings.cvs"
def logToFile(str):
  with open(logTimeFile, "a+") as myfile:
    myfile.write(str)
    myfile.write("\n")
    myfile.close()

# COMMAND ----------

# DBTITLE 1,Log to csv file (locally) and copy to Data Lake
# Log to file in blob storage/ADLS by writing locally and copy the file to the storage account
# Reason is: Random IO not supported over /dbfs anymore in new runtimes.
# So not optimal as every log entry will copy over the file
import datetime
from shutil import copyfile

def log_to_file2(log_msg, log_file_path = "/dbfs/mnt/datalakename/folername/logs/ft_log.csv", info2 = "", info3 = ""):
  args = ''.join(f"{v.replace(',','_')}," for v in locals().values() if v is not None)
  print(args + "\n")
  local_path= "/local_disk0/tmp/log.csv"
  with open(local_path, "a+") as myfile:
    dt = str(datetime.datetime.today())
    str_merge = f"{dt},{args}"
    myfile.write(str_merge)
    myfile.write("\n")
    myfile.close()
  copyfile(local_path, log_file_path)

# COMMAND ----------

# DBTITLE 1,Log to csv file (as above)
def log_to_file(*strings, log_file_path = "/dbfs/mnt/datalakename/databricks/logs/ft_log.csv"):
  str_out = str(datetime.datetime.today())
  for f in strings:
    str_out = str_out + ',' + f.replace(',', '_')
  print("logging this file: " + str_out)
  local_path= "/local_disk0/tmp/log.csv"
  with open(local_path, "a+") as myfile:
    myfile.write(str_out)
    myfile.write("\n")
    myfile.close()
  copyfile(local_path, log_file_path)

# COMMAND ----------

# DBTITLE 1,Filter column older than 14 days
# Date older than 14 days from now
today = time.time()
timeDiff = (today - F.unix_timestamp('DATE_COL'))
df_data_filtered = df_data.filter(timeDiff > 14*24*3600)

# COMMAND ----------

# DBTITLE 1,Filter out dates > today (future dates)
today = time.time()
beforeToday = (F.unix_timestamp("DATE_COL") <= lit(today))
df_data_filtered = df_data.filter(beforeToday)

# COMMAND ----------

# DBTITLE 1,Select columns from a list and at the same time filter columns by name
# Select columns from a list and filter columns by name
# Features and targets
selColumns = "COL_FEAT_A","COL_FEAT_B","COL_FEAT_C", "COL_TARGET"
# Also, get all columns starting with SPECIAL_KEEP  
condition = lambda col: col.startswith('SPECIAL_KEEP')
special_keep_cols = filter(condition, df_data.columns)
col_union = list(set().union(special_keep_cols, selColumns))
df_data_with_chosen_cols = df_data.select(col_union)

# COMMAND ----------

# DBTITLE 1,Helper method to get list of methods on any object
# Helper method to get list of methods on any object
def getMethods(object):
  for name in [method_name for method_name in dir(object) if callable(getattr(object, method_name))]:
    print(name)

# COMMAND ----------

# DBTITLE 1,Count null % for one column
# Counts null % of a column
def countNull(columnName, dataset):
  return float(dataset.select(columnName).filter("{columnName} is null".format(columnName=columnName)).count()) / dataset.select(columnName).count()

# COMMAND ----------

# DBTITLE 1,Null stats count for all columns
#Null stats for all cols
dfNullStats = dfDataSet.select([count(when(col(c).isNull(), c)).alias(c) for c in dfDataSet.columns])


# COMMAND ----------

# DBTITLE 1,Drop columns that has more than 10% nulls
noRows = dfDataSet.count()
dropCols = []
dfPanda = dfNullStats.toPandas()
for colIndex in range(len(dfPanda.columns)):
  if(float(dfPanda.iloc[0,colIndex]) / noRows > 0.1):
    dropCols.append(dfPanda.columns[colIndex])
print("Will drop columns", dropCols)

dfDataSet = dfDataSet.drop(*dropCols)
rowsBefore = dfDataSet.count()
dfDataSet = dfDataSet.dropna()
rowsAfter = dfDataSet.count()
print("Dropped rows (%):", float(rowsBefore - rowsAfter)*100/rowsBefore)
