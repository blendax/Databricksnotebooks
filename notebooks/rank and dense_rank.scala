// Databricks notebook source
// MAGIC %md
// MAGIC Exmaple of difference between rank and dense_rank in spark

// COMMAND ----------

import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._

val df = Seq(("a", 10), ("a", 10), ("a", 20)).toDF("col1", "col2")

val windowSpec = Window.partitionBy("col1").orderBy("col2")

df
  .withColumn("rank", rank().over(windowSpec))
  .withColumn("dense_rank", dense_rank().over(windowSpec))
  .withColumn("row_number", row_number().over(windowSpec)).show
