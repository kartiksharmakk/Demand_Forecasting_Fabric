# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "525bb848-3ac3-469d-8861-b50f558dbb76",
# META       "default_lakehouse_name": "Dataset_Lh",
# META       "default_lakehouse_workspace_id": "e3c5531b-e225-4c62-87da-8c5fe9a6aeea",
# META       "known_lakehouses": [
# META         {
# META           "id": "525bb848-3ac3-469d-8861-b50f558dbb76"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************


# CELL ********************

# Import Required Libraries
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor
from pyspark.sql.functions import col, to_date, to_timestamp
from pyspark.ml.feature import StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

sales_data = spark.read.format("csv").option("header","true").load("Files/dataset/Online_Retail.csv")
# df now is a Spark DataFrame containing CSV data from "Files/Online_Retail.csv".
display(sales_data)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

sales_data = sales_data.withColumn("Year", col("Year").cast("int")) \
                                   .withColumn("Month", col("Month").cast("int")) \
                                   .withColumn("Day", col("Day").cast("int")) \
                                   .withColumn("Week", col("Week").cast("int")) \
                                   .withColumn("DayOfWeek", col("DayOfWeek").cast("int"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(sales_data)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Convert InvoiceDate to datetime 
sales_data = sales_data.withColumn("InvoiceDate", to_date(to_timestamp(col("InvoiceDate"), "yyyy-MM-dd HH:mm:ss")))


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

display(sales_data)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Aggregate data into daily intervals
daily_sales_data = sales_data.groupBy("Country", "StockCode", "InvoiceDate", "Year", "Month", "Day", "Week", "DayOfWeek").agg({"Quantity": "sum",                                                                                                           "UnitPrice": "avg"})
# Rename the target column
daily_sales_data = daily_sales_data.withColumnRenamed(
    "sum(Quantity)", "Quantity")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Split the data into two sets based on the spliting date, "2011-09-25". All data up to and including this date should be in the training set, while data after this date should be in the testing set. Return a pandas Dataframe, pd_daily_train_data, containing, at least, the columns ["Country", "StockCode", "InvoiceDate", "Quantity"].

split_date_train_test = "2011-09-25"

# Creating the train and test datasets
train_data = daily_sales_data.filter(
    col("InvoiceDate") <= split_date_train_test)
test_data = daily_sales_data.filter(col("InvoiceDate") > split_date_train_test)

pd_daily_train_data = train_data.toPandas()

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Creating indexer for categorical columns
country_indexer = StringIndexer(
    inputCol="Country", outputCol="CountryIndex").setHandleInvalid("keep")
stock_code_indexer = StringIndexer(
    inputCol="StockCode", outputCol="StockCodeIndex").setHandleInvalid("keep")

# Selectiong features columns
feature_cols = ["CountryIndex", "StockCodeIndex", "Month", "Year",
                "DayOfWeek", "Day", "Week"]

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Using vector assembler to combine features
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")

# Initializing a Random Forest model
rf = RandomForestRegressor(
    featuresCol="features",
    labelCol="Quantity",
    maxBins=4000
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

pipeline = Pipeline(stages=[country_indexer, stock_code_indexer, assembler, rf])

import mlflow
experiment_name = "demand_forecasting_model_experiment_2"
mlflow.set_experiment(experiment_name)

with mlflow.start_run():
   mlflow.autolog()
   # Training the model
   model = pipeline.fit(train_data)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#Save Data Frame
test_data.write.option("header", True).mode("overwrite").csv("Files/output/test_online_retail_data.csv")


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Getting test predictions
test_predictions = model.transform(test_data)
test_predictions = test_predictions.withColumn(
    "prediction", col("prediction").cast("double"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# Initializing the evaluator
mae_evaluator = RegressionEvaluator(
    labelCol="Quantity", predictionCol="prediction", metricName="mae")

# Obtaining MAE
mae = mae_evaluator.evaluate(test_predictions)

print(f"MAE is {mae}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
