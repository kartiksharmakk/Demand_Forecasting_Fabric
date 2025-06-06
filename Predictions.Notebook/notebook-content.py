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

# CELL ********************

import mlflow


# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

#load data


test_sales_data = spark.read.format("csv").option("header","true").load("Files/output/test_online_retail_data.csv")

display(test_sales_data)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Specify the registered model URI
model_uri = "models:/demand_forecasting_model_2/1"

# Load the model
model = mlflow.spark.load_model(model_uri)

#convert into Int
from pyspark.sql.functions import col

test_sales_data = test_sales_data.withColumn("Year", col("Year").cast("int")) \
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



# Apply the model to the test Spark DataFrame
test_predictions = model.transform(test_sales_data)



test_predictions = test_predictions.withColumn(
    "prediction", col("prediction").cast("double"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************


# Getting the weekly sales of all countries
weekly_test_predictions = test_predictions.groupBy("Year", "Week").agg({"prediction": "sum"})

# Finding the quantity sold on the 39 week. 
promotion_week = weekly_test_predictions.filter(col('Week')==39)

# Storing prediction as quantity_sold_w30
quantity_sold = int(promotion_week.select("sum(prediction)").collect()[0][0])
print(f"How many units will be sold during the  week 39 of 2011 : {quantity_sold}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
