# Demand Forecasting Using Microsoft Fabric Lakehouse, PySpark ML & MLflow

## Problem Statement

Demand forecasting is a critical aspect of supply chain management, enabling businesses to predict future product demand accurately. This project aims to build a demand forecasting model using Microsoft Fabric Lakehouse, PySpark ML, and MLflow for tracking experiments and managing the machine learning lifecycle.

While online shopping offers instant convenience to customers, e-commerce companies face the difficult challenge of accurately predicting product demand. Demand uncertainty leads to stockouts, missed deliveries, and costly overstocking. We are developing a machine learning-based demand forecasting model to optimize inventory levels ahead of the year-end sales season.


## Tech Stack Overview

#### Microsoft Fabric Lakehouse

#### Fabric Data Pipelines

#### PySpark for Data Engineering and ML

#### MLflow for experiment tracking and model management

#### Random Forest Regressor (Spark MLlib)

#### GitHub integration for version control




## Data Pipeline Architecture
To automate the end-to-end process, I designed a Microsoft Fabric Data Pipeline consisting of the following stages:

1. Delete Activity
Clears old artifacts from the Lakehouse to ensure a clean training environment.

2. Copy Data Activity
Fetches a raw CSV file containing historical sales data directly from a GitHub repository into the Lakehouse Files.

3. Notebook Activity
Executes the training notebook that preprocesses the data and trains a machine learning model.

## Data Engineering & Preparation
Inside the notebook, the following steps were taken to prepare the dataset for training:

#### Data Loading: Read historical transaction records from the CSV file.

#### Date Handling: Converted raw timestamps into standard date format and derived features such as day, week, and month.

#### Aggregation: Aggregated data daily by country and product to compute total quantity sold and average unit price.

#### Train-Test Split: Created a time-based split using 2011-09-25 as the cutoff date for training and testing sets.

## Feature Engineering:

Encoded Country and StockCode using StringIndexer

Extracted date-based features (Year, Month, Week, Day of Week)

Used VectorAssembler to compile features into a single vector

## Model Training & Experiment Tracking with MLflow
To manage model lifecycle and track performance, MLflow was integrated within the notebook. Here's what was done:

1. Created and named an MLflow experiment: demand_forecasting_model_experiment_2

2. Enabled autologging to automatically track

3. Model parameters and metrics (e.g., MAE)

4. Feature schema and pipeline stages

5. Versioned model artifacts

6. Trained the Random Forest Regressor using a PySpark pipeline



## Model Deployment and Reuse
In a separate inference notebook, I reused the registered model via MLflow’s model URI:


1. Used the model to generate predictions on test data

2. Aggregated predictions by week

3. Estimated units sold during week 39 of 2011, a key promotion window

📈 Predicted Units Sold in Week 39, 2011: quantity_sold 

## Version Control with GitHub Integration
To ensure code versioning, collaboration, and traceability, I linked my Microsoft Fabric workspace to GitHub:

#### Synced notebooks and pipeline definitions with a connected GitHub repository

#### Used GitHub commits and branches to manage different stages of experimentation

#### Maintained transparency and auditability of all pipeline changes

#### This integration streamlined the development lifecycle and ensured that model and data engineering workflows were collaboratively developed and source-controlled.


## Conclusion
This end-to-end project illustrates the powerful synergy of Microsoft Fabric, Lakehouse architecture, PySpark ML, and MLflow:

✅ Automated pipeline orchestration

✅ Scalable, interpretable ML model

✅ Tracked experiments and model versions via MLflow

✅ Reusable models with easy deployment

✅ GitHub integration for version-controlled development

This demand forecasting solution empowers the business to make accurate, timely decisions and maximize sales performance during critical retail periods.