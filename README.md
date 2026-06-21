# Bank Customer Churn Predictive Engine

An end-to-end data science classification workspace designed to mitigate retail bank customer attrition. This repository features a structured machine learning pipeline that ingests historical customer portfolios, handles data cleaning, scales feature constraints, and benchmarks ensemble classifiers against simple tree models to maximize retention precision.

## Dataset Description
The model operates on a `Bank_Churn.csv` dataset containing 10,000 distinct customer transactions and profiles distributed across 13 base attributes:
* **Target Label:** `Exited` (Binary flag: `1` if the customer exited the bank, `0` if they remained active).
* **Demographic Vectors:** `Geography` (e.g., France, Spain), `Gender`, and `Age`.
* **Financial Attributes:** `CreditScore`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, and `EstimatedSalary`.
* **Customer Identifiers:** `CustomerId`, `Surname`, and `Tenure` (Years with the bank).

## Methodology
1. **Exploratory Data Integrity Check:** Analyzed column datatypes across the 10,000 observations, validating a clean data footprint with zero missing/null values and zero duplicated row instances.
2. **Feature Pruning & Engineering:** Segregated data vectors into feature matrices ($X$) and target classifications ($y$). (Non-predictive unique keys like `CustomerId` and text markers like `Surname` are dropped during preprocessing). Categorical fields (`Geography`, `Gender`) are prepared for encoding.
3. **Imbalance Analysis:** Monitored baseline target label counts, revealing a structural minority class imbalance with **7,963 staying clients** versus **2,037 exited accounts** (~20.4% churn rate).
4. **Algorithmic Benchmarking Design:** Designed a model evaluation pipeline to train and compare performance metrics across three classification frameworks:
   * **Decision Tree Classifier:** Serving as a baseline predictive tree constraint.
   * **Random Forest Classifier:** Deployed as an ensemble bootstrap aggregator to balance out single-tree variance.
   * **Logistic Regression / Gradient Boosting:** Implemented as a baseline comparator to track performance surfaces.

## Evaluation & Metrics
Models are measured using strict classification criteria including Accuracy, Precision, Recall, and F1-Scores. Given the ~20% class imbalance found in the source labels, special emphasis is placed on optimization metrics to ensure churn clients are correctly isolated without causing heavy false-alarm rates.

## How to Run Locally
1. Clone this repository to your active machine environment directory.
2. Verify you have Python installed alongside standard analytical packages:
   ```bash
   pip install numpy pandas scikit-learn jupyter
