# 🚗 Used Car Price Prediction using Linear Regression

## Overview

This project implements a **regression-based Machine Learning pipeline** to predict the selling price of used cars in the Indian market using real-world vehicle data.

The repository documents the project in **two clear stages**:

* **v1 (Baseline):** Linear Regression with feature engineering and diagnostics
* **v2 (Current):** Regularized Linear Models (Ridge & Lasso) with cross-validated hyperparameter tuning

The focus of this project is **conceptual clarity, disciplined preprocessing, and correct model evaluation**, not aggressive optimization or deployment.

---

## 🚀 Live Demo
**Try the deployed app here:**  
👉 [Used Car Price Predictor — Live on HuggingFace Spaces](https://huggingface.co/spaces/ManojSabale/Car_Price_Predictor)

Enter car details and get an instant price estimate powered by the XGBoost model.

## Objective

The primary objectives of this project are to:

* Understand and implement the complete ML regression pipeline
* Handle real-world data issues (skewness, missing values, categorical variables)
* Apply feature transformations to improve regression assumptions
* Compare baseline and regularized linear models
* Use cross-validation for robust model selection
* Develop strong intuition around bias–variance tradeoffs

---

## Problem Statement

The Indian used-car market lacks a standardized pricing mechanism.
Prices for similar cars often vary due to subjective valuation, inconsistent market knowledge, and information asymmetry.

This project aims to **predict a fair selling price** for used cars based on historical listings and vehicle attributes.

---

## Dataset

| Attribute           | Details             |
| ------------------- | ------------------- |
| Source              | CarDekho (Kaggle)   |
| Rows                | ~8,100              |
| Original Features   | 13                  |
| Final Feature Count | 44 (after encoding) |
| Target Variable     | `selling_price`     |

### Key Features Used

* Year of manufacture
* Kilometers driven
* Fuel type
* Seller type
* Transmission
* Ownership history
* Mileage, engine capacity, max power (cleaned from text)
* Brand (extracted from vehicle name)

---

## Key Challenges Addressed

* Mixed data types (numeric values stored as strings with units)
* Missing values across multiple columns
* Highly right-skewed target variable (`selling_price`)
* High-cardinality categorical features (car brands)
* Multicollinearity among technical features
* Outliers caused by luxury vehicles

---

## Approach

### 1. Data Cleaning & Preprocessing

* Extracted numeric values from text-based columns
* Handled missing values using appropriate imputation
* Dropped `torque` due to inconsistent formatting and limited predictive value
* Extracted car brand from vehicle name
* Created `car_age` from manufacturing year

### 2. Encoding & Scaling

* One-Hot Encoding for nominal categorical variables
* Ordinal Encoding for ownership history
* Feature scaling using `StandardScaler` for regularized models

### 3. Feature Transformation

* Log transformation applied to:

  * `selling_price` (target)
  * `km_driven`
* Reduced skewness and improved regression assumptions

### 4. Model Building

* Linear Regression (baseline)
* Ridge Regression (L2 regularization)
* Lasso Regression (L1 regularization)

### 5. Hyperparameter Tuning

* GridSearchCV with 5-fold cross-validation
* Tuned regularization strength (`alpha`)
* Models compared on validation and test performance

---

## Model Performance Summary

## Model Performance Summary

| Model             | Test R² | Observation                                    |
|-------------------|---------|------------------------------------------------|
| Linear Regression | 0.8809  | Strong baseline                                |
| Ridge Regression  | 0.8829  | Improved coefficient stability                 |
| Lasso Regression  | 0.8569  | Slight underfitting                            |
| Decision Tree     | 0.8900  | Good but prone to overfitting                  |
| Random Forest     | 0.9163  | Strong ensemble performance                    |
| XGBoost           | 0.9221  | Best model — selected for deployment           |

**Winner: XGBoost (R² = 92.21%)**
### Key Observations

* All models achieved similar performance (~88% R²)
* Regularization improved stability rather than raw accuracy
* Low optimal alpha values indicate effective preprocessing
* Elastic Net reduced to Ridge behavior (`l1_ratio = 0`)

---

## Evaluation Metrics

* R² Score
* Adjusted R²
* MAE, MSE, RMSE

> Error-based metrics were computed after inverse-transforming the target variable to maintain interpretability.

---

## Project Structure

Car_price_prediction_LR/
│
├── data/
│   └── Vehicle.csv
│
├── notebooks/
│   ├── v1_baseline/
│   │   └── Car_price_prediction_LR.ipynb
│   │
│   └── v2_regularized/
│       └── Used_Car_price_prediction_LR.ipynb
│
├── models/
│   └── car_price_model.pkl
│
├── README.md
└── requirements.txt


---

## Project Versions

* **v1:** Baseline Linear Regression with feature engineering and diagnostics
* **v2:** Regularized Linear Models with cross-validated hyperparameter tuning

The stored model represents the **final regularized version** selected after comparison with the baseline.

---

## Key Learnings

* Preprocessing can contribute more than model complexity
* Regularization improves robustness, not necessarily accuracy
* Log transformations significantly improve regression behavior
* Feature stability matters as much as predictive performance
* Model selection should be driven by data characteristics

---

## Improvements Made
- ✅ Trained and compared 6 models including ensemble methods
- ✅ XGBoost selected as final model (R² = 92.21%)
- ✅ Deployed as interactive web app via Streamlit + HuggingFace Spaces

## Future Improvements
- Add SHAP explainability to show which features drive each prediction
- Collect more luxury vehicle data to improve high-end price accuracy
- Add confidence intervals to predictions
- Retrain periodically as market prices shift

---

## Author

**Mas**
B.E. Artificial Intelligence & Data Science

---

## Final Note

This repository represents a **learning milestone**, not a production system.
The focus is on **strong ML fundamentals, clean implementation, and correct reasoning** rather than maximum accuracy.

---
