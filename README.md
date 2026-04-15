# Used Cars Price Prediction

## Overview
This project aims to predict used car prices using a Craigslist dataset from Kaggle.  
It is a regression task with a full machine learning pipeline, including data cleaning, feature engineering, model selection, tuning, and interpretation.

---

## Dataset
- Source: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data  
- ~426k rows, 25 features  
- Contains many missing values, outliers, and high-cardinality categorical features  

Some features were removed early (IDs, URLs, description, etc.) as they do not provide useful information for price prediction.

---

## Data Cleaning

Key steps:

- Removed invalid data:
  - price = 0
  - incorrect latitude/longitude
- Filtered unrealistic values:
  - price < 1000 (car parts, not cars)
- Capped values (outliers removal):
  - odometer capped at 400,000
  - price capped at 95th percentile
- Year filtering:
  - only cars from year ≥ 2000  
  - older cars behave differently (different price formation)

Outliers handling:
- Optional parameter: `remove_outliers`
- If enabled:
  - removes extreme price and odometer values
- If disabled:
  - only basic filtering is applied (still keeps data realistic)

Additional processing:
- Categorical values → lowercase, cleaned
- Rare manufacturers grouped into "rare"
- `cylinders` converted to numeric + missing indicator feature
- `model` NaNs replaced with empty strings (for TF-IDF)

---

## Feature Engineering

- `model` feature:
  - encoded using **TF-IDF (max 5000 features)**
  - chosen due to high cardinality (~19k unique values)

- Numerical features:
  - standard scaling (for linear models only)
  - median imputation

- Categorical features:
  - OneHotEncoding
  - most frequent imputation
 
- Catboost categorical features:
  - imputed Nan values with *missing* category

- Special handling:
  - log transformation applied to `odometer` (linear models)
  - binary feature for missing `cylinders`

---

## Train/Test Split

Three approaches were considered:

- Stratified (main method)
- Random
- Hash-based (for future data scenarios)

Final choice:
- **Stratified split (based on price bins)**  
Reason: price distribution is uneven, balanced train/test sets are wanted.

---

## Models

Compared models:

- DummyRegressor (baseline)
- Linear Regression (log target)
- Decision Tree (log target)
- Gradient Boosting (log target)
- XGBoost
- CatBoost

Evaluation:
- 5-fold cross-validation (RMSE)

Results:

| Model              | RMSE |
|--------------------|------|
| Baseline           | 11567 |
| Linear Regression  | 5019 |
| Decision Tree      | 6653 |
| Gradient Boosting  | 5431 |
| XGBoost            | 4355 |
| CatBoost           | 4034 |

---

## Model Selection & Tuning

Best candidates:
- XGBoost
- CatBoost

Tuning:
- RandomizedSearchCV
- Focus on:
  - learning rate
  - tree depth
  - number of estimators
  - regularization

Final choice:
- **XGBoost**

---

## Final Result

- CV RMSE: ~3306  
- Test RMSE: **~4188**
- 
The model does not generalize perfectly, but this behavior is expected for tree‑based boosting models trained on noisy, skewed price data. The presence of rare high‑value cars naturally increases test error.

---

## Experiments

### Outliers impact

- Without outliers:
  - RMSE ≈ 4188
- With outliers:
  - RMSE ≈ 11362260 

Strong evidence that outlier removal is critical

---

### Split strategy impact

- Stratified split gives stable results  
- Random split resulted in slightly lower RMSE (~4145), but:
  - likely caused by different price distribution in the test set
  - overall, random split is less reliable due to uneven price distribution  

---

## Model Interpretation

- Feature importance (top 20)
- SHAP summary plot (2000 samples)

Key insights:
- Numerical features have strongest impact:
  - year
  - odometer
  - cylinders
- Text features (TF-IDF) also contribute
- Total features: ~5000+

---

## How to Run

1. Clone the repository:
   git clone https://github.com/yourusername/used-cars-price-prediction.git

2. Install dependencies:
   pip install -r requirements.txt

3. Set up Kaggle API credentials:
   - Place your kaggle.json file in the appropriate directory

4. Run the notebook:
   - Open the notebook and execute all cells


## Tech Stack

- Python
- scikit-learn
- XGBoost
- CatBoost
- SHAP
- Pandas / NumPy
