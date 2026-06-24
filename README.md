# Week 3: Model Development and Evaluation

## 1. Objective

The objective of this project is to develop and evaluate a machine learning model capable of predicting automobile prices using vehicle specifications and characteristics. The project demonstrates the complete workflow from data preparation to model evaluation.

---

## 2. Dataset Description

The automobile dataset contains information about vehicles including:

* Car Name
* Fuel Type
* Engine Size
* Horsepower
* Wheelbase
* Vehicle Dimensions
* Fuel Efficiency
* Price

Dataset Size:

* Rows: 211
* Columns: 25

Target Variable:

* Price

---

## 3. Data Preparation

Before model development, the dataset was prepared using the following steps:

### Categorical Encoding

Categorical variables were converted into numerical representations using One-Hot Encoding.

```python
df = pd.get_dummies(df, drop_first=True)
```

### Feature Selection

Predictor variables (X) consisted of all vehicle attributes except Price.

Target variable (y):

* Price

---

## 4. Train-Test Split

The dataset was divided into training and testing sets.

* Training Data: 80%
* Testing Data: 20%

```python
train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

Reason:
The 80/20 split provides sufficient data for training while preserving an independent dataset for evaluating model performance.

Training Samples: 168

Testing Samples: 42

---

## 5. Model Selection

A Random Forest Regressor was selected as the baseline model.

Reasons for Selection:

* Handles non-linear relationships effectively.
* Performs well on tabular datasets.
* Resistant to overfitting compared to a single decision tree.
* Provides feature importance rankings.

Model Parameters:

```python
RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
```

---

## 6. Model Evaluation

The model was evaluated using:

### Root Mean Squared Error (RMSE)

RMSE = 2574.70

Interpretation:
Measures the average prediction error while penalizing larger mistakes.

### Mean Absolute Error (MAE)

MAE = 1921.47

Interpretation:
Average difference between predicted and actual prices.

### R² Score

R² = 0.912

Interpretation:
The model explains approximately 91.2% of the variation in automobile prices.

---

## 7. Visual Evaluation

### Actual vs Predicted Prices

Insert:
actual_vs_predicted.png

Observation:
Predicted prices closely follow actual prices, indicating strong model performance.

### Feature Importance Analysis

Insert:
feature_importance.png

Observation:
Engine-related features, vehicle weight, and fuel efficiency contribute significantly to price prediction.

---

## 8. Strengths and Weaknesses

### Strengths

* High predictive accuracy.
* Strong R² score.
* Handles complex feature interactions.
* Provides feature importance information.

### Weaknesses

* Requires more computational resources than simple regression.
* Less interpretable than linear models.
* Performance may decrease with unseen vehicle types.

---

## 9. Challenges Encountered

* Handling categorical variables.
* Selecting an appropriate train-test split.
* Interpreting model evaluation metrics.
* Identifying influential features.

---

## 10. Future Improvements

Potential improvements include:

* Hyperparameter tuning using Grid Search.
* Cross-validation for more robust evaluation.
* Additional feature engineering.
* Testing advanced models such as XGBoost or Gradient Boosting.
* Collecting larger and more diverse automotive datasets.

---

## 11. Conclusion

A Random Forest Regression model was successfully developed to predict automobile prices. The model achieved an R² score of 0.912, demonstrating strong predictive capability. Evaluation metrics indicate that the model performs effectively and can serve as a reliable baseline for future automotive price prediction projects.
