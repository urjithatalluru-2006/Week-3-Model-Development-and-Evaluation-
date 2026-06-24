import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# =====================
# LOAD DATA
# =====================

df = pd.read_csv("automobile.csv.csv")

print(df.head())

# =====================
# PREPROCESSING
# =====================

df = pd.get_dummies(df, drop_first=True)

# =====================
# TARGET VARIABLE
# =====================

X = df.drop("price", axis=1)

y = df["price"]

# =====================
# TRAIN TEST SPLIT
# =====================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# =====================
# MODEL
# =====================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# =====================
# PREDICTIONS
# =====================

predictions = model.predict(X_test)

# =====================
# EVALUATION
# =====================

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print("\nMODEL PERFORMANCE")

print("RMSE:", rmse)

print("MAE:", mae)

print("R2 Score:", r2)

# =====================
# ACTUAL VS PREDICTED
# =====================

plt.figure(figsize=(8,6))

plt.scatter(
    y_test,
    predictions
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title(
    "Actual vs Predicted Price"
)

plt.savefig(
    "actual_vs_predicted.png"
)

plt.show()

# =====================
# FEATURE IMPORTANCE
# =====================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

top10 = importance.head(10)

plt.figure(figsize=(10,6))

sns.barplot(
    data=top10,
    x="Importance",
    y="Feature"
)

plt.title(
    "Top 10 Important Features"
)

plt.savefig(
    "feature_importance.png"
)

plt.show()

print("\nModel Completed Successfully!")
