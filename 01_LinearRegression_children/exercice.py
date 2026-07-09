import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import os
import numpy as np

# Build an absolute path to the CSV file so the script works
# regardless of the current working directory
current_folder = os.path.dirname(os.path.abspath(__file__))
path_csv = os.path.join(current_folder, "data", "age_vs_poids_vs_taille_vs_sexe.csv")
children = pd.read_csv(path_csv)

# Quick sanity check on the data: preview rows and summary stats
print(children.head())
print(children.describe())

# --- Model 1: predict weight from age only ---
X = children[["age"]]
y= children["poids"]
scores = {} # store R2 scores for each model to compare them later

reg = LinearRegression()
reg.fit(X, y)
print(reg.coef_)
scores["age"] = reg.score(X, y)
print(scores["age"], "\n")

# --- Model 2: predict weight from sexe + age ---
X = children[["sexe", "age"]]
reg.fit(X, y)
print(reg.coef_)
scores["age+sexe"] = reg.score(X, y)
print(scores["age+sexe"], "\n")

# --- Model 3: predict weight from sexe + age + taille ---
X = children[["sexe", "age", "taille"]]
reg.fit(X, y)
print(reg.coef_)
scores["age+sexe+taille"] = reg.score(X, y)
print(scores["age+sexe+taille"], "\n")

# Evaluate the full model (sexe + age + taille) with error metrics
y_pred = reg.predict(X)
mse = mean_squared_error(y, y_pred)
print(f"MSE = {mse}\n")
rmse = mse**0.5 # RMSE = square root of MSE
print(f"RMSE = {rmse}\n")
mae = mean_absolute_error(y, y_pred)
print(f"MAE = {mae}\n")
mape = mean_absolute_percentage_error(y, y_pred)
print(f"MAPE = {mape}\n")

# Predict the weight of a new child: sexe=0 (boy), age=224 months, size=175 cm
weight = reg.predict(np.array([[0, 224, 175]]))
print(weight)