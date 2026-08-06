import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import os
import numpy as np

# Build an absolute path to the CSV file so the script works
current_folder = os.path.dirname(os.path.abspath(__file__))
path_csv = os.path.join(current_folder, "data", "advertising.csv")
ads = pd.read_csv(path_csv)

# Quick sanity check on the data: preview rows and summary stats
print(ads.head(), "\n\n")
print(ads.describe(), "\n\n")

# Correlation matrix to spot which feature relates most strongly to sales
corr_matrice = ads.corr()
print(corr_matrice, "\n")
print(f"The variable most strongly correlated with 'sales' is : TV\n\n")

score={} # store R2 scores for each model to compare them later

# --- Model 1: predict sales from TV budget only ---
X = ads[["tv"]]
y = ads["ventes"]
reg = LinearRegression()
reg.fit(X, y)
score["tv"] = reg.score(X,y)
print(score["tv"])

# --- Model 2: predict sales from TV + radio budget ---
X = ads[["tv", "radio"]]
reg2 = LinearRegression()
reg2.fit(X, y)
score["tv+radio"] = reg2.score(X,y)
print(score["tv+radio"])

# --- Model 3: predict sales from TV + radio + newspaper budget ---
X = ads[["tv", "radio", "journaux"]]
reg3 = LinearRegression()
reg3.fit(X, y)
score["tv+radio+journaux"] = reg3.score(X,y)
print(score["tv+radio+journaux"], "\n\n")

# Split the full model's data into train/test sets to evaluate
# generalization on unseen data instead of the training data itself
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
reg_final = LinearRegression()
reg_final.fit(X_train, y_train)
y_pred = reg_final.predict(X_test)

# Evaluate predictions against the true test values
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.2f}%\n\n")

# add an interaction term to capture potential
# synergy between TV and radio budgets (their combined effect may be
# stronger than the sum of their individual effects)
ads["tv_x_radio"] = ads["tv"] * ads["radio"]
X = ads[["tv", "radio", "tv_x_radio"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
reg_final2 = LinearRegression()
reg_final2.fit(X_train, y_train)
y_pred = reg_final2.predict(X_test)

# Evaluate the interaction model the same way, to compare directly
# against the model without the interaction term
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)  
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred) * 100

print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"MAPE : {mape:.2f}%")