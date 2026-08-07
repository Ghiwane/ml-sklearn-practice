import seaborn as sns 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score

# Load the Titanic dataset directly from seaborn
titanic = sns.load_dataset("titanic")
print(titanic.head(), "\n\n")
print(titanic.describe(), "\n\n")

# Impute missing values with the median for age and fare
titanic["age"] = titanic["age"].fillna(titanic["age"].median())
titanic["fare"] = titanic["fare"].fillna(titanic["fare"].median())

# Encode the categorical 'sex' column into numeric values
titanic['sex'] = titanic['sex'].map({'male': 1, 'female': 0})

# Select predictors and target
X = titanic[["pclass", "sex", "age", "sibsp", "parch", "fare"]]
y = titanic["survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=7)

# Train the logistic regression model
clf_titanic = LogisticRegression(random_state=808).fit(X_train, y_train)
y_pred = clf_titanic.predict(X_test)

# Look at predicted probabilities for 2 specific passengers
proba_passenger_A = clf_titanic.predict_proba(X_test.iloc[[2]])
proba_passenger_B = clf_titanic.predict_proba(X_test.iloc[[50]])
print(proba_passenger_A, proba_passenger_B, "\n\n")

# Predicted probabilities for the positive class (survived) on the test set
y_hat_proba = clf_titanic.predict_proba(X_test)[:,1]
sns.histplot(y_hat_proba)
plt.show()

# Compare accuracy/precision/recall across different classification thresholds
thresholds_list = [0.5, 0.3, 0.7]
metrics_dict = {}

for t in thresholds_list:
    y_pred_t = [0 if val < t else 1 for val in y_hat_proba]
    
    metrics_dict[f"threshold_{t}"] = {
        "accuracy": accuracy_score(y_test, y_pred_t),
        "precision": precision_score(y_test, y_pred_t, zero_division=0),
        "recall": recall_score(y_test, y_pred_t, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred_t)
    }

df_metrics = pd.DataFrame(metrics_dict).T
print(df_metrics[["accuracy", "precision", "recall"]])

# Plot the ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_hat_proba)
plt.plot(fpr, tpr)
plt.show()

# Compute the ROC-AUC score (threshold-independent performance metric)
auc_score = roc_auc_score(y_test, y_hat_proba)
print(f"ROC-AUC score on the test set: {auc_score:.4f}")