import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load data
df = pd.read_csv("creditcard.csv")

# Features & target
X = df.drop("Class", axis=1)
y = df["Class"]

# SMOTE oversampling
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# XGBoost model
model = xgb.XGBClassifier(
    scale_pos_weight=1,  # Since SMOTE balances, no need to tune this heavily
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC-AUC:", roc_auc)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "fraud_model.pkl")
