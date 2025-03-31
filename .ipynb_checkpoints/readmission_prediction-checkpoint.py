# Heart Failure Readmission Prediction - Python Script

import warnings
warnings.filterwarnings("ignore")  # Suppress warnings from libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import joblib

# Load dataset
df = pd.read_csv("C:/Users/HP/GHW_HeartFailure_Readmission_Combined.csv")

# Check for missing values
print("Missing Values:\n", df.isnull().sum())

# Drop Patient_ID and separate target
df.drop("Patient_ID", axis=1, inplace=True)
y = df["Readmission_30or60Days"]
df.drop(["Readmission_30Days", "Readmission_60Days", "Readmission_30or60Days"], axis=1, inplace=True)

# Fill missing values with median (if any)
df.fillna(df.median(numeric_only=True), inplace=True)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=["Gender", "Ethnicity", "Discharge_Disposition"], drop_first=True)

# EDA - Visualization
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Readmission Class Distribution")
plt.xlabel("Readmitted (1) vs Not (0)")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
corr_data = pd.concat([df, y], axis=1)
sns.heatmap(corr_data.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=1000),
    "Random Forest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "XGBoost": XGBClassifier(objective='binary:logistic', eval_metric='logloss', verbosity=0, random_state=42)
}

# Hyperparameters for GridSearch
params = {
    "Logistic Regression": {"C": [0.1, 1, 10], "penalty": ["l2"]},
    "Random Forest": {"n_estimators": [100, 200], "max_depth": [None, 5, 10]},
    "XGBoost": {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]}
}

# Train and evaluate models
best_models = {}
for name in models:
    print(f"\nTraining {name}...")
    grid = GridSearchCV(models[name], params[name], cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    y_pred = grid.predict(X_test)
    print(f"Best params: {grid.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.2f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_pred):.2f}")

    # Confusion Matrix
    plt.figure()
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Readmitted", "Readmitted"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.show()

# Save the best model (XGBoost assumed best here)
xgb_best = best_models["XGBoost"]
joblib.dump(xgb_best, "heart_failure_readmission_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Plot ROC curves
plt.figure(figsize=(10, 6))
for name in best_models:
    model = best_models[name]
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Readmission Prediction")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

# Plot XGBoost Feature Importance
feature_importance = pd.DataFrame({
    'Features': df.columns,
    'Importance': xgb_best.feature_importances_
}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Features')
plt.title("Feature Importance - XGBoost")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.tight_layout()
plt.show()
