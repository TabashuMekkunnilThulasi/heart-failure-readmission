# Heart Failure Readmission Prediction - Final Enhanced Script

import warnings
warnings.filterwarnings("ignore")

import os
os.environ['XGBOOST_ENABLE_WARNINGS'] = '0'

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
import joblib

# Step 1: Load & Prepare Data
data = pd.read_csv("C:/Users/HP/GHW_HeartFailure_Readmission_Combined.csv")
print("Missing Values:\n", data.isnull().sum())

data.drop("Patient_ID", axis=1, inplace=True)
target = data["Readmission_30or60Days"]
data.drop(["Readmission_30Days", "Readmission_60Days", "Readmission_30or60Days"], axis=1, inplace=True)

data.fillna(data.median(numeric_only=True), inplace=True)
data = pd.get_dummies(data, columns=["Gender", "Ethnicity", "Discharge_Disposition"], drop_first=True)

# Step 2: EDA
plt.figure(figsize=(6, 4))
sns.countplot(x=target)
plt.title("Readmission Class Distribution")
plt.xlabel("Readmitted (1) vs Not (0)")
plt.tight_layout()
plt.show()

# Add numeric summary
class_counts = target.value_counts()
class_percent = round(target.value_counts(normalize=True) * 100, 2)
print(f"\nReadmission Counts:\n{class_counts}")
print(f"\nReadmission Percentage:\n{class_percent}")

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pd.concat([data, target], axis=1).corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Step 3: Train-Test Split and SMOTE
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, target, test_size=0.2, stratify=target, random_state=42
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Show SMOTE class balance
print("\nSMOTE Resampled Class Distribution:")
print(pd.Series(y_train_resampled).value_counts())

# Step 4: Model Definitions
models_config = {
    "Logistic Regression": {
        "estimator": LogisticRegression(class_weight="balanced", max_iter=1000),
        "params": {"C": [0.1, 1, 10]}
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(class_weight="balanced", random_state=42),
        "params": {"n_estimators": [100], "max_depth": [5, 10]}
    },
    "XGBoost": {
        "estimator": XGBClassifier(eval_metric="logloss", random_state=42),
        "params": {"n_estimators": [100], "max_depth": [3], "learning_rate": [0.1]}
    },
    "SVM": {
        "estimator": SVC(kernel="rbf", probability=True),
        "params": {"C": [0.1, 1], "gamma": ["scale"]}
    }
}

# Step 5: Train and Evaluate Models
best_models = {}
model_scores = {}

for model_label, config in models_config.items():
    print(f"\n🔍 Training {model_label}...")
    clf = GridSearchCV(config["estimator"], config["params"], cv=5, scoring="f1", n_jobs=-1)
    clf.fit(X_train_resampled, y_train_resampled)

    predictions = clf.predict(X_test)
    probas = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions)
    rec = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    auc = roc_auc_score(y_test, probas) if probas is not None else 0.5

    print(f"Best Params: {clf.best_params_}")
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC AUC: {auc:.2f}")

    best_models[model_label] = clf
    model_scores[model_label] = f1  # F1 used for model selection

# Step 6: Best Model Selection
best_model_name = max(model_scores, key=model_scores.get)
print(f"\n✅ Best Performing Model Based on F1 Score: {best_model_name}")
final_model = best_models[best_model_name]
final_preds = final_model.predict(X_test)

# Confusion Matrix
cm = confusion_matrix(y_test, final_preds)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion Matrix Breakdown:\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")

plt.figure()
ConfusionMatrixDisplay.from_predictions(y_test, final_preds, display_labels=["Not Readmitted", "Readmitted"], cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()
plt.show()

# Step 7: ROC Curve
plt.figure(figsize=(10, 6))
for name, model in best_models.items():
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, probas)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc_score(y_test, probas):.2f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curve - Readmission Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()

# Step 8: XGBoost Feature Importance
if "XGBoost" in best_models:
    xgb_model = best_models["XGBoost"].best_estimator_
    importance_df = pd.DataFrame({
        "Feature": data.columns,
        "Importance": xgb_model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    print("\nTop 5 Important Features (XGBoost):")
    print(importance_df.head())

    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df.head(10), x="Importance", y="Feature")
    plt.title("Top 10 Feature Importance - XGBoost")
    plt.tight_layout()
    plt.show()

# Step 9: Save Model and Scaler
joblib.dump(final_model, "heart_failure_readmission_model.pkl")
joblib.dump(scaler, "scaler.pkl")
