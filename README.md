💓 GHW Heart Failure Readmission Prediction
This project is part of the Healthcare Analytics domain, aimed at predicting hospital readmissions for heart failure patients using machine learning models.

📌 Project Overview
Hospital readmissions after heart failure are costly and preventable. The goal of this project is to build a predictive model using clinical, demographic, and hospitalization data to identify patients at high risk of being readmitted within 30 or 60 days.

📁 Dataset
Source: Hospital Readmission dataset from Kaggle (GHW Project)

Target Variable: Readmission_30or60Days (1 = Readmitted, 0 = Not Readmitted)

Features:

Demographics: Age, Gender, Ethnicity

Medical History: Prior Admissions, NT_proBNP, Sodium, Creatinine

Vitals: BP, Heart Rate, Temperature

Hospitalization Details: Length of Stay, Discharge Disposition

🧠 Models Used
Logistic Regression

Random Forest

XGBoost

Support Vector Machine (SVM)

Best Performing Model: ✅ Random Forest
F1 Score: 0.55
Recall: 0.55
ROC AUC: 0.52

🧪 Techniques Applied
Data Cleaning and Preprocessing

Handling Missing Values

Feature Encoding and Scaling

SMOTE for Class Imbalance

Hyperparameter Tuning with GridSearchCV

Evaluation using Accuracy, Precision, Recall, F1, ROC AUC

Confusion Matrix Visualization

Feature Importance Analysis (XGBoost)

📊 Exploratory Data Analysis (EDA)
Readmission class distribution:
Readmitted: 50.1% | Not Readmitted: 49.9%

Top Predictive Features (from XGBoost):

Ethnicity_White

Creatinine

Gender_Male

BUN

Ejection Fraction


📈 Visuals Included
📌 Readmission Class Distribution Plot

🔥 Correlation Heatmap

📉 ROC Curve Comparison

🧩 Confusion Matrix (Best Model)

📌 Feature Importance (XGBoost)

👩‍💻 Author
Tabashu Thulasi
Data Analyst & Software Engineer
LinkedIn | GitHub

