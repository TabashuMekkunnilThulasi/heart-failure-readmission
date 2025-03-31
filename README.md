# 💡 GHW Heart Failure Readmission Prediction

This project uses machine learning to predict 30/60-day readmission risk for heart failure patients using demographic, clinical, and hospital visit data.

## 📁 Dataset
- Source: Kaggle (GHW Heart Failure Readmission)
- Includes age, vitals, lab results, hospital stay, etc.
- Target: `Readmission_30or60Days` (1 = Yes, 0 = No)

## ⚙️ Methods
- Data Cleaning and Preprocessing
- SMOTE for class balancing
- Models: Logistic Regression, Random Forest, XGBoost, SVM
- Evaluation using F1 Score, ROC AUC, Confusion Matrix

## 🏆 Best Model
- **Random Forest**
  - F1 Score: 0.55
  - Precision: 0.54
  - Recall: 0.55

## 📊 Insights
- Top features: Creatinine, Ejection Fraction, Sodium, Gender, Ethnicity
- Feature importance extracted using XGBoost

## 📂 Files
- `readmission_prediction.py` — Cleaned final script
- `readmission_prediction.ipynb` — Notebook version
- `GHW_HeartFailure_Readmission_Combined.csv` — Dataset
- `heart_failure_readmission_model.pkl` — Final model
- `presentation/` — PPT slides
- `visuals/` — All graphs

## 🛠️ Tools
- Python, Pandas, Scikit-learn, XGBoost, SMOTE, Matplotlib, Seaborn

## 📌 Author
- Tabashu Thulasi  
- [LinkedIn](https://www.linkedin.com/in/tabashu-mekkunnil-thulasi-681306218)  
- [GitHub](https://github.com/TabashuMekkunnilThulasi)
