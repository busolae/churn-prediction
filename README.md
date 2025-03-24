# 📖 Overview
This project predicts customer churn in the telecom industry using five classification machine learning models. The goal is to identify customers likely to leave based on behavioral and demographic features.

# 🛠️ Technologies Used
Python (pandas, NumPy, scikit-learn, Matplotlib, Seaborn)
Machine Learning Models:
Logistic Regression
Decision Tree
Random Forest
Support Vector Machine (SVM)
K-Nearest Neighbors (KNN)

# 📂 Dataset
Source: Telecom Churn Dataset (Replace with actual link if available)
Size: ~1,000+ customer records
Features:
Demographics: Age, contract type, payment method
Usage Patterns: Call minutes, internet usage, number of customer service calls
Target Variable: Churn (Yes/No)

# 📊 Feature Selection & Model Evaluation
Used SelectKBest (f_classif) to identify important features.
Evaluated models using accuracy, precision, recall, F1-score, and ROC-AUC.
Compared models with and without feature selection to see improvement.

# 🚀 How to Run the Project
1️⃣ Clone the Repository
```sg
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
```
2️⃣ Install Dependencies
```sh
pip install pandas numpy scikit-learn matplotlib seaborn
```
3️⃣ Run the Python Script
```sh
python ChurnPrediction.py
```

# 📈 Results
Best Model: Random Forest (Achieved 85% accuracy) (Adjust based on your actual results)
Key Features Affecting Churn: Tenure, contract type, customer service calls
