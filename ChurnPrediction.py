import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# Convert 'TotalCharges' to numeric and handle missing values
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df.dropna(inplace=True)  # Drop missing values

# Drop 'customerID' as it's not useful for prediction
df.drop(columns=["customerID"], inplace=True)

# Encode categorical variables using One-Hot Encoding
df = pd.get_dummies(df, drop_first=True)

# Define features (X) and target variable (y)
X = df.drop(columns=["Churn_Yes"])
y = df["Churn_Yes"]

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Support Vector Machine": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Train & evaluate models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    print(f"{name} Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    print("-" * 50)

# Feature selection using SelectKBest
selector = SelectKBest(score_func=f_classif, k=10)
X_selected = selector.fit_transform(X_scaled, y)
selected_features = np.array(X.columns)[selector.get_support()].tolist()
print("Top selected features:", selected_features)

# Retrain models with selected features
X_train_fs, X_test_fs, y_train_fs, y_test_fs = train_test_split(X_selected, y, test_size=0.2, random_state=42)
models_fs = {name: model.fit(X_train_fs, y_train_fs) for name, model in models.items()}

# Compare accuracy with and without feature selection
comparison = pd.DataFrame({"Model": results.keys(), "Accuracy (All Features)": results.values()})
comparison["Accuracy (Feature Selected)"] = [accuracy_score(y_test_fs, models_fs[name].predict(X_test_fs)) for name in models.keys()]
print(comparison)

# Visualization of accuracy comparison
bar_width = 0.35
x = np.arange(len(comparison["Model"]))

plt.figure(figsize=(10,5))
plt.bar(x - bar_width/2, comparison["Accuracy (All Features)"], width=bar_width, label="All Features", alpha=0.8)
plt.bar(x + bar_width/2, comparison["Accuracy (Feature Selected)"], width=bar_width, label="Feature Selected", alpha=0.8)
plt.xticks(ticks=x, labels=comparison["Model"], rotation=45)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# Feature importance plot for Random Forest
if "Random Forest" in models_fs:
    rf_model = models_fs["Random Forest"]
    feature_importances = pd.Series(rf_model.feature_importances_, index=selected_features)
    plt.figure(figsize=(8,5))
    feature_importances.nlargest(10).plot(kind='barh')
    plt.title("Top 10 Feature Importances")
    plt.show()

# Decision Tree Visualization
if "Decision Tree" in models:
    plt.figure(figsize=(12,8))
    plot_tree(models["Decision Tree"], feature_names=X.columns, class_names=["No Churn", "Churn"], filled=True)
    plt.title("Decision Tree Visualization")
    plt.show()

# Logistic Regression & SVM Decision Boundary Visualization (Using Top 2 Features)
if len(selected_features) >= 2:
    plt.figure(figsize=(10,5))
    X_vis = X_train_fs[:, :2]  # Select first two features for visualization
    y_vis = y_train_fs
    
    for name in ["Logistic Regression", "Support Vector Machine"]:
        model = models[name]
        model.fit(X_vis, y_vis)
        x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
        y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.3)
        plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, edgecolors='k')
        plt.title(f"Decision Boundary: {name}")
        plt.xlabel(selected_features[0])
        plt.ylabel(selected_features[1])
        plt.show()

# K-Nearest Neighbors Visualization
if "K-Nearest Neighbors" in models:
    plt.figure(figsize=(8,6))
    sns.scatterplot(x=X_train_fs[:, 0], y=X_train_fs[:, 1], hue=y_train_fs, palette='coolwarm')
    plt.title("K-Nearest Neighbors Clustering")
    plt.xlabel(selected_features[0])
    plt.ylabel(selected_features[1])
    plt.show()


