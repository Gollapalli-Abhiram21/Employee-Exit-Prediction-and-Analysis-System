import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# -------------------------------
# Setup
# -------------------------------
data_path = "dataset/HR_comma_sep.csv"
output_dir = "output"
plots_dir = os.path.join(output_dir, "plots")
reports_dir = os.path.join(output_dir, "reports")

# Ensure directories exist
os.makedirs(plots_dir, exist_ok=True)
os.makedirs(reports_dir, exist_ok=True)

# -------------------------------
# Load Data
# -------------------------------
df = pd.read_csv(data_path)
print(f"Dataset loaded with shape: {df.shape}")

# -------------------------------
# Preprocessing
# -------------------------------
# Handle missing values
if df.isnull().sum().sum() > 0:
    print("Missing values detected. Handling...")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
else:
    print("No missing values found.")

# Encode categorical variables
label_encoders = {}
for col in ['Department', 'salary']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
X = df.drop('left', axis=1)
y = df['left']

# Scale numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

# -------------------------------
# Model Training
# -------------------------------
# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------
metrics = []
for model_name, y_pred in [('Logistic Regression', y_pred_lr), ('Random Forest', y_pred_rf)]:
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    metrics.append(
        f"{model_name}:\nAccuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1 Score: {f1:.4f}\n")

# Save metrics in reports folder
metrics_file = os.path.join(reports_dir, "model_metrics.txt")
with open(metrics_file, "w") as f:
    f.write("\n".join(metrics))

# -------------------------------
# Confusion Matrix for Random Forest
# -------------------------------
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Random Forest')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(plots_dir, "confusion_matrix.png"))
plt.close()

# -------------------------------
# Feature Importance (Random Forest)
# -------------------------------
feature_importances = rf.feature_importances_
features = X.columns
importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances}).sort_values(
    by='Importance', ascending=False)

# Save feature importance plot
plt.figure(figsize=(14, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title('Feature Importance - Random Forest')
plt.savefig(os.path.join(plots_dir, "feature_importance.png"))
plt.close()

# Save feature importance data
importance_df.to_csv(os.path.join(
    reports_dir, "feature_importance.csv"), index=False)

# -------------------------------
# Save Model
# -------------------------------
joblib.dump(rf, os.path.join(reports_dir, "employee_exit_model.pkl"))

# -------------------------------
# Report
# -------------------------------
report = """Employee Exit Prediction Report
===============================
Models Tested:
- Logistic Regression
- Random Forest
Performance Metrics:
""" + "\n".join(metrics) + "\n\nTop Features Influencing Exit:\n" + str(importance_df.head())

report_file = os.path.join(reports_dir, "final_report.md")
with open(report_file, "w") as f:
    f.write(report)

print(
    f"Model built successfully. Plots saved in '{plots_dir}', reports in '{reports_dir}'.")

