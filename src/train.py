# ================================
# Credit Card Default Risk Project
# ================================


# -------------------------------
# Import Libraries
# -------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score,recall_score


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# -------------------------------
# Data Loading
# -------------------------------
# Dataset path
data_path = "data/credit_default_cleaned.csv"


df = pd.read_csv(data_path)


# -------------------------------
# Data Inspection
# -------------------------------
print(df.head())
print(df.info())
print(df.describe())


# -------------------------------
# Data Cleaning
# -------------------------------


# Rename target column
df.rename(columns={'default payment next month': 'target'}, inplace=True)
print("Columns after rename:", df.columns.tolist())

# Remove ID column
if 'ID' in df.columns:
    df.drop('ID', axis=1, inplace=True)

# Check missing values
print("Missing values :",df.isnull().sum())


# -------------------------------
# Feature Engineering & Scaling
# -------------------------------

print("Feature Engineering....")
X = df.drop('target', axis=1)
y = df['target']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


joblib.dump(scaler, "model/scaler.pkl")


# -------------------------------
# Train Test Split
# -------------------------------

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Handle Class Imbalance for XGBoost
# -------------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# -------------------------------
# Model Selection – Models
# -------------------------------

print("Initializing models...")
log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
dt_model = DecisionTreeClassifier(class_weight="balanced", random_state=42)
rf_model = RandomForestClassifier(class_weight="balanced", random_state=42)
svm_model = SVC(probability=True, class_weight="balanced", random_state=42)
xgb_model = XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos_weight, random_state=42)


models = {
"Logistic Regression": log_model,
"Decision Tree": dt_model,
"Random Forest": rf_model,
"SVM": svm_model,
"XGBoost": xgb_model
}


results = {}

# -------------------------------
# Training & Evaluation
# -------------------------------
print("Training models and evaluating performance...")


for name, model in models.items():
    print("==============================")
    print(f"Training Model: {name}")
    print("==============================")


    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]


    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"model/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.show()
    plt.close()


    results[name] = {
    "model": model,
    "accuracy": acc,
    "roc_auc": roc,
    "recall": recall
    }


    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"Recall (Default class): {recall:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    


# -------------------------------
# Model Comparison
# -------------------------------
print("MODEL COMPARISON (ROC-AUC | Recall)")
print("----------------------------------")


for name in results:
    print(f"{name}: ROC-AUC={results[name]['roc_auc']:.4f}, Recall={results[name]['recall']:.4f}")


# -------------------------------
# Best Base Model Selection
# -------------------------------
best_model_name = max(results, key=lambda x: (results[x]['roc_auc'], results[x]['recall']))
best_base_model = results[best_model_name]['model']


print("Best Base Model:", best_model_name)


# -------------------------------
# Hyperparameter Tuning – Random Forest
# -------------------------------
print("Tuning Random Forest...")


rf_param_grid = {
'n_estimators': [100, 200],
'max_depth': [10, 20, None],
'min_samples_split': [2, 5]
}


rf_grid = GridSearchCV(
RandomForestClassifier(class_weight="balanced", random_state=42),
rf_param_grid,
cv=3,
scoring='roc_auc',
n_jobs=-1
)


rf_grid.fit(X_train, y_train)


tuned_rf_model = rf_grid.best_estimator_


rf_y_prob = tuned_rf_model.predict_proba(X_test)[:, 1]
tuned_rf_roc = roc_auc_score(y_test, rf_y_prob)
tuned_rf_recall = recall_score(y_test, tuned_rf_model.predict(X_test))

# -------------------------------
# Confusion Matrix - Tuned Random Forest
# -------------------------------

tuned_rf_pred = tuned_rf_model.predict(X_test)
cm_tuned = confusion_matrix(y_test, tuned_rf_pred)

print("\nConfusion Matrix - Tuned Random Forest:")
print(cm_tuned)

plt.figure(figsize=(5, 4))
sns.heatmap(cm_tuned, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - Tuned Random Forest (Final Model)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("model/confusion_matrix_Tuned_Random_Forest.png")
plt.show()
plt.close()


print("Tuned Random Forest ROC-AUC:", round(tuned_rf_roc, 4))
print("Tuned Random Forest Recall:", round(tuned_rf_recall, 4))


# -------------------------------
# Final Model Selection
# -------------------------------
final_model = tuned_rf_model
print("Final Model Used for Deployment: Tuned Random Forest")


# -------------------------------
# Save Model & Scaler
# -------------------------------
joblib.dump(final_model, "model/best_model.pkl")


print("Model saved successfully in model/best_model.pkl")
print("Training pipeline completed.")






