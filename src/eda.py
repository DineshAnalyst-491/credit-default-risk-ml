# ==============================================
# Exploratory Data Analysis - Credit Default Data
# ==============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# -------------------------------
# Load cleaned dataset
# -------------------------------
data_path = "data/credit_default_cleaned.csv"
df = pd.read_csv(data_path)

# Rename target 
if 'default payment next month' in df.columns:
    df.rename(columns={'default payment next month': 'target'}, inplace=True)

print("Dataset shape:", df.shape)

# Create folder for plots
os.makedirs("eda_plots", exist_ok=True)

sns.set_theme(style="whitegrid")

# -------------------------------
# 1. Target Distribution
# -------------------------------
plt.figure(figsize=(7, 5))
sns.countplot(x='target', data=df)

plt.title("Loan Default Distribution", fontsize=14, fontweight="bold")
plt.suptitle(
    "Business Insight: Dataset is imbalanced - majority customers do not default.\n"
    "Impact: Accuracy alone is misleading; recall & ROC-AUC are more suitable metrics.",
    fontsize=10, y=0.95
)

plt.xlabel("Default Status (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")

plt.tight_layout()
plt.savefig("eda_plots/01_target_distribution.png")
plt.show()
plt.close()

# -------------------------------
# 2. Credit Limit Distribution
# -------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(df['LIMIT_BAL'], bins=50, kde=True)

plt.title("Credit Limit Distribution", fontsize=14, fontweight="bold")
plt.suptitle(
    "Business Insight: Most customers belong to lower credit limit segments.\n"
    "Impact: Lower credit limits often indicate higher financial vulnerability.",
    fontsize=10, y=0.95
)

plt.xlabel("Credit Limit Amount")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("eda_plots/02_credit_limit_distribution.png")
plt.show()
plt.close()

# -------------------------------
# 3. Age vs Default
# -------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='target', y='AGE', data=df)

plt.title("Age vs Loan Default", fontsize=14, fontweight="bold")
plt.suptitle(
    "Business Insight: Certain age groups show higher default variability.\n"
    "Impact: Age can be a moderate risk indicator when combined with payment behavior.",
    fontsize=10, y=0.95
)

plt.xlabel("Default Status (0 = No, 1 = Yes)")
plt.ylabel("Customer Age")

plt.tight_layout()
plt.savefig("eda_plots/03_age_vs_default.png")
plt.show()
plt.close()

# -------------------------------
# 4. Payment Delay vs Default
# -------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='target', y='PAY_1', data=df)

plt.title("Payment Delay (Months - PAY_1) vs Loan Default", fontsize=14, fontweight="bold")
plt.suptitle(
    "Business Insight: Higher payment delays strongly increase default probability.\n"
    "Impact: Repayment history is the most critical predictor of credit risk.",
    fontsize=10, y=0.95
)

plt.xlabel("Default Status (0 = No, 1 = Yes)")
plt.ylabel("Months of Payment Delay")

plt.tight_layout()
plt.savefig("eda_plots/04_pay1_vs_default.png")
plt.show()
plt.close()

# -------------------------------
# 5. Bill Amount vs Default
# -------------------------------
plt.figure(figsize=(8, 5))
sns.boxplot(x='target', y='BILL_AMT1', data=df)

plt.title("Bill Amount (Month 1) vs Loan Default", fontsize=14, fontweight="bold")
plt.suptitle(
    "Business Insight: Defaulters typically have higher outstanding bill amounts.\n"
    "Impact: High utilization combined with low payments signals financial stress.",
    fontsize=10, y=0.95
)

plt.xlabel("Default Status (0 = No, 1 = Yes)")
plt.ylabel("Bill Amount")

plt.tight_layout()
plt.savefig("eda_plots/05_bill_amt_vs_default.png")
plt.show()
plt.close()

# -------------------------------
# 6. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(15, 10))
ax = sns.heatmap(df.corr(), cmap='coolwarm', center=0)

# Main title (top-left)
plt.title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", loc="left")

# Business insight (top-right)
plt.text(
    1.0, 1.05,
    "Business Insight:\n"
    "• Payment delay features (PAY_1 - PAY_6)\n"
    "  show strongest correlation with default.\n"
    "• These variables should receive higher\n"
    "  importance in credit risk models.",
    transform=ax.transAxes,
    fontsize=10,
    ha="right",
    va="bottom"
)

plt.tight_layout()
plt.savefig("eda_plots/06_correlation_heatmap.png")
plt.show()
plt.close()

print("\nEDA completed successfully.")
print("All plots saved in folder: eda_plots/")
