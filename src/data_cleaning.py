import pandas as pd
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("data/credit_default.csv")

print("Before cleaning:")
print(df.isnull().sum())

# Target column name
target_col = 'default payment next month'

# Separate features and target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Impute missing values
imputer = SimpleImputer(strategy="median")
X_cleaned = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# Combine back
df_cleaned = pd.concat([X_cleaned, y.reset_index(drop=True)], axis=1)

print("\nAfter cleaning:")
print(df_cleaned.isnull().sum())

# Save cleaned dataset
df_cleaned.to_csv("data/credit_default_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully.")
