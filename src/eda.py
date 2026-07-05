"""
Stage 1: Exploratory Data Analysis — Customer Churn Dataset
Author: Akash

Goal: understand the data before touching any model.
Specifically checking for:
  - missing values (and WHY they're missing, not just how many)
  - class imbalance in the target (Churn)
  - data types that look wrong (e.g. numeric columns stored as strings)
  - obvious signal in features vs churn
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

# ---- 1. Load data ----
df = pd.read_csv("data/telco_churn.csv")

print("=" * 60)
print("SHAPE:", df.shape)
print("=" * 60)
print(df.dtypes)

# ---- 2. Check for the classic Telco churn "hidden" data quality bug ----
# TotalCharges is often stored as object/string with blank spaces for
# customers with tenure == 0 (they haven't been charged yet). This is a
# well known gotcha in this dataset — if you don't catch it, pd.to_numeric
# will silently produce NaNs that mess up your model later.
print("\n--- Checking TotalCharges column ---")
print("Dtype:", df["TotalCharges"].dtype)
blank_mask = df["TotalCharges"].str.strip() == ""
print(f"Rows with blank TotalCharges: {blank_mask.sum()}")
print(df.loc[blank_mask, ["customerID", "tenure", "TotalCharges"]])

# Fix: convert to numeric, coercing blanks to NaN, then decide what to do
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(f"\nNaNs after conversion: {df['TotalCharges'].isna().sum()}")
print("These are all tenure==0 customers (just signed up) -> fill with 0")
df["TotalCharges"] = df["TotalCharges"].fillna(0)

# ---- 3. Target class balance ----
print("\n--- Target class balance (Churn) ---")
churn_counts = df["Churn"].value_counts(normalize=True) * 100
print(churn_counts)
print(
    "\nThis is an IMBALANCED dataset (~73% No / ~27% Yes). "
    "Accuracy alone will be misleading — a model that always predicts "
    "'No' would already score ~73% accuracy while being useless. "
    "This is why we'll use precision/recall/F1/AUC later, not accuracy."
)

# ---- 4. Missing values overall ----
print("\n--- Missing values per column ---")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values after TotalCharges fix.")

# ---- 5. Quick look at churn rate by a few key categorical features ----
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0, 0])
axes[0, 0].set_title("Churn by Contract Type")

sns.countplot(data=df, x="InternetService", hue="Churn", ax=axes[0, 1])
axes[0, 1].set_title("Churn by Internet Service")

sns.histplot(data=df, x="tenure", hue="Churn", bins=30, kde=True, ax=axes[1, 0])
axes[1, 0].set_title("Churn by Tenure")

sns.histplot(data=df, x="MonthlyCharges", hue="Churn", bins=30, kde=True, ax=axes[1, 1])
axes[1, 1].set_title("Churn by Monthly Charges")

plt.tight_layout()
plt.savefig("notebooks/eda_overview.png", dpi=120)
print("\nSaved EDA plots to notebooks/eda_overview.png")

# ---- 6. Save cleaned version for the next stage ----
df.to_csv("data/telco_churn_cleaned.csv", index=False)
print("\nSaved cleaned dataset to data/telco_churn_cleaned.csv")
