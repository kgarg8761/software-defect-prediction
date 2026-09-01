import pandas as pd

# Load KC1 dataset
data = pd.read_csv("kc1.csv")

print("Original dataset shape:")
print(data.shape)

# -----------------------------------
# 1. Remove duplicate rows
# -----------------------------------

data = data.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(data.shape)

# -----------------------------------
# 2. Convert defects to 0 and 1
# -----------------------------------

data["defects"] = data["defects"].astype(int)

print("\nDefect distribution:")
print(data["defects"].value_counts())

# -----------------------------------
# 3. Save cleaned dataset
# -----------------------------------

data.to_csv("kc1_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")