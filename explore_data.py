import pandas as pd

# Load KC1 dataset
data = pd.read_csv("kc1.csv")

# 1. Basic information
print("========== DATASET INFORMATION ==========")
print(data.info())

# 2. Number of rows and columns
print("\n========== DATASET SHAPE ==========")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

# 3. Check missing values
print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# 4. Check duplicate rows
print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicates:", data.duplicated().sum())

# 5. Check target column
print("\n========== DEFECT DISTRIBUTION ==========")
print(data["defects"].value_counts())

# 6. Display statistical information
print("\n========== STATISTICS ==========")
print(data.describe())