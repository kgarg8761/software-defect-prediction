import pandas as pd

data = pd.read_csv("kc1.csv")

print("========== DATASET INFORMATION ==========")
print(data.info())

print("\n========== DATASET SHAPE ==========")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print("Number of duplicates:", data.duplicated().sum())

print("\n========== DEFECT DISTRIBUTION ==========")
print(data["defects"].value_counts())

print("\n========== STATISTICS ==========")
print(data.describe())
