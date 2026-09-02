import pandas as pd

data = pd.read_csv("kc1.csv")

print("DATASET INFORMATION")
print(data.info())

print("\nDATASET SHAPE")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])

print("\nMISSING VALUES")
print(data.isnull().sum())

print("\nDUPLICATE ROWS")
print("Number of duplicates:", data.duplicated().sum())

print("\nDEFECT DISTRIBUTION")
print(data["defects"].value_counts())

print("\nSTATISTICS")
print(data.describe())