import pandas as pd

data = pd.read_csv("kc1.csv")

print("First 5 rows of KC1 dataset:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns.tolist())
