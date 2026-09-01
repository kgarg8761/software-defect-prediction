import pandas as pd

# Load KC1 dataset
data = pd.read_csv("kc1.csv")

# Display first 5 rows
print("First 5 rows of KC1 dataset:")
print(data.head())

# Display dataset size
print("\nDataset shape:")
print(data.shape)

# Display column names
print("\nColumn names:")
print(data.columns.tolist())