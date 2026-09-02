import pandas as pd


data = pd.read_csv("kc1.csv")

print("Original dataset shape:")
print(data.shape)



data = data.drop_duplicates()

print("\nDataset shape after removing duplicates:")
print(data.shape)


data["defects"] = data["defects"].astype(int)

print("\nDefect distribution:")
print(data["defects"].value_counts())



data.to_csv("kc1_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
