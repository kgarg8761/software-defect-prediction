import pandas as pd

data = pd.read_csv("06-kc1_cleaned.csv")

print("SOFTWARE DEFECT PREDICTION PROJECT")
print("=" * 40)

print("\nDataset Information")
print("Rows:", data.shape[0])
print("Columns:", data.shape[1])
print("Features:", data.shape[1] - 1)

print("\nDefect Distribution")
print(data["defects"].value_counts())

print("\nProject Models")
print("1. Random Forest")
print("2. Logistic Regression")
print("3. Decision Tree")

print("\nFinal Prediction System")
print("Input: Software metrics")
print("Output: Defective or Non-Defective")

print("\nProject completed successfully!")