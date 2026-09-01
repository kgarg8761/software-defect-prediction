import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("kc1_cleaned.csv")

X = data.drop("defects", axis=1)
y = data["defects"]

print("Features (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())

print("\nShape of X:")
print(X.shape)

print("\nShape of y:")
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:")
print(X_train.shape)

print("\nTesting data:")
print(X_test.shape)
