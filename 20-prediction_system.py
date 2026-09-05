import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("06-kc1_cleaned.csv")

X = data.drop("defects", axis=1)
y = data["defects"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=5000))
])

model.fit(X_train, y_train)

print("Software Defect Prediction System")
print("Enter the following software metrics:\n")

input_data = []

for feature in X.columns:
    while True:
        try:
            value = float(input("Enter " + feature + ": "))
            input_data.append(value)
            break
        except ValueError:
            print("Please enter a numeric value.")

input_df = pd.DataFrame([input_data], columns=X.columns)

prediction = model.predict(input_df)

probability = model.predict_proba(input_df)[0][1]

if prediction[0] == 1:
    print("\nPrediction: DEFECTIVE SOFTWARE MODULE")
else:
    print("\nPrediction: NON-DEFECTIVE SOFTWARE MODULE")

print("Defect Probability:", round(probability * 100, 2), "%")