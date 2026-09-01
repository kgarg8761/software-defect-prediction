import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# -----------------------------------------
# 1. Load cleaned dataset
# -----------------------------------------

data = pd.read_csv("kc1_cleaned.csv")


# -----------------------------------------
# 2. Separate features and target
# -----------------------------------------

X = data.drop("defects", axis=1)
y = data["defects"]


# -----------------------------------------
# 3. Split into training and testing data
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------------------
# 4. Create Random Forest model
# -----------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# -----------------------------------------
# 5. Train the model
# -----------------------------------------

model.fit(X_train, y_train)


print("Model trained successfully!")


# -----------------------------------------
# 6. Make predictions
# -----------------------------------------

y_pred = model.predict(X_test)


# -----------------------------------------
# 7. Calculate accuracy
# -----------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")

print("Accuracy:", round(accuracy * 100, 2), "%")


# -----------------------------------------
# 8. Classification report
# -----------------------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred))


# -----------------------------------------
# 9. Confusion matrix
# -----------------------------------------

print("\n========== CONFUSION MATRIX ==========")

cm = confusion_matrix(y_test, y_pred)

print(cm)