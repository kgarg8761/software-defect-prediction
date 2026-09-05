import pandas as pd
import streamlit as st
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

st.title("Software Defect Prediction System")
st.write("Enter the software metrics below to predict whether the software module is defective.")

input_data = []

for feature in X.columns:
    value = st.number_input(
        "Enter " + feature,
        min_value=0.0,
        value=0.0
    )
    input_data.append(value)

if st.button("Predict Defect"):
    input_df = pd.DataFrame([input_data], columns=X.columns)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error("Prediction: DEFECTIVE SOFTWARE MODULE")
    else:
        st.success("Prediction: NON-DEFECTIVE SOFTWARE MODULE")

    st.write("Defect Probability:", round(probability * 100, 2), "%")