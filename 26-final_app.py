import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="🔍",
    layout="wide"
)

model = joblib.load("software_defect_model.pkl")
data = pd.read_csv("06-kc1_cleaned.csv")

X = data.drop("defects", axis=1)

st.title("Software Defect Prediction System")
st.write("Predict whether a software module is defective using software metrics.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Rows", len(data))

with col2:
    st.metric("Software Metrics", len(X.columns))

with col3:
    st.metric("Model", "Logistic Regression")

st.divider()

st.subheader("Enter Software Metrics")

input_data = []

columns = st.columns(3)

for i, feature in enumerate(X.columns):
    with columns[i % 3]:
        value = st.number_input(
            feature,
            min_value=0.0,
            value=0.0,
            key=feature
        )
        input_data.append(value)

st.divider()

if st.button("Predict Defect", use_container_width=True):

    input_df = pd.DataFrame([input_data], columns=X.columns)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error("DEFECTIVE SOFTWARE MODULE")
    else:
        st.success("NON-DEFECTIVE SOFTWARE MODULE")

    st.write("Defect Probability:", f"{probability * 100:.2f}%")
    st.progress(float(probability))