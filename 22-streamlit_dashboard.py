import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(
    page_title="Software Defect Prediction",
    page_icon="🔍",
    layout="wide"
)

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

accuracy = accuracy_score(y_test, model.predict(X_test))

st.title("🔍 Software Defect Prediction System")
st.write("Predict whether a software module is defective using software metrics.")

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Dataset Rows", len(data))

with col2:
    st.metric("Features", len(X.columns))

with col3:
    st.metric("Model Accuracy", f"{accuracy * 100:.2f}%")

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

if st.button("🔍 Predict Defect", use_container_width=True):

    input_df = pd.DataFrame([input_data], columns=X.columns)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    if prediction[0] == 1:
        st.error("⚠️ DEFECTIVE SOFTWARE MODULE")
    else:
        st.success("✅ NON-DEFECTIVE SOFTWARE MODULE")

    st.write("Defect Probability")
    st.progress(float(probability))
    st.write(f"{probability * 100:.2f}%")