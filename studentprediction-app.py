import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Student Pass/Fail Predictor", layout="centered")

st.title("🎓 Student Pass/Fail Prediction System")

# Load your uploaded cleaned dataset
df = pd.read_csv("Cleaned_Synthetic_Student_Data.csv")

st.subheader("Dataset Preview")
st.write(df.head())

# -----------------------------
# Identify Target Column
# -----------------------------
target_col = None
for col in df.columns:
    if "result" in col.lower() or "pass" in col.lower():
        target_col = col
        break

if target_col is None:
    st.error("Target column not found. Please ensure Pass/Fail column exists.")
    st.stop()

# Encode Target if categorical
if df[target_col].dtype == "object":
    le = LabelEncoder()
    df[target_col] = le.fit_transform(df[target_col])

# Features & Target
X = df.drop(target_col, axis=1)
y = df[target_col]

# Convert categorical features
X = pd.get_dummies(X)

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

st.success("Model Trained Successfully ✅")

# -----------------------------
# User Input Section
# -----------------------------
st.subheader("Enter Student Details")

user_input = {}

for col in X.columns:
    user_input[col] = st.number_input(f"{col}", value=0.0)

if st.button("Predict Result"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success("Prediction: PASS ✅")
    else:
        st.error("Prediction: FAIL ❌")

    st.write(f"Confidence Level: {probability:.2f}")
