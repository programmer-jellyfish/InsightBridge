import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

from google import genai

# ---------------- Gemini Setup ----------------
client = genai.Client()

st.set_page_config(
    page_title="InsightBridge ML",
    layout="wide"
)

st.title("InsightBridge ML")
st.caption("Auditing machine learning trustworthiness in structured datasets")

# ---------------- Core Logic ----------------
def run_ml_analysis(df):
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.shape[1] < 2:
        raise ValueError("Dataset must contain at least two numeric columns.")

    target = numeric_df.columns[-1]
    X = numeric_df.iloc[:, :-1].fillna(0)
    y = numeric_df[target].fillna(0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if y.nunique() <= 10:
        model = LogisticRegression(max_iter=2000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric = accuracy_score(y_test, preds)
        task = "Classification"
        metric_name = "Accuracy"
        coeffs = model.coef_[0]
    else:
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metric = mean_squared_error(y_test, preds)
        task = "Regression"
        metric_name = "Mean Squared Error"
        coeffs = model.coef_

    feature_importance = dict(
        zip(X.columns, np.abs(coeffs))
    )

    return {
        "task": task,
        "target": target,
        "metric_name": metric_name,
        "metric_value": round(float(metric), 4),
        "feature_importance": feature_importance
    }


def interpret_with_gemini(ml_results):
    prompt = f"""
You are an ML audit system.

Analyze the following ML results and report:
1. What the model has learned
2. Whether the metric is trustworthy
3. Key risks or limitations
4. Suitability for real-world usage

Use professional, concise language.

ML Results:
{ml_results}
"""

    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=prompt
    )

    return response.text


# ---------------- UI ----------------
uploaded_file = st.file_uploader("Upload a CSV dataset", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded")

    if st.button("Run ML Audit"):
        with st.spinner("Running analysis..."):
            ml_results = run_ml_analysis(df)
            interpretation = interpret_with_gemini(ml_results)

        st.subheader("Machine Learning Summary")
        st.write(ml_results)

        st.subheader("AI Interpretation")
        st.write(interpretation)
