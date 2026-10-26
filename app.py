
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(BASE_DIR / "churn_model.joblib")

def load_schema():
    with open(BASE_DIR / "feature_schema.json", "r") as f:
        return json.load(f)

st.title("📊 Telco Customer Churn Predictor")
st.write(
    "Enter a customer's service details to estimate "
    "their likelihood of leaving the telecom company."
)

try:
    model = load_model()
    schema = load_schema()
except Exception as e:
    st.error(f"Could not load model or schema: {e}")
    st.stop()

with st.form("customer_form"):
    st.subheader("Customer details")
    inputs = {}

    for field in schema:
        name = field["name"]

        if field["type"] == "category":
            options = field["options"]

            if options:
                inputs[name] = st.selectbox(
                    name, options, key=f"field_{name}"
                )
            else:
                inputs[name] = st.text_input(name)

        else:
            minimum = field["min"]
            maximum = field["max"]
            default = field["default"]

            inputs[name] = st.number_input(
                name,
                min_value=minimum,
                max_value=maximum,
                value=min(max(default, minimum), maximum),
                key=f"field_{name}"
            )

    submitted = st.form_submit_button("Predict churn")

if submitted:
    customer = pd.DataFrame([inputs], columns=[
        field["name"] for field in schema
    ])

    try:
        prediction = int(model.predict(customer)[0])
        probability = float(model.predict_proba(customer)[0][1])

        st.subheader("Prediction result")

        if prediction == 1:
            st.error("Prediction: Customer is at risk of churn.")
        else:
            st.success("Prediction: Customer is predicted to stay.")

        st.metric(
            "Estimated churn probability",
            f"{probability:.1%}"
        )
        st.progress(probability)

        st.caption(
            "This is a model estimate, not a guarantee. "
            "The prediction depends on the quality and representativeness "
            "of the training data."
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
