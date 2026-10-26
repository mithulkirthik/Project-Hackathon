# Telco Customer Churn Predictor

A machine learning project that estimates whether a telecommunications customer is likely to leave a company.

## Features
- Customer churn data analysis and preprocessing
- Comparison of 15 machine learning model configurations
- Evaluation using Accuracy, Precision, Recall, F1-score, and ROC-AUC
- Interactive prediction interface built with Streamlit

## Technologies
- Python 3.12
- Pandas
- Scikit-learn
- Streamlit
- Joblib

## Project Files
- `app.py` — Streamlit application
- `churn_model.joblib` — Trained machine learning pipeline
- `feature_schema.json` — Input feature definitions
- `model_comparison.csv` — Model performance comparison
- `requirements.txt` — Required Python packages

## Run Locally

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the application:

```bash
python -m streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## Disclaimer
Predictions are estimates, not guarantees of customer behavior.
