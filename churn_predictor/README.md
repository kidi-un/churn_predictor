# Customer Churn Prediction App

**ML-powered churn prediction** built by Vasanth A (AI & Data Science, 2026 batch)

## Live Demo
[Open the app](https://vasanth-churn.streamlit.app)

## What it does
- Predicts churn probability for any customer in real time (0–100%)
- Shows risk level: High / Medium / Low with retention recommendations
- Feature importance from Random Forest — tenure and charges are top drivers
- ROC curve, confusion matrix, classification report
- Bulk CSV upload — predict churn for hundreds of customers at once

## Tech stack
| Layer | Tool |
|---|---|
| ML Model | Random Forest (200 trees, scikit-learn) |
| Explainability | Feature importance + SHAP-style analysis |
| Frontend | Streamlit |
| Charts | Plotly Express |
| Data | Telco-style generated dataset (7,043 customers) |

## Model performance
- Accuracy: ~88%
- ROC-AUC: ~92%
- Dataset: 7,043 customers, 26.5% churn rate

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Cloud
1. Push to GitHub repo `customer-churn-predictor`
2. share.streamlit.io → New app → select repo → app.py → Deploy

## Resume line
> "Built a Customer Churn Prediction app — Random Forest (88% accuracy, 0.92 AUC), real-time prediction with retention recommendations, bulk CSV scoring — deployed live at [url]"
