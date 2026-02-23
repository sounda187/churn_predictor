import streamlit as st
import polars as pl
import joblib
import numpy as np

st.title("🔮 Churn Predictor Pro")
st.markdown("**Built with: Polars + RandomForest + Streamlit**")

# Load model
@st.cache_resource
def load_model():
    return joblib.load('data/churn_model.pkl')

model = load_model()

# Inputs
st.sidebar.header("Customer Details")
age = st.sidebar.slider("Age", 18, 80, 35)
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly = st.sidebar.slider("Monthly Charges", 18.0, 120.0, 70.0)

if st.button("🚀 PREDICT", type="primary"):
    # Create input
    features = [tenure, monthly, monthly*tenure, age, tenure/12, monthly*tenure, monthly]
    prob = model.predict_proba([features])[0][1]
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Churn Risk", f"{prob:.1%}")
    with col2:
        st.error("🚨 HIGH RISK!" if prob > 0.5 else "✅ LOW RISK")
