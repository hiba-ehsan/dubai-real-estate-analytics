import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from src.data_loader_airbnb import load_airbnb_data
from src.model import train_demand_model, optimize_price, prepare_features

st.title("🤖 ML Price Prediction & Simulation")
st.markdown("Train demand models and simulate booking predictions for different price scenarios")

# Explanation of calculations
with st.expander("**Explanation of calculations.**", expanded=True):
    st.markdown("""
    **Linear Regression Model (OLS)**
    
    We predict booking demand as a linear function of price and other features:
    
    $$y = \\beta_0 + \\beta_1 \\cdot price + \\beta_2 \\cdot x_2 + ... + \\epsilon$$
    
    - **β₁ (slope):** How demand changes per $1 price increase (typically negative)
    - **Intercept (β₀):** Baseline demand when price = 0
    - **R-squared:** How well the model fits the data
    
    **Revenue Calculation:**
    $$Revenue = predicted\\_demand \\times price$$
    """)

# Load data
if 'airbnb_data' in st.session_state:
    df = st.session_state['airbnb_data']
else:
    df = load_airbnb_data()

if df is None or df.empty:
    st.error("No Airbnb data available. Please provide a dataset.")
    st.stop()

# Model settings
st.sidebar.header("Model Settings")
# Only linear OLS model is supported for transparency and hypothesis testing
model_type = 'linear'
st.sidebar.info("Using linear OLS model for prediction and hypothesis testing")

# Listing selection (if listing id exists)
listing_col = None
for c in ['id','listing_id','listingid']:
    if c in df.columns:
        listing_col = c
        break

selected_listing = None
if listing_col:
    listing_choices = df[listing_col].unique().tolist()
    selected_listing = st.sidebar.selectbox("Choose listing (optional)", [None] + listing_choices)

# Train model
st.header("Train Demand Model")
if st.button("Train Model"):
    with st.spinner("Training demand model..."):
        model, scaler, metrics = train_demand_model(df, model_type=model_type)
        if model is None:
            st.error("Model training failed or insufficient data")
        else:
            st.success("Model trained")
            st.write(metrics)
            st.session_state['demand_model'] = (model, scaler, metrics)

# Price simulation
st.header("Price Simulation")
price_input = st.number_input("Test Price ($)", min_value=1.0, value=float(df['our_current_price'].median()))
if st.button("Run Simulation"):
    if 'demand_model' not in st.session_state:
        st.warning("Please train a model first")
    else:
        model, scaler, metrics = st.session_state['demand_model']
        # Use optimize_price to run price sensitivity using trained model type
        result = optimize_price(df, room_type=None, target_occupancy=0.85, model_type=model_type)
        if result:
            st.subheader("Optimization Result (global)")
            st.write({
                'current_price': result['current_price'],
                'recommended_price': result['recommended_price'],
                'predicted_revenue': result['predicted_revenue'],
                'revenue_improvement_pct': result['revenue_improvement_pct']
            })
        else:
            st.error("Failed to compute optimization")

# Single listing prediction (if applicable)
if selected_listing:
    st.header("Single Listing Predictions")
    idx = df[df[listing_col] == selected_listing].index
    if len(idx) > 0:
        sample = df.loc[idx[0]]
        st.write(sample[['our_current_price','competitor_price','occupancy_rate','revenue']])
        # Simple price sensitivity: vary price +/- 30%
        base_price = sample['our_current_price']
        price_range = [base_price * x for x in [0.7, 0.85, 1.0, 1.15, 1.3]]
        sim = []
        for p in price_range:
            temp = df.copy()
            temp.loc[temp[listing_col] == selected_listing, 'our_current_price'] = p
            res = optimize_price(temp, room_type=None, model_type=model_type)
            if res:
                sim.append({'price': p, 'predicted_revenue': res['predicted_revenue']})
        if sim:
            st.table(sim)
        else:
            st.info("Not enough data to simulate single listing")
                        
