import streamlit as st
from src.data_loader_airbnb import load_airbnb_data
from src.visualization import (
    plot_price_comparison, plot_revenue_trend, plot_demand_heatmap,
    plot_3d_scatter, plot_correlation_heatmap, plot_boxplot_by_day, plot_price_elasticity
)
from src.multi_library_viz import (
    plot_altair_revenue_by_room, plot_altair_price_scatter, plot_seaborn_regression
)

st.title("🎨 Airbnb Visualization Gallery")
st.markdown("Explore Airbnb-specific visualizations using multiple libraries")

# Load data
if 'airbnb_data' in st.session_state:
    df = st.session_state['airbnb_data']
else:
    df = load_airbnb_data()

if df is None or df.empty:
    st.error("No data available. Please upload or provide Airbnb CSV.")
    st.stop()

# Filters
st.sidebar.header("Filters")
if 'room_type' in df.columns:
    selected_rooms = st.sidebar.multiselect("Listing Types", options=df['room_type'].unique().tolist(), default=None)
    if selected_rooms:
        df = df[df['room_type'].isin(selected_rooms)]

st.header("Core Visualizations")

try:
    fig = plot_price_comparison(df)
    if fig:
        st.subheader("Price Comparison")
        st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info("Price comparison not available")

try:
    fig = plot_revenue_trend(df)
    if fig:
        st.subheader("Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info("Revenue trend not available")

try:
    fig = plot_demand_heatmap(df)
    if fig:
        st.subheader("Occupancy Heatmap")
        st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info("Occupancy heatmap not available")

# Additional charts
st.subheader("Additional Charts")
try:
    fig = plot_price_elasticity(df)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
except Exception:
    st.info("Price elasticity not available")

try:
    img = plot_boxplot_by_day(df, column='our_current_price')
    if img:
        st.image(f"data:image/png;base64,{img}", use_container_width=True)
except Exception:
    st.info("Boxplot not available")

with st.expander("📋 Sample Data"):
    st.dataframe(df.head(10), use_container_width=True)
