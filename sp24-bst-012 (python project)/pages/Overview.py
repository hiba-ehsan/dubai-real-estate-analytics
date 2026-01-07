import streamlit as st
from src.data_loader_airbnb import load_airbnb_data
from src.visualization import (
    plot_revenue_trend, plot_demand_heatmap, plot_time_series_with_forecast,
    plot_weekly_aggregates, plot_correlation_heatmap, plot_time_series_decomposition
)
from src.utils import calculate_seasonality_strength
from src.utils import calculate_revenue_metrics, resample_time_series

st.title("📊 Airbnb Overview")

# Get data from session state or load
if 'airbnb_data' in st.session_state:
    df = st.session_state['airbnb_data']
else:
    use_db = st.sidebar.checkbox("Use Database (if available)", value=False)
    df = load_airbnb_data()

if df is None or df.empty:
    st.error("No Airbnb data available. Please upload or provide a CSV.")
    st.stop()

# Data info
with st.expander("ℹ️ Data Information", expanded=False):
    st.info(f"""
    **Data loaded:** {len(df):,} records
    **Date range:** {df['date'].min().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'} to {df['date'].max().strftime('%Y-%m-%d') if 'date' in df.columns else 'N/A'}
    **Listing types:** {', '.join(df['room_type'].unique().tolist()) if 'room_type' in df.columns else 'N/A'}
    **Cities / Neighbourhoods:** {df['city'].nunique() if 'city' in df.columns else 'N/A'}
    """)

# Filters
st.sidebar.header("Filters")
date_range = None
if 'date' in df.columns:
    min_date = df['date'].min().date()
    max_date = df['date'].max().date()
    date_range = st.sidebar.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if len(date_range) == 2:
        from datetime import datetime
        date_range = (datetime.combine(date_range[0], datetime.min.time()),
                     datetime.combine(date_range[1], datetime.max.time()))

room_types = None
if 'room_type' in df.columns:
    room_types = st.sidebar.multiselect(
        "Listing Types",
        options=df['room_type'].unique().tolist(),
        default=df['room_type'].unique().tolist()
    )

# Apply filters
from src.data_loader import filter_data
filtered_df = filter_data(df, date_range=date_range, room_types=room_types)

# Key Metrics
st.header("Key Performance Indicators")
metrics = calculate_revenue_metrics(filtered_df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Revenue", f"${metrics.get('total_revenue', 0):,.0f}")
col2.metric("Avg Occupancy", f"{metrics.get('avg_occupancy', 0):.1%}")
col3.metric("Avg Price", f"${metrics.get('avg_price', 0):.0f}")
col4.metric("Total Bookings", f"{metrics.get('total_bookings', 0):,.0f}")

# Time Series Visualizations
st.header("Time Series Analysis")

tab1, tab2, tab3 = st.tabs(["Price / Revenue Trend", "Decomposition", "Weekly Aggregates"])

with tab1:
    revenue_fig = plot_revenue_trend(filtered_df)
    if revenue_fig:
        st.plotly_chart(revenue_fig, use_container_width=True)
    else:
        st.info("Preparing revenue trend visualization...")

with tab2:
    st.subheader("Time Series Decomposition")
    decomposition_model = st.selectbox("Decomposition Model", ["additive", "multiplicative"], key="airbnb_decomp")
    fig_decomp = plot_time_series_decomposition(filtered_df, 'revenue', decomposition_model)
    if fig_decomp:
        st.plotly_chart(fig_decomp, use_container_width=True)
        seasonality_strength = calculate_seasonality_strength(filtered_df, 'revenue')
        if seasonality_strength is not None:
            st.metric("Seasonality Strength", f"{seasonality_strength:.2%}")
    else:
        st.info("Insufficient data for decomposition")

with tab3:
    weekly_fig = plot_weekly_aggregates(filtered_df)
    if weekly_fig:
        st.plotly_chart(weekly_fig, use_container_width=True)

# Occupancy Heatmap
st.header("Occupancy Analysis")
heatmap_fig = plot_demand_heatmap(filtered_df)
if heatmap_fig:
    st.plotly_chart(heatmap_fig, use_container_width=True)

# Data preview
with st.expander("📋 Preview Data", expanded=False):
    st.dataframe(filtered_df.head(10), use_container_width=True)
