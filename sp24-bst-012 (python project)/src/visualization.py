import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64


# ========== Plotly Visualizations ==========

def plot_price_comparison(df):
    """Compare our prices vs competitor prices over time using line chart."""
    if df.empty or 'date' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    fig = go.Figure()
    
    if 'our_current_price' in df_sorted.columns:
        fig.add_trace(go.Scatter(
            x=df_sorted['date'],
            y=df_sorted['our_current_price'],
            mode='lines+markers',
            name='Our Price',
            line=dict(color='#E74C3C', width=3),
            marker=dict(size=6)
        ))
    
    if 'competitor_price' in df_sorted.columns:
        fig.add_trace(go.Scatter(
            x=df_sorted['date'],
            y=df_sorted['competitor_price'],
            mode='lines+markers',
            name='Competitor Price',
            line=dict(color='#95A5A6', width=3),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Our Price vs Competitor Price Over Time",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=500,
        hovermode='x unified'
    )
    return fig


def plot_revenue_trend(df):
    """Plot daily revenue trend with confidence intervals."""
    if df.empty or 'date' not in df.columns or 'revenue' not in df.columns:
        return None
    
    fig = px.line(df, x="date", y="revenue", title="Daily Revenue Trend", 
                  markers=True, color_discrete_sequence=["#27AE60"])
    fig.update_layout(height=500, xaxis_title="Date", yaxis_title="Revenue ($)")
    return fig


def plot_demand_heatmap(df):
    """Heatmap showing occupancy by day of week and room type.

    The function is resilient if `day_of_week` is missing: it will create it from
    a `date` column if available. If required columns are missing or there is no
    usable data, it returns None so callers can handle absence gracefully.
    """
    if df is None or df.empty:
        return None

    # Need occupancy rate to compute heatmap
    if 'occupancy_rate' not in df.columns:
        return None

    # Work on a copy to avoid mutating caller's dataframe
    df = df.copy()

    # Create day_of_week from date if missing
    if 'day_of_week' not in df.columns:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['day_of_week'] = df['date'].dt.day_name()
        else:
            # Can't create day_of_week — abort
            return None

    # Ensure room_type exists
    if 'room_type' not in df.columns:
        df['room_type'] = 'Listing'

    # Clean occupancy_rate
    df['occupancy_rate'] = pd.to_numeric(df['occupancy_rate'], errors='coerce')

    # Drop rows missing essential values
    clean = df.dropna(subset=['day_of_week', 'room_type', 'occupancy_rate'])
    if clean.empty:
        return None

    try:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        pivot = clean.pivot_table(values="occupancy_rate", index="day_of_week", columns="room_type", aggfunc="mean")
        pivot = pivot.reindex([d for d in day_order if d in pivot.index])

        if pivot.empty:
            return None

        fig = px.imshow(pivot,
                        text_auto=".2%",
                        aspect="auto",
                        title="Occupancy Heatmap by Day & Room Type",
                        color_continuous_scale="Blues")
        fig.update_xaxes(side="top")
        fig.update_layout(height=400)
        return fig
    except Exception:
        return None


def plot_time_series_with_forecast(df, column='revenue', periods=7):
    """Plot time series with simple moving average forecast."""
    if 'date' not in df.columns or column not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=df_sorted['date'],
        y=df_sorted[column],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#3498DB', width=2)
    ))
    
    # Moving average
    window = min(7, len(df_sorted))
    df_sorted['ma'] = df_sorted[column].rolling(window=window, center=True).mean()
    fig.add_trace(go.Scatter(
        x=df_sorted['date'],
        y=df_sorted['ma'],
        mode='lines',
        name=f'{window}-Day Moving Average',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    # Simple forecast (extend trend)
    if len(df_sorted) > 1:
        last_date = df_sorted['date'].max()
        last_value = df_sorted[column].iloc[-1]
        trend = (df_sorted[column].iloc[-1] - df_sorted[column].iloc[-window]) / window if len(df_sorted) >= window else 0
        
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq='D')
        forecast_values = [last_value + trend * (i+1) for i in range(periods)]
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#95A5A6', width=2, dash='dot')
        ))
    
    fig.update_layout(
        title=f"{column.title()} Time Series with Forecast",
        xaxis_title="Date",
        yaxis_title=column.title(),
        height=500,
        hovermode='x unified'
    )
    return fig


def plot_3d_scatter(df):
    """3D scatter plot: Price vs Occupancy vs Revenue."""
    if not all(col in df.columns for col in ['our_current_price', 'occupancy_rate', 'revenue']):
        return None
    
    fig = go.Figure(data=go.Scatter3d(
        x=df['our_current_price'],
        y=df['occupancy_rate'],
        z=df['revenue'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['revenue'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Revenue")
        ),
        text=df.get('room_type', ''),
        hovertemplate='Price: $%{x}<br>Occupancy: %{y:.1%}<br>Revenue: $%{z}<extra></extra>'
    ))
    
    fig.update_layout(
        title="3D Analysis: Price vs Occupancy vs Revenue",
        scene=dict(
            xaxis_title="Price ($)",
            yaxis_title="Occupancy Rate",
            zaxis_title="Revenue ($)"
        ),
        height=600
    )
    return fig


def plot_correlation_heatmap(df):
    """Plot correlation heatmap using Seaborn (converted to base64 for Streamlit)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return None
    
    corr = df[numeric_cols].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
    plt.title('Correlation Heatmap of Hotel Metrics', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Convert to base64 for Streamlit
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_str


def plot_boxplot_by_day(df, column='our_current_price'):
    """Boxplot showing distribution by day of week using Matplotlib.

    This function will create `day_of_week` from `date` if missing, and return
    None if required data is not available.
    """
    if df is None or df.empty:
        return None

    # Ensure requested column exists
    if column not in df.columns:
        return None

    df = df.copy()

    # Create day_of_week if missing
    if 'day_of_week' not in df.columns:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['day_of_week'] = df['date'].dt.day_name()
        else:
            return None

    # Drop missing values
    clean = df.dropna(subset=['day_of_week', column])
    if clean.empty:
        return None

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    data_by_day = [clean[clean['day_of_week'] == day][column].values 
                   for day in day_order if day in clean['day_of_week'].values]
    labels = [day for day in day_order if day in clean['day_of_week'].values]

    if len(data_by_day) == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(data_by_day, labels=labels)
    ax.set_title(f'{column.replace("_", " ").title()} Distribution by Day of Week')
    ax.set_ylabel(column.replace("_", " ").title())
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return img_str


def plot_price_elasticity(df):
    """Scatter plot showing price vs bookings (demand elasticity)."""
    if not all(col in df.columns for col in ['our_current_price', 'bookings']):
        return None
    
    fig = px.scatter(df, x='our_current_price', y='bookings',
                     color='room_type', size='revenue',
                     title='Price Elasticity: Price vs Bookings',
                     labels={'our_current_price': 'Price ($)', 'bookings': 'Bookings'},
                     trendline="ols",
                     hover_data=['date'])
    fig.update_layout(height=500)
    return fig


def plot_weekly_aggregates(df):
    """Plot weekly aggregated metrics."""
    if 'date' not in df.columns:
        return None
    
    df_weekly = df.set_index('date').resample('W').agg({
        'revenue': 'sum',
        'bookings': 'sum',
        'our_current_price': 'mean',
        'occupancy_rate': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_weekly['date'], y=df_weekly['revenue'],
                            mode='lines+markers', name='Weekly Revenue',
                            yaxis='y', line=dict(color='#27AE60')))
    
    fig.add_trace(go.Scatter(x=df_weekly['date'], y=df_weekly['our_current_price'],
                            mode='lines+markers', name='Avg Price',
                            yaxis='y2', line=dict(color='#E74C3C')))
    
    fig.update_layout(
        title='Weekly Aggregated Metrics',
        xaxis_title='Week',
        yaxis=dict(title='Revenue ($)', side='left'),
        yaxis2=dict(title='Price ($)', side='right', overlaying='y'),
        height=500,
        hovermode='x unified'
    )
    return fig


def plot_time_series_decomposition(df, column='revenue', model='additive'):
    """Plot time series decomposition (trend, seasonal, residual)."""
    from src.utils import decompose_time_series
    
    decomposition = decompose_time_series(df, column, model)
    if decomposition is None:
        return None
    
    fig = go.Figure()
    
    # Original
    fig.add_trace(go.Scatter(
        x=decomposition.observed.index,
        y=decomposition.observed.values,
        mode='lines',
        name='Original',
        line=dict(color='#3498DB', width=2)
    ))
    
    # Trend
    fig.add_trace(go.Scatter(
        x=decomposition.trend.index,
        y=decomposition.trend.values,
        mode='lines',
        name='Trend',
        line=dict(color='#E74C3C', width=2, dash='dash')
    ))
    
    # Seasonal
    fig.add_trace(go.Scatter(
        x=decomposition.seasonal.index,
        y=decomposition.seasonal.values,
        mode='lines',
        name='Seasonal',
        line=dict(color='#27AE60', width=2, dash='dot')
    ))
    
    # Residual
    fig.add_trace(go.Scatter(
        x=decomposition.resid.index,
        y=decomposition.resid.values,
        mode='lines',
        name='Residual',
        line=dict(color='#95A5A6', width=1)
    ))
    
    fig.update_layout(
        title=f'Time Series Decomposition - {column.title()} ({model})',
        xaxis_title='Date',
        yaxis_title=column.title(),
        height=600,
        hovermode='x unified'
    )
    return fig


def plot_density_by_room(df, column='our_current_price'):
    """Density plot (KDE) showing distribution by room type."""
    if column not in df.columns or 'room_type' not in df.columns:
        return None
    
    fig = go.Figure()
    
    for room_type in df['room_type'].unique():
        room_data = df[df['room_type'] == room_type][column].dropna()
        
        if len(room_data) > 0:
            # Create KDE using scipy if available, otherwise use empirical CDF
            try:
                from scipy import stats
                kde = stats.gaussian_kde(room_data)
                x_range = np.linspace(room_data.min(), room_data.max(), 100)
                y_range = kde(x_range)
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    name=room_type,
                    fill='tozeroy',
                    opacity=0.6,
                    line=dict(width=2)
                ))
            except ImportError:
                # Fallback: use empirical distribution
                sorted_data = np.sort(room_data)
                y_vals = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
                fig.add_trace(go.Scatter(
                    x=sorted_data,
                    y=y_vals,
                    mode='lines',
                    name=room_type,
                    fill='tozeroy',
                    opacity=0.6,
                    line=dict(width=2)
                ))
    
    fig.update_layout(
        title=f'Density Distribution of {column.replace("_", " ").title()} by Room Type',
        xaxis_title=column.replace("_", " ").title(),
        yaxis_title='Density',
        height=500,
        hovermode='x unified'
    )
    return fig


def plot_violin_by_day(df, column='our_current_price'):
    """Violin plot showing distribution by day of week using Matplotlib.

    This function will derive `day_of_week` from `date` if needed and return None
    when insufficient data exists.
    """
    if df is None or df.empty:
        return None

    if column not in df.columns:
        return None

    df = df.copy()

    if 'day_of_week' not in df.columns:
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['day_of_week'] = df['date'].dt.day_name()
        else:
            return None

    clean = df.dropna(subset=['day_of_week', column])
    if clean.empty:
        return None

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    data_by_day = [clean[clean['day_of_week'] == day][column].values 
                   for day in day_order if day in clean['day_of_week'].values]
    labels = [day for day in day_order if day in clean['day_of_week'].values]

    if len(data_by_day) == 0:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create violin plot
    parts = ax.violinplot(data_by_day, positions=range(len(labels)), showmeans=True, showmedians=True)

    # Customize colors
    for pc in parts['bodies']:
        pc.set_facecolor('#3498DB')
        pc.set_alpha(0.7)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel(column.replace("_", " ").title())
    ax.set_title(f'{column.replace("_", " ").title()} Distribution by Day of Week (Violin Plot)')
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)

    return img_str


def plot_advanced_forecast_comparison(df, column='revenue', periods=7):
    """Compare multiple forecasting methods."""
    from src.utils import forecast_simple, forecast_exponential_smoothing
    
    if column not in df.columns or 'date' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    fig = go.Figure()
    
    # Actual data
    fig.add_trace(go.Scatter(
        x=df_sorted['date'],
        y=df_sorted[column],
        mode='lines+markers',
        name='Actual',
        line=dict(color='#3498DB', width=3)
    ))
    
    # Simple linear forecast
    simple_forecast = forecast_simple(df, column, periods)
    if simple_forecast is not None:
        fig.add_trace(go.Scatter(
            x=simple_forecast['date'],
            y=simple_forecast[column],
            mode='lines+markers',
            name='Linear Forecast',
            line=dict(color='#E74C3C', width=2, dash='dash')
        ))
    
    # Exponential smoothing forecast
    exp_forecast = forecast_exponential_smoothing(df, column, periods)
    if exp_forecast is not None:
        fig.add_trace(go.Scatter(
            x=exp_forecast['date'],
            y=exp_forecast[column],
            mode='lines+markers',
            name='Exponential Smoothing',
            line=dict(color='#27AE60', width=2, dash='dot')
        ))
    
    fig.update_layout(
        title=f'Forecast Comparison - {column.title()}',
        xaxis_title='Date',
        yaxis_title=column.title(),
        height=500,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    return fig


def plot_candlestick_price(df):
    """Candlestick chart for price movements (daily high/low/open/close)."""
    if 'date' not in df.columns or 'our_current_price' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    
    # Create OHLC data (simplified - using same price for open/close/high/low)
    # In real scenario, you'd have separate columns
    fig = go.Figure(data=go.Candlestick(
        x=df_sorted['date'],
        open=df_sorted['our_current_price'],
        high=df_sorted['our_current_price'] * 1.02,  # Simulated
        low=df_sorted['our_current_price'] * 0.98,   # Simulated
        close=df_sorted['our_current_price'],
        name='Price'
    ))
    
    fig.update_layout(
        title='Price Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price ($)',
        height=500,
        xaxis_rangeslider_visible=False
    )
    return fig