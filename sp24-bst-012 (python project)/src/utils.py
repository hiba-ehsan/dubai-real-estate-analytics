import pandas as pd
import numpy as np
from datetime import datetime, timedelta
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False


def calculate_revenue_metrics(df):
    """Calculate key revenue metrics."""
    if df.empty:
        return {}
    
    # Try to calculate revenue if not present
    revenue_col = None
    if 'revenue' in df.columns:
        revenue_col = 'revenue'
    elif 'bookings' in df.columns and 'our_current_price' in df.columns:
        df = df.copy()
        df['revenue'] = df['bookings'] * df['our_current_price']
        revenue_col = 'revenue'
    
    metrics = {
        'total_revenue': df[revenue_col].sum() if revenue_col else 0,
        'avg_daily_revenue': df[revenue_col].mean() if revenue_col else 0,
        'total_bookings': df['bookings'].sum() if 'bookings' in df.columns else 0,
        'avg_occupancy': df['occupancy_rate'].mean() if 'occupancy_rate' in df.columns else 0,
        'avg_price': df['our_current_price'].mean() if 'our_current_price' in df.columns else 0,
        'avg_competitor_price': df['competitor_price'].mean() if 'competitor_price' in df.columns else 0,
    }
    
    # Try to find date column
    date_col = None
    for col in ['date', 'Date', 'DATE', 'booking_date', 'check_in_date']:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        try:
            metrics['date_range'] = (df[date_col].min(), df[date_col].max())
            if pd.api.types.is_datetime64_any_dtype(df[date_col]):
                metrics['days'] = (df[date_col].max() - df[date_col].min()).days + 1
            else:
                metrics['days'] = len(df[date_col].unique())
        except:
            pass
    
    return metrics


def resample_time_series(df, freq='D'):
    """Resample time series data to different frequencies."""
    if 'date' not in df.columns:
        return df
    
    df_indexed = df.set_index('date')
    
    resampled = df_indexed.resample(freq).agg({
        'revenue': 'sum',
        'bookings': 'sum',
        'our_current_price': 'mean',
        'competitor_price': 'mean',
        'occupancy_rate': 'mean'
    }).reset_index()
    
    return resampled


def detect_anomalies(df, column='revenue', threshold=2):
    """Detect anomalies using z-score method."""
    if column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return z_scores > threshold


def calculate_price_elasticity(df):
    """Calculate price elasticity of demand."""
    if not all(col in df.columns for col in ['our_current_price', 'bookings']):
        return None
    
    # Simple elasticity: % change in bookings / % change in price
    price_changes = df['our_current_price'].pct_change()
    booking_changes = df['bookings'].pct_change()
    
    # Remove infinite and NaN values
    valid = (price_changes != 0) & (price_changes.notna()) & (booking_changes.notna())
    elasticity = (booking_changes[valid] / price_changes[valid]).mean()
    
    return elasticity


def forecast_simple(df, column='revenue', periods=7):
    """Simple forecast using linear trend."""
    if column not in df.columns or 'date' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    values = df_sorted[column].values
    
    if len(values) < 2:
        return None
    
    # Linear trend
    x = np.arange(len(values))
    coeffs = np.polyfit(x, values, 1)
    
    # Forecast
    future_x = np.arange(len(values), len(values) + periods)
    forecast = np.polyval(coeffs, future_x)
    
    # Generate future dates
    last_date = df_sorted['date'].max()
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
    
    forecast_df = pd.DataFrame({
        'date': future_dates,
        column: forecast
    })
    
    return forecast_df


def decompose_time_series(df, column='revenue', model='additive', period=None):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Args:
        df: DataFrame with date and column
        column: Column to decompose
        model: 'additive' or 'multiplicative'
        period: Seasonal period (auto-detected if None)
    
    Returns:
        Decomposition result or None
    """
    if not STATSMODELS_AVAILABLE:
        return None
    
    if column not in df.columns or 'date' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date').set_index('date')
    
    if len(df_sorted) < 2 * (period or 7):  # Need at least 2 periods
        return None
    
    try:
        # Auto-detect period if not provided
        if period is None:
            # Try weekly seasonality (7 days)
            if len(df_sorted) >= 14:
                period = 7
            else:
                period = min(7, len(df_sorted) // 2)
        
        decomposition = seasonal_decompose(
            df_sorted[column],
            model=model,
            period=period,
            extrapolate_trend='freq'
        )
        return decomposition
    except Exception:
        return None


def forecast_exponential_smoothing(df, column='revenue', periods=7):
    """
    Forecast using exponential smoothing (Holt-Winters).
    
    Args:
        df: DataFrame with date and column
        column: Column to forecast
        periods: Number of periods to forecast
    
    Returns:
        Forecast DataFrame or None
    """
    if not STATSMODELS_AVAILABLE:
        return None
    
    if column not in df.columns or 'date' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date').set_index('date')
    
    if len(df_sorted) < 10:  # Need minimum data
        return None
    
    try:
        # Try to fit exponential smoothing model
        model = ExponentialSmoothing(
            df_sorted[column],
            seasonal_periods=min(7, len(df_sorted) // 2),
            trend='add',
            seasonal='add'
        )
        fitted_model = model.fit(optimized=True)
        forecast = fitted_model.forecast(steps=periods)
        
        # Generate future dates
        last_date = df_sorted.index.max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=periods, freq='D')
        
        forecast_df = pd.DataFrame({
            'date': future_dates,
            column: forecast.values
        })
        
        return forecast_df
    except Exception:
        return None


def calculate_seasonality_strength(df, column='revenue'):
    """Calculate strength of seasonality in time series."""
    if column not in df.columns or 'date' not in df.columns:
        return None
    
    decomposition = decompose_time_series(df, column)
    if decomposition is None:
        return None
    
    # Strength = Var(seasonal) / Var(original)
    seasonal_var = np.var(decomposition.seasonal)
    original_var = np.var(df[column])
    
    if original_var == 0:
        return 0
    
    strength = seasonal_var / original_var
    return min(strength, 1.0)  # Cap at 1.0

