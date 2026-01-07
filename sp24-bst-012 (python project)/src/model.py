import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


# Note: We now use OLS (statsmodels) as the canonical linear model for
# interpretability and hypothesis testing (t-stats, p-values, F-test).


def prepare_features(df):
    """Prepare features for ML model."""
    if df.empty:
        return None, None
    
    # Create feature dataframe
    features = pd.DataFrame()
    
    # Time features
    if 'date' in df.columns:
        features['day_of_week_num'] = df['date'].dt.dayofweek
        features['day_of_month'] = df['date'].dt.day
        features['week_of_year'] = df['date'].dt.isocalendar().week
        features['month'] = df['date'].dt.month
    
    # Price features
    if 'competitor_price' in df.columns:
        features['competitor_price'] = df['competitor_price']
        features['price_diff'] = df.get('our_current_price', 0) - df['competitor_price']
        features['price_ratio'] = df.get('our_current_price', 1) / (df['competitor_price'] + 1)
    
    # Historical features (if available)
    if 'occupancy_rate' in df.columns:
        features['occupancy_rate'] = df['occupancy_rate']
    
    # Room type encoding
    if 'room_type' in df.columns:
        room_dummies = pd.get_dummies(df['room_type'], prefix='room')
        features = pd.concat([features, room_dummies], axis=1)
    
    # Target variable: bookings or revenue
    if 'bookings' in df.columns:
        target = df['bookings']
    elif 'revenue' in df.columns and 'our_current_price' in df.columns:
        # Estimate bookings from revenue
        target = df['revenue'] / (df['our_current_price'] + 1)
    else:
        return None, None
    
    # Remove rows with NaN
    valid_idx = ~(features.isna().any(axis=1) | target.isna())
    features = features[valid_idx]
    target = target[valid_idx]
    
    if len(features) == 0:
        return None, None
    
    return features, target


def train_demand_model(df, model_type='linear'):
    """
    Train a model to predict demand (bookings) based on features.

    Currently supports only linear OLS (statsmodels) for full interpretability
    and hypothesis testing (t-stats, p-values, F-test).

    Returns:
        fitted_statsmodels_obj, scaler (None), metrics (dict)
    """
    features, target = prepare_features(df)

    if features is None or len(features) < 10:
        return None, None, None

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    # Add constant for intercept
    X_train_sm = sm.add_constant(X_train, has_constant='add')
    X_test_sm = sm.add_constant(X_test, has_constant='add')

    # Fit OLS model
    try:
        ols_model = sm.OLS(y_train, X_train_sm).fit()
    except Exception as e:
        # Fallback: return None on failure
        return None, None, None

    # Evaluate
    y_pred = ols_model.predict(X_test_sm)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'mae': mae,
        'r2': r2,
        'train_size': len(X_train),
        'test_size': len(X_test),
        # include summary statistics
        'fvalue': getattr(ols_model, 'fvalue', None),
        'f_pvalue': getattr(ols_model, 'f_pvalue', None)
    }

    return ols_model, None, metrics


def optimize_price(df, room_type=None, target_occupancy=0.85, model_type='random_forest'):
    """
    Optimize price to maximize revenue while maintaining target occupancy.
    
    Args:
        df: DataFrame with hotel data
        room_type: Specific room type to optimize (None for all)
        target_occupancy: Target occupancy rate
        model_type: Model type for demand prediction
    
    Returns:
        Dictionary with optimization results
    """
    if room_type:
        df_filtered = df[df['room_type'] == room_type].copy()
    else:
        df_filtered = df.copy()
    
    if df_filtered.empty:
        return None
    
    # Train model
    model, scaler, metrics = train_demand_model(df_filtered, model_type)
    
    if model is None:
        return None
    
    # Get current average price
    current_price = df_filtered['our_current_price'].mean()
    competitor_price = df_filtered['competitor_price'].mean()
    current_occupancy = df_filtered['occupancy_rate'].mean()
    
    # Try different price points
    price_range = np.linspace(
        max(competitor_price * 0.7, current_price * 0.8),
        min(competitor_price * 1.3, current_price * 1.5),
        50
    )
    
    best_price = current_price
    best_revenue = df_filtered['revenue'].sum() if 'revenue' in df_filtered.columns else 0
    best_occupancy = current_occupancy
    
    results = []
    
    for test_price in price_range:
        # Prepare features for prediction
        test_df = df_filtered.copy()
        test_df['our_current_price'] = test_price
        test_df['price_diff'] = test_price - test_df['competitor_price']
        test_df['price_ratio'] = test_price / (test_df['competitor_price'] + 1)

        features, _ = prepare_features(test_df)
        if features is None or len(features) == 0:
            continue

        # Predict bookings
        try:
            # If model is statsmodels OLS result (scaler is None), add constant and predict
            if scaler is None:
                features_sm = sm.add_constant(features, has_constant='add')
                predicted_bookings = model.predict(features_sm)
            else:
                features_scaled = scaler.transform(features)
                predicted_bookings = model.predict(features_scaled)
        except Exception:
            # Skip this price if prediction fails
            continue

        predicted_bookings = np.maximum(predicted_bookings, 0)  # No negative bookings
        
        # Estimate occupancy (assuming fixed room capacity)
        if 'occupancy_rate' in df_filtered.columns:
            avg_rooms = df_filtered['occupancy_rate'].mean() * 100  # Assume 100 rooms
            predicted_occupancy = min(predicted_bookings.mean() / avg_rooms, 1.0) if avg_rooms > 0 else 0
        else:
            predicted_occupancy = min(predicted_bookings.mean() / 50, 1.0)  # Rough estimate
        
        # Calculate revenue
        predicted_revenue = (predicted_bookings * test_price).sum()
        
        results.append({
            'price': test_price,
            'predicted_bookings': predicted_bookings.mean(),
            'predicted_occupancy': predicted_occupancy,
            'predicted_revenue': predicted_revenue
        })
        
        # Update best if better and meets occupancy target
        if predicted_occupancy >= target_occupancy * 0.9:  # Allow 10% flexibility
            if predicted_revenue > best_revenue:
                best_price = test_price
                best_revenue = predicted_revenue
                best_occupancy = predicted_occupancy
    
    results_df = pd.DataFrame(results)
    
    return {
        'current_price': current_price,
        'recommended_price': best_price,
        'price_change_pct': ((best_price - current_price) / current_price * 100) if current_price > 0 else 0,
        'current_revenue': df_filtered['revenue'].sum() if 'revenue' in df_filtered.columns else 0,
        'predicted_revenue': best_revenue,
        'revenue_improvement_pct': ((best_revenue - (df_filtered['revenue'].sum() if 'revenue' in df_filtered.columns else 0)) / 
                                   (df_filtered['revenue'].sum() + 1) * 100),
        'current_occupancy': current_occupancy,
        'predicted_occupancy': best_occupancy,
        'model_metrics': metrics,
        'price_sensitivity': results_df
    }




