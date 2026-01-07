"""Data preprocessing utilities for hotel pricing data."""
import pandas as pd
import numpy as np
from datetime import datetime


def clean_column_names(df):
    """Clean and standardize column names."""
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^a-z0-9_]', '', regex=True)
    return df


def handle_missing_values(df, strategy='drop'):
    """
    Handle missing values in the dataset.
    
    Args:
        df: DataFrame to clean
        strategy: 'drop', 'mean', 'median', or 'forward_fill'
    
    Returns:
        Cleaned DataFrame
    """
    if strategy == 'drop':
        return df.dropna()
    elif strategy == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        return df
    elif strategy == 'median':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        return df
    elif strategy == 'forward_fill':
        return df.fillna(method='ffill')
    return df


def remove_outliers(df, column, method='iqr', threshold=1.5):
    """
    Remove outliers from a specific column.
    
    Args:
        df: DataFrame
        column: Column name to check for outliers
        method: 'iqr' or 'zscore'
        threshold: IQR multiplier or z-score threshold
    
    Returns:
        DataFrame with outliers removed
    """
    if column not in df.columns:
        return df
    
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    elif method == 'zscore':
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        return df[z_scores < threshold]
    
    return df


def normalize_dates(df, date_column='date', date_format=None):
    """
    Normalize date columns to datetime objects.
    
    Args:
        df: DataFrame
        date_column: Name of the date column
        date_format: Optional date format string
    
    Returns:
        DataFrame with normalized dates
    """
    if date_column in df.columns:
        if date_format:
            df[date_column] = pd.to_datetime(df[date_column], format=date_format, errors='coerce')
        else:
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    return df


def add_time_features(df, date_column='date'):
    """
    Add time-based features from a date column.
    
    Args:
        df: DataFrame
        date_column: Name of the date column
    
    Returns:
        DataFrame with additional time features
    """
    if date_column not in df.columns:
        return df
    
    df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
    
    df['year'] = df[date_column].dt.year
    df['month'] = df[date_column].dt.month
    df['day'] = df[date_column].dt.day
    df['day_of_week'] = df[date_column].dt.day_name()
    df['day_of_week_num'] = df[date_column].dt.dayofweek
    df['week_of_year'] = df[date_column].dt.isocalendar().week
    df['quarter'] = df[date_column].dt.quarter
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6]).astype(int)
    
    return df


def encode_categorical(df, columns, method='onehot'):
    """
    Encode categorical variables.
    
    Args:
        df: DataFrame
        columns: List of column names to encode
        method: 'onehot' or 'label'
    
    Returns:
        DataFrame with encoded categorical variables
    """
    if method == 'onehot':
        return pd.get_dummies(df, columns=columns, drop_first=True)
    elif method == 'label':
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        for col in columns:
            if col in df.columns:
                df[col] = le.fit_transform(df[col].astype(str))
    return df


def scale_features(df, columns, method='standard'):
    """
    Scale numerical features.
    
    Args:
        df: DataFrame
        columns: List of column names to scale
        method: 'standard' or 'minmax'
    
    Returns:
        DataFrame with scaled features, scaler object
    """
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    
    if method == 'standard':
        scaler = StandardScaler()
    else:
        scaler = MinMaxScaler()
    
    df_scaled = df.copy()
    df_scaled[columns] = scaler.fit_transform(df[columns])
    
    return df_scaled, scaler