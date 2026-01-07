import pandas as pd
import streamlit as st
import os
from sqlalchemy import create_engine
from datetime import datetime


def get_db_engine():
    """Create database engine from environment variables or return None."""
    try:
        user = os.getenv("DB_USER", "root")
        password = os.getenv("DB_PASSWORD", "")
        host = os.getenv("DB_HOST", "localhost")
        database = os.getenv("DB_NAME", "data")
        
        if password:
            engine = create_engine(f"mysql+pymysql://{user}:{password}@{host}/{database}")
            return engine
    except Exception as e:
        st.warning(f"Database connection not available: {e}")
    return None


def load_and_transform_real_data(booking_path, tripadvisor_path):
    """
    Load booking_hotel.csv and tripadvisor_room.csv and transform them into
    the format expected by the dashboard.
    
    Args:
        booking_path: Path to booking_hotel.csv
        tripadvisor_path: Path to tripadvisor_room.csv
    
    Returns:
        DataFrame with columns: date, room_type, our_current_price, competitor_price, bookings, occupancy_rate
    """
    import numpy as np
    
    # Try different encodings
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    # Load booking data
    booking_df = None
    for encoding in encodings:
        try:
            booking_df = pd.read_csv(booking_path, encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue
    
    # Load tripadvisor data
    tripadvisor_df = None
    for encoding in encodings:
        try:
            tripadvisor_df = pd.read_csv(tripadvisor_path, encoding=encoding)
            break
        except (UnicodeDecodeError, Exception):
            continue
    
    if booking_df is None or tripadvisor_df is None:
        return pd.DataFrame()
    
    # Clean column names
    booking_df.columns = booking_df.columns.str.strip()
    tripadvisor_df.columns = tripadvisor_df.columns.str.strip()
    
    # Clean price columns (remove commas and spaces)
    def clean_price(price_str):
        if pd.isna(price_str):
            return np.nan
        try:
            # Remove spaces, commas, and convert to float
            cleaned = str(price_str).replace(',', '').replace(' ', '')
            return float(cleaned)
        except:
            return np.nan
    
    # Process booking data
    booking_df['our_current_price'] = booking_df['Room Price (in BDT or any other currency)'].apply(clean_price)
    booking_df['room_type'] = booking_df['Room Type'].fillna('Standard')
    booking_df['rating'] = pd.to_numeric(booking_df['Rating'], errors='coerce')
    
    # Process tripadvisor data as competitor prices
    tripadvisor_df['competitor_price'] = tripadvisor_df['Room Price (in BDT or any other currency)'].apply(clean_price)
    
    # Create time series data (30 days)
    dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
    
    # Sample hotels from booking data
    booking_sample = booking_df[booking_df['our_current_price'].notna()].head(100)
    
    # Get average competitor price
    avg_competitor_price = tripadvisor_df['competitor_price'].median()
    
    # Create dataset
    records = []
    for date in dates:
        for idx, row in booking_sample.iterrows():
            # Base values
            our_price = row['our_current_price']
            
            # Add some time-based variation (weekend prices higher)
            day_of_week = date.dayofweek
            price_multiplier = 1.0
            if day_of_week >= 5:  # Weekend
                price_multiplier = 1.15
            
            our_price = our_price * price_multiplier
            
            # Competitor price: use tripadvisor prices with variation
            # Match by similar price range
            price_range_lower = our_price * 0.8
            price_range_upper = our_price * 1.2
            competitor_matches = tripadvisor_df[
                (tripadvisor_df['competitor_price'] >= price_range_lower) &
                (tripadvisor_df['competitor_price'] <= price_range_upper)
            ]
            
            if len(competitor_matches) > 0:
                competitor_price = competitor_matches['competitor_price'].sample(1).iloc[0]
            else:
                # Fallback: use our price with some variation
                competitor_price = our_price * np.random.uniform(0.95, 1.05)
            
            # Generate synthetic occupancy based on rating and day of week
            base_occupancy = min(0.5 + (row['rating'] - 5) / 10, 0.95)
            if day_of_week >= 5:  # Weekend
                base_occupancy = min(base_occupancy + 0.1, 0.98)
            
            # Add some randomness
            occupancy = base_occupancy + np.random.uniform(-0.1, 0.1)
            occupancy = max(0.4, min(0.98, occupancy))
            
            # Calculate bookings (assume 50 rooms available)
            bookings = int(occupancy * 50)
            
            records.append({
                'date': date,
                'room_type': row['room_type'],
                'our_current_price': our_price,
                'competitor_price': competitor_price,
                'bookings': bookings,
                'occupancy_rate': occupancy
            })
    
    df = pd.DataFrame(records)
    
    # Remove any rows with NaN prices
    df = df.dropna(subset=['our_current_price', 'competitor_price'])
    
    return df


@st.cache_data(show_spinner="Loading hotel pricing data...")
def load_data(use_db=False, csv_path="data/hotels.csv"):
    """
    Loads hotel pricing data from database or CSV file.
    
    Args:
        use_db: If True, try to load from database first
        csv_path: Path to CSV file as fallback
    
    Returns:
        DataFrame with hotel pricing data
    """
    # Try database first if requested
    if use_db:
        engine = get_db_engine()
        if engine:
            try:
                df = pd.read_sql("SELECT * FROM hotels", engine)
                if not df.empty:
                    st.success("Loaded data from database")
                    return process_data(df)
            except Exception as e:
                st.warning(f"Database query failed: {e}. Falling back to CSV.")
    
    # Fallback to CSV - try to load actual data files first
    try:
        import os
        import numpy as np
        
        # Check if booking and tripadvisor files exist
        booking_path = "data/booking_hotel.csv"
        tripadvisor_path = "data/tripadvisor_room.csv"
        
        if os.path.exists(booking_path) and os.path.exists(tripadvisor_path):
            # Load and transform the actual data
            df = load_and_transform_real_data(booking_path, tripadvisor_path)
            if not df.empty:
                return process_data(df)
        
        # Fallback to hotels.csv if it exists
        if os.path.exists(csv_path):
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if df is not None:
                return process_data(df)
        
        st.error("No data files found. Please ensure booking_hotel.csv and tripadvisor_room.csv are in the data/ directory.")
        return pd.DataFrame()
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def process_data(df):
    """Process and enrich the hotel pricing dataframe."""
    if df.empty:
        return df
    
    # Convert date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Calculate revenue if not present
    if 'revenue' not in df.columns:
        if 'bookings' in df.columns and 'our_current_price' in df.columns:
            df['revenue'] = df['bookings'] * df['our_current_price']
    
    # Add time-series features
    if 'date' in df.columns:
        df['day_of_week'] = df['date'].dt.day_name()
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['month'] = df['date'].dt.month
        df['year'] = df['date'].dt.year
    
    # Sort by date
    if 'date' in df.columns:
        df = df.sort_values('date').reset_index(drop=True)
    
    return df


def filter_data(df, date_range=None, room_types=None):
    """Filter dataframe by date range and room types."""
    filtered = df.copy()
    
    if date_range and 'date' in filtered.columns:
        start_date, end_date = date_range
        filtered = filtered[(filtered['date'] >= start_date) & (filtered['date'] <= end_date)]
    
    if room_types and 'room_type' in filtered.columns:
        filtered = filtered[filtered['room_type'].isin(room_types)]
    
    return filtered