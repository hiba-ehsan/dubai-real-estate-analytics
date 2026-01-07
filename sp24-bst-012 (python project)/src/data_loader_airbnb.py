"""Data loader and processor for Airbnb dataset."""
import pandas as pd
import numpy as np
import streamlit as st
import json
import re


@st.cache_data(show_spinner="Loading Airbnb data...")
def load_airbnb_data(csv_path="data/train.csv"):
    """
    Load and preprocess Airbnb train.csv dataset.
    
    Returns:
        DataFrame with processed Airbnb listing data
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Convert log_price to actual price
        df['price'] = np.exp(df['log_price'])
        
        # Process dates
        df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
        df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        
        # Calculate days since first/last review
        reference_date = pd.Timestamp('2017-01-01')  # Approximate dataset date
        df['days_since_first_review'] = (reference_date - df['first_review']).dt.days
        df['days_since_last_review'] = (reference_date - df['last_review']).dt.days
        df['host_experience_days'] = (reference_date - df['host_since']).dt.days
        
        # Convert boolean columns
        df['cleaning_fee'] = df['cleaning_fee'].astype(int)
        df['host_has_profile_pic'] = (df['host_has_profile_pic'] == 't').astype(int)
        df['host_identity_verified'] = (df['host_identity_verified'] == 't').astype(int)
        df['instant_bookable'] = (df['instant_bookable'] == 't').astype(int)
        
        # Process host_response_rate (remove % and convert to float)
        df['host_response_rate'] = df['host_response_rate'].str.rstrip('%').astype(float) / 100
        
        # Extract amenities count and specific amenities
        df = extract_amenities_features(df)
        
        # Fill missing values
        df['bathrooms'].fillna(df['bathrooms'].median(), inplace=True)
        df['bedrooms'].fillna(df['bedrooms'].median(), inplace=True)
        df['beds'].fillna(df['beds'].median(), inplace=True)
        df['review_scores_rating'].fillna(df['review_scores_rating'].median(), inplace=True)
        df['host_response_rate'].fillna(0, inplace=True)
        
        # Create price categories
        df['price_category'] = pd.cut(df['price'], 
                                      bins=[0, 75, 150, 250, np.inf],
                                      labels=['Budget', 'Moderate', 'Premium', 'Luxury'])
        
        # Calculate review frequency
        df['review_frequency'] = df['number_of_reviews'] / (df['days_since_first_review'] / 30 + 1)
        
        # Map to dashboard schema expected by pages and models
        df['our_current_price'] = df['price']

        # Competitor price: neighbourhood or city median fallback
        neigh_col = None
        for c in ['neighbourhood_cleansed','neighbourhood','city','region']:
            if c in df.columns:
                neigh_col = c
                break
        if neigh_col:
            try:
                df['competitor_price'] = df.groupby(neigh_col)['our_current_price'].transform('median').fillna(df['our_current_price'].median())
            except Exception:
                df['competitor_price'] = df['our_current_price'].median()
        else:
            df['competitor_price'] = df['our_current_price'].median()

        # Occupancy heuristics
        if 'availability_365' in df.columns:
            df['occupancy_rate'] = 1 - (df['availability_365'] / 365).clip(0,1)
        else:
            # Estimate occupancy from review counts (simple heuristic)
            max_reviews = df['number_of_reviews'].replace(0, np.nan).max()
            df['occupancy_rate'] = 0.25 + 0.5 * (df['number_of_reviews'] / (max_reviews + 1)).fillna(0)

        df['occupancy_rate'] = df['occupancy_rate'].clip(0.01, 0.99)
        df['bookings'] = df['occupancy_rate']

        # Revenue
        df['revenue'] = df['our_current_price'] * df['bookings']

        # Ensure there is a date column for time-based charts
        if 'last_review' in df.columns and 'date' not in df.columns:
            df['date'] = df['last_review'].fillna(pd.Timestamp.today())
        elif 'date' not in df.columns:
            df['date'] = pd.Timestamp.today()

        # Add room_type fallback
        if 'room_type' not in df.columns:
            if 'property_type' in df.columns:
                df['room_type'] = df['property_type']
            else:
                df['room_type'] = 'Listing'

        # Final processing
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)

        return df
        
    except FileNotFoundError:
        st.error(f"File not found: {csv_path}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


def extract_amenities_features(df):
    """Extract features from amenities JSON string."""
    
    # Key amenities to track
    key_amenities = [
        'Wifi', 'Wireless Internet', 'Internet',
        'Kitchen', 
        'Air conditioning', 'Heating',
        'TV', 'Cable TV',
        'Washer', 'Dryer',
        'Pool', 'Hot tub',
        'Gym', 'Elevator',
        'Doorman', 'Parking',
        'Breakfast',
        'Pets allowed', 'Pet',
        'Smoking allowed'
    ]
    
    def count_amenities(amenities_str):
        """Count number of amenities."""
        if pd.isna(amenities_str):
            return 0
        try:
            # Parse JSON-like string
            amenities_str = amenities_str.replace('""', '"')
            amenities = json.loads(amenities_str)
            return len(amenities)
        except:
            # Fallback: count commas
            return len(amenities_str.split(','))
    
    def has_amenity(amenities_str, amenity_name):
        """Check if listing has a specific amenity."""
        if pd.isna(amenities_str):
            return 0
        return 1 if any(a.lower() in amenities_str.lower() for a in [amenity_name]) else 0
    
    # Total amenities count
    df['amenities_count'] = df['amenities'].apply(count_amenities)
    
    # Specific amenities
    df['has_wifi'] = df['amenities'].apply(lambda x: has_amenity(x, 'wifi') or has_amenity(x, 'internet'))
    df['has_kitchen'] = df['amenities'].apply(lambda x: has_amenity(x, 'kitchen'))
    df['has_ac'] = df['amenities'].apply(lambda x: has_amenity(x, 'air conditioning'))
    df['has_heating'] = df['amenities'].apply(lambda x: has_amenity(x, 'heating'))
    df['has_tv'] = df['amenities'].apply(lambda x: has_amenity(x, 'tv'))
    df['has_pool'] = df['amenities'].apply(lambda x: has_amenity(x, 'pool'))
    df['has_parking'] = df['amenities'].apply(lambda x: has_amenity(x, 'parking'))
    
    return df


def get_price_statistics(df):
    """Calculate price statistics for the dataset."""
    return {
        'mean_price': df['price'].mean(),
        'median_price': df['price'].median(),
        'min_price': df['price'].min(),
        'max_price': df['price'].max(),
        'std_price': df['price'].std(),
        'total_listings': len(df),
        'cities': df['city'].nunique(),
        'property_types': df['property_type'].nunique(),
        'avg_reviews': df['number_of_reviews'].mean(),
        'avg_rating': df['review_scores_rating'].mean()
    }


def filter_airbnb_data(df, cities=None, property_types=None, room_types=None, 
                       price_range=None, min_reviews=None):
    """Filter Airbnb data based on various criteria."""
    filtered = df.copy()
    
    if cities:
        filtered = filtered[filtered['city'].isin(cities)]
    
    if property_types:
        filtered = filtered[filtered['property_type'].isin(property_types)]
    
    if room_types:
        filtered = filtered[filtered['room_type'].isin(room_types)]
    
    if price_range:
        filtered = filtered[(filtered['price'] >= price_range[0]) & 
                          (filtered['price'] <= price_range[1])]
    
    if min_reviews is not None:
        filtered = filtered[filtered['number_of_reviews'] >= min_reviews]
    
    return filtered
