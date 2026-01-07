# SaaS Conversion Funnel Analysis
# Analyzing user behavior and conversion patterns from CSV data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. DATA LOADING & CLEANING


def load_and_clean_data():
    """Load the CSV data and perform initial cleaning"""
    
    # Load the data
    df = pd.read_csv('C:/Users/Bo$$/Downloads/data-formatted.csv')
    
    # Clean column names (remove extra spaces/commas)
    df.columns = df.columns.str.strip().str.replace(',', '')
    
    # Convert date columns
    date_columns = ['last_synced', 'lastactivetime', 'createdtime']
    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        # Make all datetimes timezone-naive
        df[col] = df[col].dt.tz_localize(None)
    
    # Create derived features
    df['days_since_created'] = (datetime.now() - df['createdtime']).dt.days
    df['days_since_active'] = (datetime.now() - df['lastactivetime']).dt.days
    df['is_activated'] = df['things'] > 0
    df['is_engaged'] = df['things'] >= 5
    
    # Clean plan subscriptions
    df['plan_type'] = df['Plan Subscribed'].fillna('Free Trial')
    df['is_paid'] = df['plan_type'].str.contains('Plan', na=False)
    
    # Revenue estimation (based on plan types)
    revenue_mapping = {
        'Free Trial': 0,
        'Prototype Plan $9': 9,
        'Startup Plan $99': 99
    }
    df['monthly_revenue'] = df['plan_type'].map(revenue_mapping).fillna(0)
    
    return df

# Load the data
df = load_and_clean_data()

print(" Data Loaded Successfully!")
print(f" Dataset Shape: {df.shape}")
print(f" Total Users: {len(df)}")
print(f" Total Monthly Revenue: ${df['monthly_revenue'].sum():,}")


# 2. EXPLORATORY DATA ANALYSIS

def analyze_conversion_funnel(df):
    """Analyze the conversion funnel from trial to paid"""
    
    total_users = len(df)
    activated_users = df['is_activated'].sum()
    engaged_users = df['is_engaged'].sum()
    paid_users = df['is_paid'].sum()
    
    funnel_data = {
        'Stage': ['Free Trial', 'Activated (1+ things)', 'Engaged (5+ things)', 'Paid Plans'],
        'Users': [total_users, activated_users, engaged_users, paid_users],
        'Conversion_Rate': [100, (activated_users/total_users)*100, 
                           (engaged_users/total_users)*100, (paid_users/total_users)*100]
    }
    
    funnel_df = pd.DataFrame(funnel_data)
    
    # Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Funnel chart
    ax1.barh(funnel_df['Stage'], funnel_df['Users'], color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
    ax1.set_xlabel('Number of Users')
    ax1.set_title('Conversion Funnel Analysis')
    for i, v in enumerate(funnel_df['Users']):
        ax1.text(v + 2, i, str(v), va='center', fontweight='bold')
    
    # Conversion rates
    ax2.bar(funnel_df['Stage'], funnel_df['Conversion_Rate'], color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'])
    ax2.set_ylabel('Conversion Rate (%)')
    ax2.set_title('Stage-wise Conversion Rates')
    ax2.tick_params(axis='x', rotation=45)
    for i, v in enumerate(funnel_df['Conversion_Rate']):
        ax2.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return funnel_df

# Run funnel analysis
print(" CONVERSION FUNNEL ANALYSIS")
print("=" * 50)
funnel_results = analyze_conversion_funnel(df)
print(funnel_results)

# 3. USER SEGMENTATION ANALYSIS

def analyze_user_segments(df):
    """Segment users based on behavior patterns"""
    
    # Create segments based on usage patterns
    def categorize_user(row):
        if row['things'] == 0:
            return 'Inactive'
        elif row['things'] < 5:
            return 'Explorer'
        elif row['things'] < 20:
            return 'Active'
        else:
            return 'Power User'
    
    df['user_segment'] = df.apply(categorize_user, axis=1)
    
    # Analyze segments
    segment_analysis = df.groupby('user_segment').agg({
        'userid': 'count',
        'is_paid': 'mean',
        'monthly_revenue': 'mean',
        'days_since_active': 'mean',
        'things': 'mean'
    }).round(2)
    
    segment_analysis.columns = ['Users', 'Conversion_Rate', 'Avg_Revenue', 'Days_Since_Active', 'Avg_Things']
    segment_analysis['Conversion_Rate'] = segment_analysis['Conversion_Rate'] * 100
    
    # Visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Segment distribution
    segment_counts = df['user_segment'].value_counts()
    ax1.pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', startangle=90)
    ax1.set_title('User Segment Distribution')
    
    # Conversion rates by segment
    ax2.bar(segment_analysis.index, segment_analysis['Conversion_Rate'])
    ax2.set_title('Conversion Rate by Segment')
    ax2.set_ylabel('Conversion Rate (%)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Revenue by segment
    ax3.bar(segment_analysis.index, segment_analysis['Avg_Revenue'])
    ax3.set_title('Average Revenue by Segment')
    ax3.set_ylabel('Monthly Revenue ($)')
    ax3.tick_params(axis='x', rotation=45)
    
    # Activity vs conversion scatter
    ax4.scatter(df['things'], df['monthly_revenue'], alpha=0.6, c=df['days_since_active'], cmap='viridis')
    ax4.set_xlabel('Number of Things Created')
    ax4.set_ylabel('Monthly Revenue ($)')
    ax4.set_title('Usage vs Revenue (Color = Days Since Active)')
    plt.colorbar(ax4.collections[0], ax=ax4, label='Days Since Active')
    
    plt.tight_layout()
    plt.show()
    
    return segment_analysis

print("\n👥 USER SEGMENTATION ANALYSIS")
print("=" * 50)
segment_results = analyze_user_segments(df)
print(segment_results)

# 4. GEOGRAPHIC & DORMANCY ANALYSIS

def analyze_geography_and_dormancy(df):
    """Analyze geographic patterns and user dormancy"""
    
    # Geographic analysis
    geo_analysis = df.groupby('location').agg({
        'userid': 'count',
        'is_paid': 'mean',
        'monthly_revenue': 'sum',
        'things': 'mean'
    }).round(2)
    
    geo_analysis.columns = ['Users', 'Conversion_Rate', 'Total_Revenue', 'Avg_Things']
    geo_analysis['Conversion_Rate'] = geo_analysis['Conversion_Rate'] * 100
    geo_analysis = geo_analysis.sort_values('Users', ascending=False).head(10)
    
    # Dormancy analysis
    dormancy_bins = [0, 7, 30, 90, 365, float('inf')]
    dormancy_labels = ['0-7 days', '8-30 days', '31-90 days', '91-365 days', '365+ days']
    
    df['dormancy_bucket'] = pd.cut(df['Days Dormant'], bins=dormancy_bins, labels=dormancy_labels, right=False)
    
    dormancy_analysis = df.groupby('dormancy_bucket').agg({
        'userid': 'count',
        'is_paid': 'mean',
        'things': 'mean'
    }).round(2)
    
    dormancy_analysis.columns = ['Users', 'Conversion_Rate', 'Avg_Things']
    dormancy_analysis['Conversion_Rate'] = dormancy_analysis['Conversion_Rate'] * 100
    
    # Visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top locations by users
    ax1.barh(geo_analysis.index[::-1], geo_analysis['Users'][::-1])
    ax1.set_title('Users by Geographic Location')
    ax1.set_xlabel('Number of Users')
    
    # Conversion rate by location
    ax2.bar(range(len(geo_analysis)), geo_analysis['Conversion_Rate'])
    ax2.set_title('Conversion Rate by Location')
    ax2.set_ylabel('Conversion Rate (%)')
    ax2.set_xticks(range(len(geo_analysis)))
    ax2.set_xticklabels(geo_analysis.index, rotation=45, ha='right')
    
    # Dormancy distribution
    ax3.bar(dormancy_analysis.index, dormancy_analysis['Users'])
    ax3.set_title('User Distribution by Dormancy')
    ax3.set_ylabel('Number of Users')
    ax3.tick_params(axis='x', rotation=45)
    
    # Dormancy vs conversion
    ax4.plot(dormancy_analysis.index, dormancy_analysis['Conversion_Rate'], marker='o', linewidth=2, markersize=8)
    ax4.set_title('Conversion Rate vs User Dormancy')
    ax4.set_ylabel('Conversion Rate (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return geo_analysis, dormancy_analysis

print("\n GEOGRAPHIC & DORMANCY ANALYSIS")
print("=" * 50)
geo_results, dormancy_results = analyze_geography_and_dormancy(df)

print("\nTop Geographic Markets:")
print(geo_results)

print("\nDormancy Analysis:")
print(dormancy_results)

# 5. PREDICTIVE MODELING SETUP

def prepare_features_for_modeling(df):
    """Prepare features for machine learning models"""
    
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, roc_auc_score
    
    # Feature engineering
    features_df = df.copy()
    
    # Encode categorical variables
    le_location = LabelEncoder()
    features_df['location_encoded'] = le_location.fit_transform(features_df['location'].fillna('Unknown'))
    
    le_timezone = LabelEncoder()
    features_df['timezone_encoded'] = le_timezone.fit_transform(features_df['timezone'].fillna('Unknown'))
    
    # Select features for modeling
    feature_columns = [
        'things', 'links', 'reports', 'dashboards', 'users', 'groups', 'datapoints',
        'emails', 'sms', 'days_since_created', 'days_since_active', 'Days Dormant',
        'location_encoded', 'timezone_encoded'
    ]
    
    # Prepare data
    X = features_df[feature_columns].fillna(0)
    y = features_df['is_paid']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    print(" Training Predictive Models...")
    
    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    
    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Model evaluation
    print("\n MODEL PERFORMANCE")
    print("=" * 30)
    print("\nRandom Forest:")
    print(f"AUC-ROC: {roc_auc_score(y_test, rf_prob):.3f}")
    print(classification_report(y_test, rf_pred))
    
    print("\nLogistic Regression:")
    print(f"AUC-ROC: {roc_auc_score(y_test, lr_prob):.3f}")
    print(classification_report(y_test, lr_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n TOP FEATURES FOR CONVERSION PREDICTION:")
    print(feature_importance.head(10))
    
    # Visualization
    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance['feature'][:10][::-1], feature_importance['importance'][:10][::-1])
    plt.title('Top 10 Features for Conversion Prediction')
    plt.xlabel('Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return rf_model, lr_model, feature_importance

# Run predictive modeling
print("\n PREDICTIVE MODELING")
print("=" * 50)
rf_model, lr_model, feature_importance = prepare_features_for_modeling(df)

# 6. BUSINESS INSIGHTS & RECOMMENDATIONS


def generate_business_insights(df, funnel_results, segment_results):
    """Generate actionable business insights"""
    
    print("\n💡 KEY BUSINESS INSIGHTS")
    print("=" * 50)
    
    # Critical metrics
    total_users = len(df)
    activation_rate = (df['is_activated'].sum() / total_users) * 100
    conversion_rate = (df['is_paid'].sum() / total_users) * 100
    avg_revenue_per_user = df['monthly_revenue'].mean()
    
    print(f"\nCurrent Performance:")
    print(f"   • Total Users: {total_users}")
    print(f"   • Activation Rate: {activation_rate:.1f}%")
    print(f"   • Trial-to-Paid Conversion: {conversion_rate:.1f}%")
    print(f"   • Average Revenue per User: ${avg_revenue_per_user:.2f}")
    
    # Identify issues
    inactive_users = (df['user_segment'] == 'Inactive').sum()
    dormant_users = (df['Days Dormant'] > 90).sum()
    
    print(f"\n Critical Issues:")
    print(f"   • {inactive_users} users ({(inactive_users/total_users)*100:.1f}%) never activated")
    print(f"   • {dormant_users} users ({(dormant_users/total_users)*100:.1f}%) dormant >90 days")
    print(f"   • Major drop-off at activation stage: {100-activation_rate:.1f}% loss")
    
    # Opportunities
    print(f"\n High-Impact Opportunities:")
    print(f"   • Improve onboarding: Could activate +{int((total_users * 0.25))} users")
    print(f"   • Smart upgrade prompts: Could convert +{int((total_users * 0.15))} users")
    print(f"   • Re-engagement campaigns: Could recover {int(dormant_users * 0.2)} dormant users")
    
    # Revenue projections
    potential_revenue = (total_users * 0.25 * 9) + (total_users * 0.15 * 50)  # Conservative estimates
    print(f"   • Potential monthly revenue increase: ${potential_revenue:.0f}")
    
    return {
        'total_users': total_users,
        'activation_rate': activation_rate,
        'conversion_rate': conversion_rate,
        'potential_revenue': potential_revenue
    }

# Generate insights
insights = generate_business_insights(df, funnel_results, segment_results)

print("\n ANALYSIS COMPLETE!")
print("=" * 50)
print("\n Next Steps:")
print("   1. Implement guided onboarding flow")
print("   2. Set up behavioral email triggers")
print("   3. Create A/B testing framework")
print("   4. Build real-time conversion dashboard")
print("   5. Deploy churn prediction model")