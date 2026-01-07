import streamlit as st
import pandas as pd
from src.data_loader import load_data, process_data

st.set_page_config(
    page_title="Airbnb Dynamic Pricing Dashboard (By: Hiba Ehsan)",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Purple Theme (Used AI for this since I'm not good at CSS)
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #1a0a2e 0%, #2d1b4e 50%, #1a0a2e 100%);
    }
   
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d1b4e 0%, #1a0a2e 100%);
        border-right: 2px solid #9b59b6;
    }
   
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #d8b4fe !important;
    }
   
    /* Regular text */
    .stMarkdown, p, span, label {
        color: #e9d5ff !important;
    }
   
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #c084fc !important;
        font-weight: bold;
    }
   
    [data-testid="stMetricLabel"] {
        color: #a78bfa !important;
    }
   
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #7c3aed 0%, #9b59b6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
   
    .stButton > button:hover {
        background: linear-gradient(90deg, #8b5cf6 0%, #a855f7 100%);
        box-shadow: 0 0 15px rgba(139, 92, 246, 0.5);
    }
   
    /* Selectbox and inputs */
    .stSelectbox, .stTextInput, .stNumberInput {
        background-color: #2d1b4e;
    }
   
    [data-testid="stSelectbox"] > div > div {
        background-color: #3d2b5e !important;
        border: 1px solid #7c3aed;
    }
   
    /* Radio buttons */
    .stRadio > div {
        background-color: transparent;
    }
   
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #2d1b4e !important;
        color: #d8b4fe !important;
        border: 1px solid #7c3aed;
        border-radius: 8px;
    }
   
    /* Info, success, warning boxes */
    .stAlert {
        background-color: #2d1b4e;
        border: 1px solid #7c3aed;
        border-radius: 8px;
    }
   
    /* Dataframe */
    [data-testid="stDataFrame"] {
        border: 1px solid #7c3aed;
        border-radius: 8px;
    }
   
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #2d1b4e;
        border: 2px dashed #7c3aed;
        border-radius: 8px;
        padding: 10px;
    }
   
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #2d1b4e;
        border-radius: 8px;
    }
   
    .stTabs [data-baseweb="tab"] {
        color: #d8b4fe;
    }
   
    .stTabs [aria-selected="true"] {
        background-color: #7c3aed !important;
    }
   
    /* Slider */
    .stSlider > div > div > div {
        background-color: #7c3aed !important;
    }
   
    /* Progress bar */
    .stProgress > div > div {
        background-color: #7c3aed !important;
    }
   
    /* Cards/containers */
    [data-testid="stVerticalBlock"] > div {
        border-radius: 8px;
    }
   
    /* Links */
    a {
        color: #a78bfa !important;
    }
   
    a:hover {
        color: #c4b5fd !important;
    }
   
    /* Caption */
    .stCaption {
        color: #a78bfa !important;
    }
   
    /* Divider */
    hr {
        border-color: #7c3aed !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏠 Airbnb Dynamic Pricing Dashboard")
st.markdown("*Make smarter pricing decisions with data-driven insights*")
st.caption("DSC327 Data Visualization Project | USA Airbnb Dataset via Kaggle | By: Hiba Ehsan | SP24-BST-012")

# CSV Upload
st.sidebar.header("📤 Data Source")
data_source = st.sidebar.radio(
    "Choose data source",
    options=["Sample Data (CSV)", "Upload CSV", "Database"],
    help="Select where to load data from"
)

df = None
if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload Airbnb pricing CSV",
        type=['csv'],
        help="CSV should contain columns: date, listing_id, price, available, neighbourhood, room_type, number_of_reviews (price format e.g. 120 or $120)"
    )
   
    if uploaded_file is not None:
        try:
            # Try different encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)  # Reset file pointer
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
           
            if df is not None:
                df = process_data(df)
                st.sidebar.success(f"✅ Loaded {len(df):,} records")
                st.session_state['uploaded_data'] = df
            else:
                st.sidebar.error("Could not read CSV file")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
   
    # Use uploaded data if available
    if 'uploaded_data' in st.session_state:
        df = st.session_state['uploaded_data']
    else:
        st.info("👆 Please upload a CSV file in the sidebar")
        st.stop()
elif data_source == "Database":
    use_db = True
    df = load_data(use_db=use_db)
    if df.empty:
        st.warning("⚠️ Database connection failed or no data available. Using sample data.")
        df = load_data(use_db=False)
else:
    from src.data_loader_airbnb import load_airbnb_data
    df = load_airbnb_data()

# Store in session state for pages to use
if df is not None and not df.empty:
    st.session_state['airbnb_data'] = df
    st.session_state['hotel_data'] = df  # kept for older pages that still reference hotel_data

# Navigation
st.sidebar.header("🧭 Navigation")
st.sidebar.markdown("Use the pages above to explore:")
st.sidebar.markdown("- **📊 Overview**: Airbnb key metrics and time-series analysis")
st.sidebar.markdown("- **💹 Price Analysis**: Price distributions and neighborhood comparisons")
st.sidebar.markdown("- **🤖 ML Price Prediction**: Predict demand and simulate price scenarios")
st.sidebar.markdown("- **🔍 Price Optimization**: ML-powered price recommendations")
st.sidebar.markdown("- **🎨 Visualization Gallery**: Airbnb-specific visualizations")

# Home page content
if df is not None and not df.empty:
    st.header(" Kindly check my R Project too after this :((")
    st.markdown("""
    This dashboard helps you analyze and optimize **Airbnb listing** pricing using:
    - **Time-series analysis** for trend identification
    - **Machine learning** for price optimization and demand forecasting
    - **Interactive visualizations** from multiple libraries
    - **What-if simulations** for revenue and occupancy planning
   
    ### Quick Stats
    """)
   
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Date Range", f"{(df['date'].max() - df['date'].min()).days + 1} days")
    col3.metric("Listing Types", len(df['room_type'].unique()))
    col4.metric("Total Revenue", f"${df['revenue'].sum():,.0f}")
   
    st.info("💡 **Tip:** Use the sidebar to switch between data sources or navigate to specific analysis pages.")
   
    # Data preview
    with st.expander("📋 Preview Data", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
else:
    st.warning("⚠️ No data loaded. Please check your data source or upload a CSV file.")
