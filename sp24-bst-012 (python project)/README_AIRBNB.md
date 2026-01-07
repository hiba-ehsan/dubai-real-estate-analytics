# 🏡 Airbnb Dynamic Pricing Dashboard

A Streamlit dashboard focused on Airbnb listing pricing analysis and optimization using time-series analysis, machine learning, and interactive visualizations.

## 📋 Overview

This project applies pricing analytics specifically to Airbnb datasets. It demonstrates:

- **Listing-level analysis** including neighborhood comparisons
- **Time-series and trend analysis** where applicable
- **Machine learning** for demand (bookings) prediction and price optimization
- **Multiple visualization libraries** (Plotly, Matplotlib, Seaborn)

## 🚀 Features

- Overview dashboard with KPIs and time-series insights
- Price analysis and neighborhood comparisons
- ML-based demand modeling and price optimization
- Visualization gallery tailored to Airbnb features and amenities
- CSV upload or sample `data/train.csv` support

## 📦 Installation

1. Create and activate a virtual environment (Python 3.11 recommended)
   - Windows (PowerShell): `python -m venv .venv; .\\.venv\\Scripts\\Activate.ps1`
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app locally:

```bash
streamlit run app.py
```

> Tip: Run `pytest -q` to execute the test suite while developing.

## 📊 Data Expectations

The app supports typical Airbnb listings CSVs (e.g., Kaggle-style `train.csv`). Recommended columns (one or more):

- `price` or `log_price`
- `availability_365` (optional)
- `neighbourhood` / `city` (for competitor price estimates)
- `room_type` / `property_type`
- `number_of_reviews`, `review_scores_rating` (used to estimate occupancy)
- `first_review`, `last_review`, `host_since` (optional date fields)

The loader will try to infer the important columns and will create normalized columns for:

- `our_current_price`, `competitor_price`, `bookings`, `occupancy_rate`, `revenue`, `date`

## 🧭 Pages

- **Overview**: KPIs, time-series, seasonality
- **Price Analysis**: Price distributions and neighborhood comparisons
- **ML Price Prediction**: Train models and simulate price scenarios (includes a small child-friendly "Explain like I'm 10" dropdown in the Math & Calculations panel for presenters and demos)
- **Price Optimization**: ML-powered recommendations (global and by listing type)
- **Visualization Gallery**: Multiple charts and libraries

## 🔧 Tips

- If your CSV contains a `log_price` column, the loader will convert it to `price`.
- If `availability_365` is present, occupancy is derived from it; otherwise a heuristic based on reviews and price is used.
- The app stores dataset in session as `airbnb_data` and also `hotel_data` (for compatibility with existing modules).

## 🧪 Notes

This is a starting point for Airbnb pricing analysis. For production use, consider:
- More accurate competitor pricing (web-scraped or marketplace APIs)
- Better occupancy/booking ground-truth data
- More robust ML models and feature engineering
- Tests and CI, authentication, and data validation

---

## 🤖 Agent onboarding & contributor docs

Detailed onboarding steps for contributors and AI agents are available in `AGENT_INSTRUCTIONS.md`. It includes setup steps, example prompts for the agent, and guidance on where to find key files and how to add tests.

If you want, I can also update the top-level `README.md` to link to this Airbnb README or replace the project description entirely with Airbnb-focused content.