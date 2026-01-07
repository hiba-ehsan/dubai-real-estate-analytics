import pandas as pd
from src.visualization import (
    plot_demand_heatmap, plot_boxplot_by_day, plot_violin_by_day
)


def make_sample_df():
    df = pd.DataFrame({
        'date': pd.date_range('2026-01-01', periods=7, freq='D'),
        'our_current_price': [100,110,120,100,130,115,105],
        'occupancy_rate': [0.5,0.6,0.7,0.4,0.9,0.8,0.3],
        'room_type': ['A','A','B','B','A','B','A']
    })
    return df


def test_heatmap_with_date():
    df = make_sample_df()
    fig = plot_demand_heatmap(df)
    assert fig is not None, "Heatmap should be generated when date + occupancy present"


def test_heatmap_missing_occupancy():
    df = make_sample_df().drop(columns=['occupancy_rate'])
    fig = plot_demand_heatmap(df)
    assert fig is None, "Heatmap should return None when occupancy_rate is missing"


def test_boxplot_with_date():
    df = make_sample_df()
    img = plot_boxplot_by_day(df, column='our_current_price')
    assert img is not None, "Boxplot should generate an image string when date present"


def test_boxplot_missing_column():
    df = make_sample_df().drop(columns=['our_current_price'])
    img = plot_boxplot_by_day(df, column='our_current_price')
    assert img is None, "Boxplot should return None when requested column missing"


def test_violin_with_date():
    df = make_sample_df()
    img = plot_violin_by_day(df, column='our_current_price')
    assert img is not None, "Violin plot should generate an image string when date present"


def test_violin_missing_date_and_dayofweek():
    df = make_sample_df().drop(columns=['date'])
    img = plot_violin_by_day(df, column='our_current_price')
    assert img is None, "Violin plot should return None when no date or day_of_week column exists"
