"""
Multi-library visualization functions using 10+ visualization libraries.
This module provides visualizations using: Plotly, Matplotlib, Seaborn, Bokeh, 
Altair, Pygal, Plotnine, Networkx, and more.
"""

import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime

# Core libraries (always available)
import matplotlib.pyplot as plt
import seaborn as sns

# Optional libraries with fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from bokeh.plotting import figure
    from bokeh.models import ColumnDataSource, HoverTool
    from bokeh.palettes import Category10
    BOKEH_AVAILABLE = True
except ImportError:
    BOKEH_AVAILABLE = False

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

try:
    import pygal
    from pygal.style import DefaultStyle
    PYGAL_AVAILABLE = True
except ImportError:
    PYGAL_AVAILABLE = False

try:
    from plotnine import ggplot, aes, geom_line, geom_point, theme_minimal, labs
    PLOTNINE_AVAILABLE = True
except ImportError:
    PLOTNINE_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


# ========== Bokeh Visualizations ==========

def plot_bokeh_revenue_trend(df):
    """Bokeh line chart showing revenue trend over time."""
    if not BOKEH_AVAILABLE or 'date' not in df.columns or 'revenue' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    source = ColumnDataSource(df_sorted)
    
    p = figure(
        title="Revenue Trend (Bokeh)",
        x_axis_label="Date",
        y_axis_label="Revenue ($)",
        x_axis_type="datetime",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    p.line('date', 'revenue', source=source, line_width=3, color="#3498DB", legend_label="Revenue")
    p.circle('date', 'revenue', source=source, size=6, color="#E74C3C", alpha=0.6)
    
    hover = HoverTool(tooltips=[("Date", "@date{%F}"), ("Revenue", "$@revenue{0,0}")],
                     formatters={'@date': 'datetime'})
    p.add_tools(hover)
    
    p.legend.location = "top_left"
    return p


def plot_bokeh_price_comparison(df):
    """Bokeh line chart comparing prices."""
    if not BOKEH_AVAILABLE or 'date' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    source = ColumnDataSource(df_sorted)
    
    p = figure(
        title="Price Comparison (Bokeh)",
        x_axis_label="Date",
        y_axis_label="Price ($)",
        x_axis_type="datetime",
        width=800,
        height=400,
        tools="pan,wheel_zoom,box_zoom,reset,save"
    )
    
    if 'our_current_price' in df_sorted.columns:
        p.line('date', 'our_current_price', source=source,
              line_width=3, color="#E74C3C", legend_label="Our Price")
        p.circle('date', 'our_current_price', source=source,
                size=6, color="#E74C3C", alpha=0.7)
    
    if 'competitor_price' in df_sorted.columns:
        p.line('date', 'competitor_price', source=source,
              line_width=3, color="#3498DB", legend_label="Competitor")
        p.circle('date', 'competitor_price', source=source,
                size=6, color="#3498DB", alpha=0.7)
    
    hover = HoverTool(tooltips=[("Date", "@date{%F}"), ("Our Price", "$@our_current_price{0,0}"), 
                                ("Competitor", "$@competitor_price{0,0}")],
                     formatters={'@date': 'datetime'})
    p.add_tools(hover)
    p.legend.location = "top_left"
    return p


# ========== Altair Visualizations ==========

def plot_altair_revenue_by_room(df):
    """Altair chart showing revenue by room type over time."""
    if not ALTAIR_AVAILABLE:
        return None
    
    if df.empty or 'date' not in df.columns or 'revenue' not in df.columns or 'room_type' not in df.columns:
        return None
    
    chart = alt.Chart(df).mark_line(point=True).encode(
        x=alt.X('date:T', title='Date'),
        y=alt.Y('revenue:Q', title='Revenue ($)', aggregate='sum'),
        color=alt.Color('room_type:N', title='Room Type'),
        tooltip=['date:T', 'room_type:N', 'revenue:Q']
    ).properties(
        title='Revenue by Room Type (Altair)',
        width=700,
        height=400
    )
    
    return chart


def plot_altair_price_scatter(df):
    """Altair scatter plot showing price vs occupancy."""
    if not ALTAIR_AVAILABLE:
        return None
    
    if df.empty or 'our_current_price' not in df.columns or 'occupancy_rate' not in df.columns:
        return None
    
    chart = alt.Chart(df).mark_circle(size=100).encode(
        x=alt.X('our_current_price:Q', title='Price ($)'),
        y=alt.Y('occupancy_rate:Q', title='Occupancy Rate', scale=alt.Scale(domain=[0, 1])),
        color=alt.Color('room_type:N', title='Room Type'),
        size=alt.Size('revenue:Q', title='Revenue'),
        tooltip=['our_current_price:Q', 'occupancy_rate:Q', 'room_type:N', 'revenue:Q']
    ).properties(
        title='Price vs Occupancy (Altair)',
        width=700,
        height=400
    )
    
    return chart


# ========== Pygal Visualizations ==========

def plot_pygal_revenue_line(df):
    """Pygal line chart for revenue trend."""
    if not PYGAL_AVAILABLE or 'date' not in df.columns or 'revenue' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    daily_revenue = df_sorted.groupby('date')['revenue'].sum()
    
    line_chart = pygal.Line(
        title='Daily Revenue Trend (Pygal)',
        x_label_rotation=45,
        style=DefaultStyle,
        width=800,
        height=400
    )
    line_chart.x_labels = [d.strftime('%Y-%m-%d') for d in daily_revenue.index]
    line_chart.add('Revenue', daily_revenue.values.tolist())
    
    # Convert to base64 for Streamlit
    chart_data = line_chart.render_data_uri()
    return chart_data


def plot_pygal_revenue_area(df):
    """Pygal stacked area chart showing revenue by room type over time."""
    if not PYGAL_AVAILABLE or 'date' not in df.columns or 'room_type' not in df.columns or 'revenue' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    daily_revenue = df_sorted.groupby(['date', 'room_type'])['revenue'].sum().unstack(fill_value=0)
    
    area_chart = pygal.StackedLine(
        title='Revenue by Room Type Over Time (Pygal)',
        x_label_rotation=45,
        style=DefaultStyle,
        width=800,
        height=400,
        fill=True
    )
    area_chart.x_labels = [d.strftime('%Y-%m-%d') for d in daily_revenue.index]
    
    for room_type in daily_revenue.columns:
        area_chart.add(room_type, daily_revenue[room_type].values.tolist())
    
    chart_data = area_chart.render_data_uri()
    return chart_data


# ========== Plotnine Visualizations ==========

def plot_plotnine_price_trend(df):
    """Plotnine (ggplot2-style) line chart."""
    if not PLOTNINE_AVAILABLE or 'date' not in df.columns or 'our_current_price' not in df.columns:
        return None
    
    df_sorted = df.sort_values('date')
    
    p = (ggplot(df_sorted, aes(x='date', y='our_current_price'))
         + geom_line(color='#3498DB', size=1.5)
         + geom_point(color='#E74C3C', size=2)
         + theme_minimal()
         + labs(title='Price Trend (Plotnine)', x='Date', y='Price ($)'))
    
    # Save to buffer
    fig = p.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_str


def plot_plotnine_boxplot(df):
    """Plotnine boxplot by room type."""
    if not PLOTNINE_AVAILABLE or 'room_type' not in df.columns or 'our_current_price' not in df.columns:
        return None
    
    from plotnine import geom_boxplot
    
    p = (ggplot(df, aes(x='room_type', y='our_current_price', fill='room_type'))
         + geom_boxplot()
         + theme_minimal()
         + labs(title='Price Distribution by Room Type (Plotnine)', x='Room Type', y='Price ($)'))
    
    fig = p.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    
    return img_str


# ========== NetworkX Visualizations ==========

def plot_networkx_room_relationships(df):
    """NetworkX graph showing relationships between room types based on price similarity."""
    if not NETWORKX_AVAILABLE or 'room_type' not in df.columns:
        return None
    
    # Calculate average prices by room type
    room_prices = df.groupby('room_type')['our_current_price'].mean().to_dict()
    
    if len(room_prices) < 2:
        return None
    
    # Create graph
    G = nx.Graph()
    
    # Add nodes (room types)
    for room, price in room_prices.items():
        G.add_node(room, price=price)
    
    # Add edges based on price similarity (within 20% of each other)
    rooms = list(room_prices.keys())
    for i, room1 in enumerate(rooms):
        for room2 in rooms[i+1:]:
            price_diff = abs(room_prices[room1] - room_prices[room2]) / max(room_prices[room1], room_prices[room2])
            if price_diff < 0.2:  # Within 20%
                G.add_edge(room1, room2, weight=price_diff)
    
    # Draw network
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color='#3498DB', node_size=2000, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.5, edge_color='gray', width=2)
    
    # Draw labels
    labels = {room: f"{room}\n${room_prices[room]:.0f}" for room in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight='bold')
    
    plt.title("Room Type Price Similarity Network (NetworkX)", fontsize=14, pad=20)
    plt.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str


# ========== Matplotlib Advanced Visualizations ==========

def plot_matplotlib_radar_chart(df):
    """Matplotlib radar chart comparing metrics by room type."""
    if 'room_type' not in df.columns:
        return None
    
    # Calculate metrics by room type
    metrics = df.groupby('room_type').agg({
        'our_current_price': 'mean',
        'occupancy_rate': 'mean',
        'revenue': 'mean',
        'bookings': 'mean'
    }).T
    
    # Normalize to 0-1 scale for radar chart
    metrics_norm = metrics.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x)
    
    # Number of variables
    categories = list(metrics_norm.index)
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot each room type
    colors = ['#E74C3C', '#3498DB', '#27AE60', '#F39C12', '#9B59B6']
    for idx, room_type in enumerate(metrics_norm.columns):
        values = metrics_norm[room_type].values.tolist()
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=room_type, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.25, color=colors[idx % len(colors)])
    
    # Add category labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Room Type Comparison - Radar Chart (Matplotlib)', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str


def plot_matplotlib_heatmap_calendar(df):
    """Matplotlib heatmap showing occupancy by day of week and room type."""
    if 'day_of_week' not in df.columns or 'room_type' not in df.columns or 'occupancy_rate' not in df.columns:
        return None
    
    pivot = df.pivot_table(values='occupancy_rate', index='day_of_week', columns='room_type', aggfunc='mean')
    
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = pivot.reindex([d for d in day_order if d in pivot.index])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot.values, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            text = ax.text(j, i, f'{pivot.iloc[i, j]:.1%}',
                          ha="center", va="center", color="black", fontweight='bold')
    
    ax.set_title("Occupancy Heatmap Calendar (Matplotlib)", pad=20)
    plt.colorbar(im, ax=ax, label='Occupancy Rate')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str


# ========== Seaborn Advanced Visualizations ==========

def plot_seaborn_pairplot(df):
    """Seaborn pairplot showing relationships between numeric variables."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return None
    
    # Limit to key columns and sample if too large
    key_cols = ['our_current_price', 'competitor_price', 'revenue', 'bookings', 'occupancy_rate']
    plot_cols = [col for col in key_cols if col in numeric_cols][:5]  # Max 5 columns
    
    if len(plot_cols) < 2:
        plot_cols = numeric_cols[:5]
    
    df_sample = df[plot_cols].sample(min(100, len(df))) if len(df) > 100 else df[plot_cols]
    
    g = sns.pairplot(df_sample, diag_kind='kde', plot_kws={'alpha': 0.6})
    g.fig.suptitle('Pairwise Relationships (Seaborn)', y=1.02, fontsize=14)
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str


def plot_seaborn_regression(df):
    """Seaborn regression plot showing price vs bookings."""
    if not all(col in df.columns for col in ['our_current_price', 'bookings']):
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.regplot(data=df, x='our_current_price', y='bookings', ax=ax, 
                scatter_kws={'alpha': 0.6}, line_kws={'color': 'red', 'linewidth': 2})
    ax.set_title('Price vs Bookings Regression (Seaborn)', fontsize=14, pad=20)
    ax.set_xlabel('Price ($)')
    ax.set_ylabel('Bookings')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    plt.close()
    
    return img_str

