import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from bokeh.plotting import figure, save, output_file
from bokeh.io import curdoc
import altair as alt
import pygal
from plotnine import ggplot, aes, geom_line, geom_point, geom_smooth, ggtitle, xlab, ylab, theme_bw, ggsave
import geopandas as gpd
from shapely.geometry import Point
import folium
import networkx as nx
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

# Set style for matplotlib and seaborn
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Generate sample data
np.random.seed(42)
dates = pd.date_range(start='2025-01-01', periods=50, freq='D')
categories = ['A', 'B', 'C', 'D']
data = {
    'date': dates,
    'category': np.random.choice(categories, 50),
    'value1': np.random.randn(50).cumsum() + 100,
    'value2': np.random.randn(50).cumsum() + 50,
    'value3': np.random.randn(50) * 10 + 30,
    'value4': np.random.randn(50) * 5 + 20,
    'temperature': np.random.randn(50) * 5 + 25,
    'humidity': np.random.randn(50) * 10 + 60
}
df = pd.DataFrame(data)

# Create output directory
os.makedirs('output', exist_ok=True)

print("=" * 80)
print("DATA VISUALIZATION ASSIGNMENT - 10 Libraries, 2 Graphs Each")
print("=" * 80)


# 1. MATPLOTLIB.PYPLOT

print("\n1. MATPLOTLIB.PYPLOT")

# Graph 1: Line plot with multiple series
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(df['date'], df['value1'], label='Value 1', linewidth=2, marker='o', markersize=4)
ax.plot(df['date'], df['value2'], label='Value 2', linewidth=2, marker='s', markersize=4)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Values', fontsize=12)
ax.set_title('Time Series Line Plot - Multiple Series', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('output/1_matplotlib_line.png', dpi=300, bbox_inches='tight')
print("  ✓ Graph 1: Line plot saved to output/1_matplotlib_line.png")
plt.close()

# Graph 2: Scatter plot with color mapping
fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(df['value1'], df['value3'], c=df['value4'], 
                     s=df['value2']*2, alpha=0.6, cmap='viridis')
ax.set_xlabel('Value 1', fontsize=12)
ax.set_ylabel('Value 3', fontsize=12)
ax.set_title('Scatter Plot with Color and Size Mapping', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='Value 4')
plt.tight_layout()
plt.savefig('output/1_matplotlib_scatter.png', dpi=300, bbox_inches='tight')
print("  ✓ Graph 2: Scatter plot saved to output/1_matplotlib_scatter.png")
plt.close()


# 2. SEABORN

print("\n2. SEABORN")

# Graph 1: Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
correlation_matrix = df[['value1', 'value2', 'value3', 'value4', 'temperature', 'humidity']].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('output/2_seaborn_heatmap.png', dpi=300, bbox_inches='tight')
print("  ✓ Graph 1: Heatmap saved to output/2_seaborn_heatmap.png")
plt.close()

# Graph 2: Violin plot
fig, ax = plt.subplots(figsize=(10, 6))
sns.violinplot(data=df, x='category', y='value1', inner='box', palette='Set2', ax=ax)
ax.set_xlabel('Category', fontsize=12)
ax.set_ylabel('Value 1', fontsize=12)
ax.set_title('Violin Plot - Value1 Distribution by Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('output/2_seaborn_violin.png', dpi=300, bbox_inches='tight')
print("  ✓ Graph 2: Violin plot saved to output/2_seaborn_violin.png")
plt.close()


# 3. PLOTLY.EXPRESS

print("\n3. PLOTLY.EXPRESS")

# Graph 1: 3D Scatter plot
fig = px.scatter_3d(df, x='value1', y='value2', z='value3', 
                    color='category', size='value4',
                    title='3D Scatter Plot - Value1 vs Value2 vs Value3',
                    labels={'value1': 'Value 1', 'value2': 'Value 2', 'value3': 'Value 3'})
fig.update_layout(width=900, height=700)
fig.write_html('output/3_plotly_express_3d_scatter.html')
print("  ✓ Graph 1: 3D Scatter plot saved to output/3_plotly_express_3d_scatter.html")

# Graph 2: Area chart
fig = px.area(df, x='date', y='value1', color='category',
              title='Stacked Area Chart - Value1 over Time by Category',
              labels={'value1': 'Value 1', 'date': 'Date'})
fig.update_layout(width=1000, height=600)
fig.write_html('output/3_plotly_express_area.html')
print("  ✓ Graph 2: Area chart saved to output/3_plotly_express_area.html")


# 4. BOKEH.PLOTTING

print("\n4. BOKEH.PLOTTING")

# Graph 1: Multi-line plot with interactive features
p1 = figure(width=900, height=600, title='Time Series - Multiple Values', 
           x_axis_label='Date', y_axis_label='Values',
           x_axis_type='datetime')
p1.line(df['date'], df['value1'], legend_label='Value 1', line_width=2, color='navy')
p1.line(df['date'], df['value2'], legend_label='Value 2', line_width=2, color='firebrick')
p1.line(df['date'], df['value3'], legend_label='Value 3', line_width=2, color='green')
p1.legend.location = 'top_left'
p1.legend.click_policy = 'hide'
output_file('output/4_bokeh_line.html')
save(p1)
print("  ✓ Graph 1: Multi-line plot saved to output/4_bokeh_line.html")

# Graph 2: Scatter plot with hover tooltips
p2 = figure(width=900, height=600, title='Scatter Plot - Value1 vs Value3',
           x_axis_label='Value 1', y_axis_label='Value 3',
           tools='hover,pan,wheel_zoom,box_zoom,reset')
colors = {'A': 'red', 'B': 'blue', 'C': 'green', 'D': 'orange'}
for cat in df['category'].unique():
    mask = df['category'] == cat
    p2.scatter(df[mask]['value1'], df[mask]['value3'], 
              size=8, alpha=0.6, color=colors[cat], legend_label=cat)
p2.legend.location = 'top_left'
output_file('output/4_bokeh_scatter.html')
save(p2)
print("  ✓ Graph 2: Scatter plot saved to output/4_bokeh_scatter.html")


# 5. ALTAIR

print("\n5. ALTAIR")

# Graph 1: Area chart with multiple layers
chart1 = alt.Chart(df).mark_area(opacity=0.7).encode(
    x=alt.X('date:T', title='Date'),
    y=alt.Y('value1:Q', title='Value 1'),
    color=alt.Color('category:N', scale=alt.Scale(scheme='category10'))
).properties(
    width=800,
    height=400,
    title='Stacked Area Chart - Value1 by Category'
)
chart1.save('output/5_altair_area.html')
print("  ✓ Graph 1: Area chart saved to output/5_altair_area.html")

# Graph 2: Scatter plot with regression line
scatter = alt.Chart(df).mark_circle(size=60, opacity=0.6).encode(
    x=alt.X('value1:Q', title='Value 1'),
    y=alt.Y('value3:Q', title='Value 3'),
    color=alt.Color('category:N', scale=alt.Scale(scheme='set2')),
    size=alt.Size('value4:Q', title='Value 4')
)
regression = alt.Chart(df).mark_line(color='red', strokeDash=[5, 5]).transform_regression(
    'value1', 'value3', method='linear'
).encode(
    x='value1:Q',
    y='value3:Q'
)
chart2 = (scatter + regression).properties(
    width=800,
    height=400,
    title='Scatter Plot with Regression Line - Value1 vs Value3'
)
chart2.save('output/5_altair_scatter.html')
print("  ✓ Graph 2: Scatter plot saved to output/5_altair_scatter.html")


# 6. PYGAL

print("\n6. PYGAL")

# Graph 1: Line chart
line_chart = pygal.Line(style=pygal.style.DarkStyle, width=1000, height=600)
line_chart.title = 'Time Series Line Chart - Multiple Values'
line_chart.x_labels = [str(d.date()) for d in df['date'][::5]]
line_chart.add('Value 1', df['value1'].tolist())
line_chart.add('Value 2', df['value2'].tolist())
line_chart.add('Value 3', df['value3'].tolist())
line_chart.render_to_file('output/6_pygal_line.svg')
print("  ✓ Graph 1: Line chart saved to output/6_pygal_line.svg")

# Graph 2: Radar chart
radar_chart = pygal.Radar(style=pygal.style.DarkStyle, width=800, height=600)
radar_chart.title = 'Radar Chart - Category Comparison'
categories_agg = df.groupby('category').agg({
    'value1': 'mean',
    'value2': 'mean',
    'value3': 'mean',
    'value4': 'mean'
}).reset_index()
for idx, row in categories_agg.iterrows():
    radar_chart.add(row['category'], [
        row['value1'], row['value2'], row['value3'], row['value4']
    ])
radar_chart.x_labels = ['Value 1', 'Value 2', 'Value 3', 'Value 4']
radar_chart.render_to_file('output/6_pygal_radar.svg')
print("  ✓ Graph 2: Radar chart saved to output/6_pygal_radar.svg")


# 7. GGPLOT (using plotnine)

print("\n7. GGPLOT (plotnine)")

# Graph 1: Line plot (plotnine)
p1 = (ggplot(df, aes(x='date', y='value1', color='category')) +
      geom_line(size=1.5) +
      ggtitle('Line Plot - Value1 over Time by Category') +
      xlab('Date') +
      ylab('Value 1') +
      theme_bw())
# Use plotnine.savefig via ggsave
p1.save(filename='output/7_ggplot_line.png', width=10, height=6, dpi=300)
print("  ✓ Graph 1: Line plot saved to output/7_ggplot_line.png")

# Graph 2: Scatter plot with smooth (plotnine)
p2 = (ggplot(df, aes(x='value1', y='value3', color='category')) +
      geom_point(size=3, alpha=0.6) +
      geom_smooth(method='lm', se=True) +
      ggtitle('Scatter Plot with Regression Line - Value1 vs Value3') +
      xlab('Value 1') +
      ylab('Value 3') +
      theme_bw())
p2.save(filename='output/7_ggplot_scatter.png', width=10, height=6, dpi=300)
print("  ✓ Graph 2: Scatter plot saved to output/7_ggplot_scatter.png")


print("\n8. GEOPANDAS + FOLIUM")

# Create sample geographic data
np.random.seed(42)
cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 
          'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose']
lats = [40.7128, 34.0522, 41.8781, 29.7604, 33.4484, 
        39.9526, 29.4241, 32.7157, 32.7767, 37.3382]
lons = [-74.0060, -118.2437, -87.6298, -95.3698, -112.0740,
        -75.1652, -98.4936, -117.1611, -96.7970, -121.8863]
geo_df = pd.DataFrame({
    'city': cities,
    'latitude': lats,
    'longitude': lons,
    'population': np.random.randint(500000, 5000000, len(cities)),
    'temperature': np.random.randint(15, 35, len(cities))
})

# Create GeoDataFrame
geometry = [Point(xy) for xy in zip(geo_df['longitude'], geo_df['latitude'])]
gdf = gpd.GeoDataFrame(geo_df, geometry=geometry, crs='EPSG:4326')

# Graph 1: Folium Choropleth-style map with markers
m1 = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
for idx, row in geo_df.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=row['population']/100000,
        popup=f"{row['city']}<br>Population: {row['population']:,}<br>Temp: {row['temperature']}°C",
        color='blue',
        fill=True,
        fillColor='red',
        fillOpacity=0.6
    ).add_to(m1)
m1.save('output/8_folium_markers.html')
print("  ✓ Graph 1: Marker map saved to output/8_folium_markers.html")

# Graph 2: Folium Heatmap
from folium.plugins import HeatMap
m2 = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
heat_data = [[row['latitude'], row['longitude'], row['temperature']] 
             for idx, row in geo_df.iterrows()]
HeatMap(heat_data, radius=15, blur=10, max_zoom=1).add_to(m2)
for idx, row in geo_df.iterrows():
    folium.Marker(
        [row['latitude'], row['longitude']],
        popup=f"{row['city']}: {row['temperature']}°C"
    ).add_to(m2)
m2.save('output/8_folium_heatmap.html')
print("  ✓ Graph 2: Heatmap saved to output/8_folium_heatmap.html")


print("\n9. NETWORKX")

# Create sample network data
G1 = nx.Graph()
nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
G1.add_nodes_from(nodes)
edges = [('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'E'), 
         ('D', 'F'), ('E', 'F'), ('F', 'G'), ('G', 'H'), ('A', 'H')]
G1.add_edges_from(edges)


fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.spring_layout(G1, k=1, iterations=50)
nx.draw_networkx_nodes(G1, pos, node_color='lightblue', 
                       node_size=1000, alpha=0.9, ax=ax)
nx.draw_networkx_edges(G1, pos, width=2, alpha=0.5, edge_color='gray', ax=ax)
nx.draw_networkx_labels(G1, pos, font_size=12, font_weight='bold', ax=ax)
ax.set_title('Network Graph - Spring Layout', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('output/9_networkx_spring.png', dpi=300, bbox_inches='tight')
print("  ✓ Graph 1: Spring layout network saved to output/9_networkx_spring.png")
plt.close()


fig, ax = plt.subplots(figsize=(10, 8))
pos = nx.circular_layout(G1)
nx.draw_networkx_nodes(G1, pos, node_color='coral', 
                       node_size=1200, alpha=0.8, ax=ax)
nx.draw_networkx_edges(G1, pos, width=2.5, alpha=0.6, 
                       edge_color='darkblue', style='dashed', ax=ax)
nx.draw_networkx_labels(G1, pos, font_size=14, font_weight='bold', 
                        font_color='white', ax=ax)
ax.set_title('Network Graph - Circular Layout', fontsize=14, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig('output/9_networkx_circular.png', dpi=300, bbox_inches='tight')
print("  ✓ Graph 2: Circular layout network saved to output/9_networkx_circular.png")
plt.close()


print("\n10. PLOTLY (graph_objects)")

# Graph 1: 3D Surface plot
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
fig.update_layout(
    title='3D Surface Plot - Sinusoidal Function',
    scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ),
    width=900,
    height=700
)
fig.write_html('output/10_plotly_surface.html')
print("  ✓ Graph 1: 3D Surface plot saved to output/10_plotly_surface.html")


dates_ohlc = pd.date_range(start='2025-01-01', periods=20, freq='D')
np.random.seed(42)
ohlc_data = pd.DataFrame({
    'date': dates_ohlc,
    'open': 100 + np.random.randn(20).cumsum(),
    'high': 100 + np.random.randn(20).cumsum() + np.abs(np.random.randn(20) * 2),
    'low': 100 + np.random.randn(20).cumsum() - np.abs(np.random.randn(20) * 2),
    'close': 100 + np.random.randn(20).cumsum() + np.random.randn(20) * 0.5
})
fig = go.Figure(data=[go.Candlestick(
    x=ohlc_data['date'],
    open=ohlc_data['open'],
    high=ohlc_data['high'],
    low=ohlc_data['low'],
    close=ohlc_data['close']
)])
fig.update_layout(
    title='Candlestick Chart - OHLC Data',
    xaxis_title='Date',
    yaxis_title='Price',
    width=1000,
    height=600
)
fig.write_html('output/10_plotly_candlestick.html')
print("  ✓ Graph 2: Candlestick chart saved to output/10_plotly_candlestick.html")


print("\n" + "=" * 80)
print("ASSIGNMENT COMPLETE!")
print("=" * 80)
print("\nAll visualizations have been generated:")
print("  • 10 libraries used")
print("  • 20 graphs created (2 per library)")
print("  • No pie or bar charts used")
print("\nOutput files saved in the 'output' directory:")
print("  • PNG files: matplotlib, seaborn, ggplot, networkx")
print("  • HTML files: plotly.express, bokeh, altair, folium, plotly.graph_objects")
print("  • SVG files: pygal")
print("\nTo view HTML files, open them in a web browser.")
print("=" * 80)



