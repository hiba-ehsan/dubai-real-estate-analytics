import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import folium
from folium.plugins import MarkerCluster


sns.set(style="whitegrid", palette="Set2")
plt.style.use('ggplot')

#Load & Process
df1 = pd.read_csv("C:/Codes/bayut_selling_properties.csv")
df1 = df1.dropna(subset=['price', 'area_name', 'Latitude', 'Longitude'])
df1['price'] = df1['price'].astype(str).str.replace('AED', '', regex=False).str.replace(',', '', regex=False).astype(float)
df1['area_sqft'] = df1['area_name'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
df1['price_per_sqft'] = df1['price'] / df1['area_sqft']


print(df1[['price', 'area_sqft', 'price_per_sqft']].describe())

#Prices 
plt.figure(figsize=(10, 5))
sns.histplot(df1['price'], bins=100, kde=True, color='mediumvioletred')
plt.title('Distribution of Property Prices', fontsize=16, color='navy')
plt.xlabel('Price (AED)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xlim(0, df1['price'].quantile(0.95))
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# By City 
plt.figure(figsize=(12, 6))
sns.boxplot(x='city', y='price', data=df1, palette='rainbow')
plt.title('Price Distribution by City', fontsize=16, color='navy')
plt.yscale('log')
plt.ylabel('Price (Log Scale)', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

#By Property Type 
plt.figure(figsize=(12, 6))
sns.boxplot(x='type', y='price', data=df1, palette='coolwarm')
plt.title('Price by Property Type', fontsize=16, color='navy')
plt.xticks(rotation=45, fontsize=10)
plt.yscale('log')
plt.grid(True, linestyle='-.', alpha=0.5)
plt.tight_layout()
plt.show()

#Predict Price From area_sqft
df1.dropna(subset=['area_sqft', 'price'], inplace=True)
X = df1[['area_sqft']]
Y = df1['price']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
print(f"Linear Regression MSE: {mse:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}, Intercept: {model.intercept_:.2f}")

plt.figure(figsize=(10, 5))
plt.scatter(X_test, Y_test, color='deepskyblue', alpha=0.6, label='Actual')
plt.plot(X_test, Y_pred, color='darkorange', linewidth=2, label='Predicted')
plt.title('Linear Regression: Price vs Area', fontsize=16, color='navy')
plt.xlabel('Area (sqft)', fontsize=12)
plt.ylabel('Price (AED)', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("regression_plot.png", dpi=300)
plt.show()

#Map Visualization
dubai_df = df1[df1['city'].str.lower() == 'dubai']
dubai_map = folium.Map(location=[25.2048, 55.2708], zoom_start=11, tiles='cartodbpositron')
marker_cluster = MarkerCluster().add_to(dubai_map)

for _, row in dubai_df.iterrows():
    folium.Marker(
        location=[row['Latitude'], row['Longitude']],
        popup=f"{row['type']} - AED {row['price']:,.0f}",
        tooltip=row['area_name'],
        icon=folium.Icon(color='purple', icon='home', prefix='fa')
    ).add_to(marker_cluster)

dubai_map.save("dubai_properties_map.html")

