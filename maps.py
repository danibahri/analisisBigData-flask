import pandas as pd
from sklearn.cluster import KMeans
import folium
from folium.plugins import MarkerCluster

# 1. Data Dummy dengan Koordinat
data = {
    'latitude': [1.5, 1.7, 1.8, 2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
    'longitude': [101.5, 101.7, 101.8, 102.0, 102.1, 102.2, 103.0, 103.1, 103.2]
}
df = pd.DataFrame(data)

# 2. Clustering dengan K-Means
kmeans = KMeans(n_clusters=3, random_state=0).fit(df[['latitude', 'longitude']])
df['cluster'] = kmeans.labels_

# 3. Buat Peta Dasar
map = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=10)

# 4. Tambahkan Marker dengan MarkerCluster
marker_cluster = MarkerCluster().add_to(map)
for _, row in df.iterrows():
    folium.Marker(
        location=[row['latitude'], row['longitude']],
        popup=f"Cluster: {row['cluster']}"
    ).add_to(marker_cluster)

# 5. Tampilkan Peta
map.save("map.html")
map
