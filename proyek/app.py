from flask import Flask, render_template
import folium
import pandas as pd

app = Flask(__name__)

# Load data
clustering_data = pd.read_csv('modified_data.csv')
# Country coordinates for visualization

asia_countries = {
    'Indonesia': [-0.7893, 113.9213],
    'Malaysia': [4.2105, 101.9758],
    'Thailand': [15.8700, 100.9925],
    'Vietnam': [14.0583, 108.2772],
    'Japan': [36.2048, 138.2529],
    'China': [35.8617, 104.1954],
    'India': [20.5937, 78.9629],
    'Singapore': [1.3521, 103.8198],
    'Philippines': [12.8797, 121.7740],
    'South Korea': [35.9078, 127.7669]
}

# Add coordinates to the clustering data
clustering_data['Coordinates'] = clustering_data['Country'].map(asia_countries)

@app.route('/')
def index():
    # Create map
    m = folium.Map(location=[20.0, 100.0], zoom_start=4)

    # Group data by country and calculate cluster counts
    cluster_counts = clustering_data.groupby(['Country', 'differentiate']).size().unstack(fill_value=0)

    for country, rows in clustering_data.groupby('Country'):
        # Calculate center coordinates for the country (assuming single coordinates per country)
        coords = rows.iloc[0]['Coordinates']  # Assuming Coordinates is a tuple (lat, lon)
        
        # Get cluster counts
        low_risk = cluster_counts.loc[country, 0] if 0 in cluster_counts.columns else 0
        medium_risk = cluster_counts.loc[country, 1] if 1 in cluster_counts.columns else 0
        high_risk = cluster_counts.loc[country, 2] if 2 in cluster_counts.columns else 0

        # Create popup text
        popup_text = f"""
        <b>Country:</b> {country}<br>
        <b>Low Risk:</b> {low_risk}<br>
        <b>Medium Risk:</b> {medium_risk}<br>
        <b>High Risk:</b> {high_risk}<br>
        """

        # Assign color based on predominant cluster
        if high_risk > medium_risk and high_risk > low_risk:
            color = 'red'
        elif medium_risk > low_risk:
            color = 'orange'
        else:
            color = 'green'

        # Add marker to map
        folium.Marker(location=coords, popup=popup_text, icon=folium.Icon(color=color)).add_to(m)

    # Save map to a file
    map_file = 'templates/map.html'
    m.save(map_file)

    return render_template('map.html')

if __name__ == '__main__':
    app.run(debug=True)
