import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
from modul.preprocess_image import preprocess_image
import folium
import pandas as pd


app = Flask(__name__)

# ----- CNN -----
# Pastikan path model benar
MODEL_PATH = os.path.join('model', 'CNN_model.keras')
# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
    print("Model Summary:")
    model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")

# ----- MAPS -----
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

@app.route('/', methods=['GET'])
def index():
    # Create map
    m = folium.Map(location=[20.0, 100.0], zoom_start=4)

    # Group data by country and calculate cluster counts
    cluster_counts = clustering_data.groupby(['Country', 'differentiate']).size().unstack(fill_value=0)

    for country, rows in clustering_data.groupby('Country'):
        coords = rows.iloc[0]['Coordinates']
        
        low_risk = cluster_counts.loc[country, 0] if 0 in cluster_counts.columns else 0
        medium_risk = cluster_counts.loc[country, 1] if 1 in cluster_counts.columns else 0
        high_risk = cluster_counts.loc[country, 2] if 2 in cluster_counts.columns else 0

        popup_text = f"""
        <b>Country:</b> {country}<br>
        <b>Low Risk:</b> {low_risk}<br>
        <b>Medium Risk:</b> {medium_risk}<br>
        <b>High Risk:</b> {high_risk}<br>
        """

        if high_risk > medium_risk and high_risk > low_risk:
            color = 'red'
        elif medium_risk > low_risk:
            color = 'orange'
        else:
            color = 'green'

        folium.Marker(location=coords, popup=popup_text, icon=folium.Icon(color=color)).add_to(m)

    # Convert map to HTML string
    map_html = m._repr_html_()

    # Render the template and pass map_html
    return render_template('index.html', map_html=map_html)

# ----- Route untuk Prediksi -----
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'})
    
    file = request.files['file']
    
    # Validasi file
    if file.filename == '':
        return jsonify({'error': 'Tidak ada file yang dipilih'})
    
    # Validasi tipe file
    allowed_extensions = {'png', 'jpg', 'jpeg'}
    if not file.filename.lower().endswith(tuple(allowed_extensions)):
        return jsonify({'error': 'Format file tidak didukung'})
    
    try:
        # Baca dan validasi gambar
        image = Image.open(file.stream)
        
        # Proses gambar
        processed_image, img_rgb = preprocess_image(image)
        
        # Prediksi
        prediction = model.predict(processed_image)
        
        # Ambil probabilitas
        prob_no_cancer = float(prediction[0][0]) * 100
        prob_cancer = float(prediction[0][1]) * 100
        
        # Tentukan kelas
        result_class = "Kanker" if prob_cancer > prob_no_cancer else "Normal"
        confidence = max(prob_cancer, prob_no_cancer)
        
        return jsonify({
            'success': True,
            'class': result_class,
            'confidence': round(confidence, 2),
            'prob_no_cancer': round(prob_no_cancer, 2),
            'prob_cancer': round(prob_cancer, 2)
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Terjadi kesalahan dalam pemrosesan gambar'
        })
   

app.run(debug=True)