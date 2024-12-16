import os
import cv2
import matplotlib
matplotlib.use('Agg')  # Menggunakan backend non-GUI
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from modul.preprocess_image import preprocess_image
import folium
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import load_breast_cancer



app = Flask(__name__)

# ----- FORECASTING -----
GRAPH_FOLDER = 'static/graphs'
if not os.path.exists(GRAPH_FOLDER):
    os.makedirs(GRAPH_FOLDER)
data = pd.read_csv('model/breast-cancer-cases-rate-per-100000-population-in-england-1995-2021.csv')
# Preprocess Data
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)
# Build ARIMA model
p, d, q = 1, 1, 1
model = ARIMA(data, order=(p, d, q))
model_fit = model.fit()
# Forecast for the next 10 years
steps = 6
forecast = model_fit.forecast(steps=steps)
# Prepare Forecast DataFrame
future_dates = pd.date_range(data.index[-1], periods=steps+1, freq='Y')[1:]  # Generate future years
forecast_df = pd.DataFrame({'Year': future_dates, 'Forecasted Cases': forecast})
# Round the forecasted values to 2 decimal places
forecast_df['Forecasted Cases'] = forecast_df['Forecasted Cases'].round(1)
forecast_df.set_index('Year', inplace=True)
# Save Plot
def save_plot():
    plt.figure(figsize=(10,6))
    plt.plot(data.index, data['Cases'], label='Original Data')
    plt.title('Breast Cancer Cases per 100,000 Population (1995-2021)')
    plt.xlabel('Year')
    plt.ylabel('Cases per 100,000 Population')
    plt.legend()
    plot_path = os.path.join(GRAPH_FOLDER, 'breast_cancer_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

# ----- KLASIFIKASI IMAGE -----
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

# ----- KLASIFIKASI DECISION TREE -----
data_dcs = load_breast_cancer()
dcs_df = pd.DataFrame(data_dcs.data, columns=data_dcs.feature_names)
dcs_df['target'] = data_dcs.target

mean_features = [col for col in dcs_df.columns if "mean" in col]
X_kls = dcs_df[mean_features]
y_kls = dcs_df['target']

label_encoder = LabelEncoder()
y_kls = label_encoder.fit_transform(y_kls)

X_train, X_test, y_train, y_test = train_test_split(X_kls, y_kls, test_size=0.2, random_state=42)

model_dcs = DecisionTreeClassifier(criterion="entropy", random_state=42)
model_dcs.fit(X_train, y_train)


# ----- KLASTERING MAPS -----
# Load data
clustering_data = pd.read_csv('model/modified_data.csv')
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
    # Forecasting data
    plot_path = save_plot()
    historical_data = data.tail(10)  

    # Klastering data
    m = folium.Map(location=[20.0, 100.0], zoom_start=4)
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
    
    return render_template('index.html', map_html=map_html,historical_data=historical_data.to_dict(),forecast_data=forecast_df.to_dict(),plot_url=plot_path, features=mean_features)

# ----- Route untuk Forecasting -----
@app.route('/forecast', methods=['GET'])
def forecast():
    # Simpan grafik ke dalam folder static
    plot_path = save_plot()
    historical_data = data.tail(10)  # Last 10 rows of data for historical display
    return render_template(
        'layout/forecast.html',
        historical_data=historical_data.to_dict(),
        forecast_data=forecast_df.to_dict(),
        plot_url=plot_path 
    )

# ----- API untuk Forecasting -----
@app.route('/api/forecast/')
def api_forecast():
    forecast_json = {
        "dates": future_dates.strftime('%Y-%m-%d').tolist(),
        "forecasted_cases": forecast.tolist()
    }
    return jsonify(forecast_json)

# ----- Route untuk mengakses gambar dari folder static -----
@app.route('/static/graphs/<filename>/')
def send_plot(filename):
    return send_from_directory(GRAPH_FOLDER, filename)

# ----- Route untuk Prediksi Image -----
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
    
# ----- Route untuk Prediksi Decision Tree -----
@app.route('/predict-dcs', methods=['POST'])
def predict_dcs():
    try:
        # Ambil data dari form
        inputs = [float(request.form[feature]) for feature in mean_features]
        inputs = pd.DataFrame([inputs], columns=mean_features)

        # Prediksi
        prediction = model_dcs.predict(inputs)[0]
        result = label_encoder.inverse_transform([prediction])[0]
        
        # Mengonversi hasil prediksi dan pesan menjadi tipe yang bisa diserialisasi
        result = str(result)  # pastikan result adalah string
        message = f"Prediksi: {result} (0=Malignant, 1=Benign)"
        
        # Kirim hasil sebagai response JSON
        return jsonify({
            'success': True,
            'prediction': result,
            'message': message
        })
        
    except Exception as e:
        # Tangani error dan kirimkan response JSON dengan error
        return jsonify({
            'success': False,
            'error': f"Terjadi kesalahan: {str(e)}"
        })
    
if __name__ == '__main__':
    app.run(debug=True)