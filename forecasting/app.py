import os
import matplotlib.pyplot as plt
from flask import Flask, render_template, jsonify, send_from_directory
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import numpy as np

app = Flask(__name__)

# Tentukan direktori untuk menyimpan gambar
GRAPH_FOLDER = 'static/graphs'
if not os.path.exists(GRAPH_FOLDER):
    os.makedirs(GRAPH_FOLDER)

# Load dataset
data = pd.read_csv('data/breast-cancer-cases-rate-per-100000-population-in-england-1995-2021.csv')

# Preprocess Data
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)

# Build ARIMA model
p, d, q = 1, 1, 1
model = ARIMA(data, order=(p, d, q))
model_fit = model.fit()

# Forecast for the next 10 years
steps = 10
forecast = model_fit.forecast(steps=steps)

# Prepare Forecast DataFrame
future_dates = pd.date_range(data.index[-1], periods=steps+1, freq='Y')[1:]  # Generate future years
forecast_df = pd.DataFrame({'Year': future_dates, 'Forecasted Cases': forecast})
# Bulatkan hasil prediksi hingga 2 desimal
forecast_df['Forecasted Cases'] = forecast_df['Forecasted Cases'].round(1)

forecast_df.set_index('Year', inplace=True)

# Menyimpan Grafik
def save_plot():
    plt.figure(figsize=(10,6))
    plt.plot(data.index, data['Cases'], label='Data Asli')
    plt.title('Kasus Kanker Payudara per 100.000 Populasi (1995-2021)')
    plt.xlabel('Tahun')
    plt.ylabel('Kasus per 100.000 Populasi')
    plt.legend()
    plot_path = os.path.join(GRAPH_FOLDER, 'breast_cancer_plot.png')
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@app.route('/')
def index():
    # Simpan grafik ke dalam folder static
    plot_path = save_plot()

    historical_data = data.tail(10)  # Last 10 rows of data for historical display

    return render_template(
        'index.html',
        historical_data=historical_data.to_dict(),
        forecast_data=forecast_df.to_dict(),
        plot_url=plot_path  # Kirimkan URL gambar
    )

@app.route('/api/forecast')
def api_forecast():
    # Return forecast data as JSON
    forecast_json = {
        "dates": future_dates.strftime('%Y-%m-%d').tolist(),
        "forecasted_cases": forecast.tolist()
    }
    return jsonify(forecast_json)

# Route untuk mengakses gambar dari folder static
@app.route('/static/graphs/<filename>')
def send_plot(filename):
    return send_from_directory(GRAPH_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)
