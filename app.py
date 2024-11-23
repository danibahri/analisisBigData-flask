import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
from modul.preprocess_image import preprocess_image


app = Flask(__name__)

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

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

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