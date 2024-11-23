import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image

app = Flask(__name__)

# Pastikan path model benar
MODEL_PATH = os.path.join('CNN_model.keras')

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
    print("Model Summary:")
    model.summary()
except Exception as e:
    print(f"Error loading model: {str(e)}")

# Fungsi untuk memproses gambar
def preprocess_image(image):
    try:
        # Konversi gambar ke RGB
        image_rgb = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize gambar ke 50x50
        image_resized = cv2.resize(image_rgb, (50, 50))
        
        # Normalisasi
        image_normalized = image_resized / 255.0
        
        # Reshape untuk model
        image_processed = image_normalized.reshape(1, 50, 50, 3)
        
        return image_processed, image_rgb
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.route('/')
def home():
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

if __name__ == '__main__':
    app.run(debug=True)
