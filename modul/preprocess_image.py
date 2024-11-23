import cv2
import numpy as np


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