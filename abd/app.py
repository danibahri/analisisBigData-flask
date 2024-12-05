import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from flask import Flask, request, render_template

# Flask app
app = Flask(__name__)

# Load dataset
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Pilih kolom "mean" saja
mean_features = [col for col in df.columns if "mean" in col]
X = df[mean_features]
y = df['target']

# Encode target labels (jika belum)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (Decision Tree with entropy)
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)



@app.route("/")
def home():
    return render_template("index.html", features=mean_features)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil data dari form
        inputs = [float(request.form[feature]) for feature in mean_features]
        inputs = pd.DataFrame([inputs], columns=mean_features)

        # Prediksi
        prediction = model.predict(inputs)[0]
        result = label_encoder.inverse_transform([prediction])[0]
        return render_template("index.html", result=f"Prediksi: {result} (0=Malignant, 1=Benign)")
    except Exception as e:
        return render_template("index.html", result=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)