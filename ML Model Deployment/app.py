from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import pickle


# Inisialisasi app
app = Flask(__name__)

# Load model dan preprocessor
with open('trained_model_regression.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_encoders_regression.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Categorical columns & numeric
categorical_cols = ['brand', 'model', 'title_status', 'color', 'state', 'condition']
numeric_cols = ['year', 'mileage']


# class CustomJSONProvider(DefaultJSONProvider):
#     def default(self, obj):
#         if isinstance(obj, (np.integer, np.int64)):
#             return int(obj)
#         return super().default(obj)

# app.json = CustomJSONProvider(app)

# Route test
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API is running!",
        "example_input": {
            "brand": "ford",
            "model": "focus",
            "year": 2022,
            "mileage": 25000,
            "title_status": "clean vehicle",
            "color": "black",
            "state": "tennessee",
            "condition": "10 days left"
        }
    })

# Route prediksi
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ambil input JSON dari user
        data = request.get_json()

        # Validasi input
        required_fields = categorical_cols + numeric_cols
        missing = [col for col in required_fields if col not in data]
        if missing:
            return jsonify({"status": "error", "message": f"Missing fields: {missing}"}), 400

        # Format ke DataFrame (harap input sesuai urutan dan kolom yang dipakai)
        df = pd.DataFrame([data])  # Assumes all fields are included

        # Encode categorical
        for col in categorical_cols:
            le = encoders[col]
            df[f"{col}_encoded"] = le.transform(df[col])

        # Gabungkan fitur
        final_features = numeric_cols + [f"{col}_encoded" for col in categorical_cols]
        input_data = df[final_features]

        # Prediksi
        prediction = model.predict(input_data)

        return jsonify({"status": "ok", "prediction": float(prediction[0])})
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
