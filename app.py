from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

model = None
feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
class_names = ['setosa', 'versicolor', 'virginica']

def load_model():
    global model
    model_path = 'flower_classifier_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model file not found at {model_path}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        
        features = [
            data.get('sepal_length'),
            data.get('sepal_width'),
            data.get('petal_length'),
            data.get('petal_width')
        ]
        
        if None in features:
            return jsonify({
                'error': 'Missing required features',
                'required': feature_names
            }), 400
        
        input_data = np.array([features])
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        result = {
            'prediction': class_names[prediction[0]],
            'confidence': float(prediction_proba[0][prediction[0]]),
            'probabilities': {
                class_names[i]: float(prediction_proba[0][i]) 
                for i in range(len(class_names))
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    load_model()
    app.run(debug=True, host='0.0.0.0', port=5000)
