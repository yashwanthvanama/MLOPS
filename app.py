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

# Load model when module is imported (works with Gunicorn)
load_model()

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
        
        if isinstance(data, list):
            features = data
        elif isinstance(data, dict) and 'features' in data:
            features = data['features']
        else:
            return jsonify({
                'error': 'Invalid input format',
                'expected': 'List of 4 features or {"features": [...]}'
            }), 400
        
        if len(features) != 4:
            return jsonify({
                'error': f'Expected 4 features, got {len(features)}',
                'features_order': feature_names
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
    app.run(debug=True, host='0.0.0.0', port=5001)
