"""
Iris Flower Classification API
A Flask web application for classifying iris flowers
"""

import json
import numpy as np
import joblib
from flask import Flask, request, render_template, jsonify
import warnings
warnings.filterwarnings('ignore')

# Initialize Flask app
app = Flask(__name__)

# Load the trained model, scaler, and label encoder
print("Loading model and preprocessing objects...")
try:
    model = joblib.load('iris_classification_model_20251228_161424.pkl')
    scaler = joblib.load('scaler_20251228_161424.pkl')
    label_encoder = joblib.load('label_encoder_20251228_161424.pkl')
    print("✓ Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"✗ Error loading files: {e}")
    model = scaler = label_encoder = None

# Feature names (must match training)
FEATURE_NAMES = ['sepal length (cm)', 'sepal width (cm)', 
                 'petal length (cm)', 'petal width (cm)']

# Class names (mapping from encoded labels)
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    """Render the home page with prediction form"""
    return render_template('index.html', 
                          feature_names=FEATURE_NAMES,
                          class_names=CLASS_NAMES)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        # Get data from form or JSON
        if request.form:
            data = [float(request.form[f'feature{i}']) 
                   for i in range(len(FEATURE_NAMES))]
        elif request.json:
            data = request.json['features']
        else:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to numpy array and reshape
        features = np.array(data).reshape(1, -1)
        
        # Validate input length
        if len(data) != len(FEATURE_NAMES):
            return jsonify({
                'error': f'Expected {len(FEATURE_NAMES)} features, got {len(data)}'
            }), 400
        
        # Scale features (same as training)
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction_encoded = model.predict(features_scaled)
        prediction_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(features_scaled)[0].tolist()
        
        # Decode the prediction
        prediction_name = label_encoder.inverse_transform(prediction_encoded)[0]
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'class': prediction_name,
                'class_encoded': int(prediction_encoded[0]),
                'confidence': None
            },
            'input_features': dict(zip(FEATURE_NAMES, data))
        }
        
        # Add probabilities if available
        if prediction_proba:
            probabilities = dict(zip(CLASS_NAMES, prediction_proba))
            response['prediction']['probabilities'] = probabilities
            response['prediction']['confidence'] = max(prediction_proba)
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """REST API endpoint (alternative)"""
    return predict()

@app.route('/api/info', methods=['GET'])
def api_info():
    """Get model information"""
    info = {
        'model_type': str(type(model).__name__),
        'features': FEATURE_NAMES,
        'classes': CLASS_NAMES,
        'features_count': len(FEATURE_NAMES),
        'classes_count': len(CLASS_NAMES)
    }
    return jsonify(info)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if model and scaler and label_encoder:
        return jsonify({'status': 'healthy', 'message': 'Model is ready'})
    return jsonify({'status': 'unhealthy', 'message': 'Model not loaded'}), 503

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict multiple samples at once"""
    try:
        if request.json and 'samples' in request.json:
            samples = request.json['samples']
            
            # Validate input
            if not isinstance(samples, list):
                return jsonify({'error': 'Samples must be a list'}), 400
            
            predictions = []
            for i, sample in enumerate(samples):
                if len(sample) != len(FEATURE_NAMES):
                    return jsonify({
                        'error': f'Sample {i}: Expected {len(FEATURE_NAMES)} features, got {len(sample)}'
                    }), 400
                
                # Prepare and scale features
                features = np.array(sample).reshape(1, -1)
                features_scaled = scaler.transform(features)
                
                # Predict
                pred_encoded = model.predict(features_scaled)
                pred_name = label_encoder.inverse_transform(pred_encoded)[0]
                
                predictions.append({
                    'sample_id': i,
                    'prediction': pred_name,
                    'prediction_encoded': int(pred_encoded[0]),
                    'features': dict(zip(FEATURE_NAMES, sample))
                })
            
            return jsonify({
                'success': True,
                'predictions': predictions,
                'count': len(predictions)
            })
        else:
            return jsonify({'error': 'No samples provided'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("IRIS FLOWER CLASSIFICATION API")
    print("="*50)
    print(f"Model: {type(model).__name__}")
    print(f"Features: {', '.join(FEATURE_NAMES)}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")
    print("="*50)
    print("\nStarting server...")
    print("Web interface: http://localhost:5000")
    print("API endpoint: http://localhost:5000/predict")
    print("Health check: http://localhost:5000/api/health")
    print("\nPress Ctrl+C to stop the server")
    
    # Run the app
    app.run(host='0.0.0.0', port=5000, debug=True)