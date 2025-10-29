"""
Flask API for churn prediction model.
Provides REST endpoints for making predictions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Global variables for model and preprocessor
model = None
preprocessor = None


def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor
    
    model_path = os.environ.get('MODEL_PATH', '../models/churn_model.pkl')
    preprocessor_path = os.environ.get('PREPROCESSOR_PATH', '../data/processed/')
    
    try:
        print(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
        print("Model loaded successfully!")
        
        # Try to load preprocessor
        try:
            from src.data_preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor(preprocessor_path)
            print("Preprocessor loaded successfully!")
        except Exception as e:
            print(f"Warning: Could not load preprocessor: {e}")
            preprocessor = None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint."""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        # Get data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided'
            }), 400
        
        # Handle single prediction
        if 'features' in data:
            features = data['features']
            df = pd.DataFrame([features])
        # Handle batch predictions
        elif 'customers' in data:
            df = pd.DataFrame(data['customers'])
        else:
            return jsonify({
                'error': 'Invalid data format. Expected "features" or "customers"'
            }), 400
        
        # Preprocess if preprocessor available
        if preprocessor:
            X, _ = preprocessor.preprocess(df, target_col='Churn', fit=False)
        else:
            X = df
        
        # Make predictions
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)[:, 1]
        
        # Format response
        if 'features' in data:
            # Single prediction
            result = {
                'churn_prediction': int(predictions[0]),
                'churn_probability': float(probabilities[0]),
                'churn_risk': get_risk_level(probabilities[0])
            }
        else:
            # Batch predictions
            result = {
                'predictions': [
                    {
                        'churn_prediction': int(pred),
                        'churn_probability': float(prob),
                        'churn_risk': get_risk_level(prob)
                    }
                    for pred, prob in zip(predictions, probabilities)
                ]
            }
        
        return jsonify(result), 200
        
    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': 'An error occurred while making the prediction. Please check your input and try again.'
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information."""
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded'
            }), 500
        
        info = {
            'model_type': type(model).__name__,
            'features_count': None
        }
        
        # Get feature count if available
        if hasattr(model, 'n_features_in_'):
            info['features_count'] = int(model.n_features_in_)
        
        return jsonify(info), 200
        
    except Exception as e:
        # Log the full error for debugging
        app.logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': 'Unable to retrieve model information'
        }), 500


def get_risk_level(probability):
    """Categorize churn risk level."""
    if probability < 0.3:
        return 'Low'
    elif probability < 0.6:
        return 'Medium'
    else:
        return 'High'


if __name__ == '__main__':
    # Load model and preprocessor
    load_model_and_preprocessor()
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
