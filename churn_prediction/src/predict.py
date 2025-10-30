"""
Prediction module for churn prediction.
Makes predictions on new customer data.
"""

import pandas as pd
import numpy as np
import argparse
import os
import joblib
import json


class ChurnPredictor:
    """Makes churn predictions on new data."""
    
    def __init__(self, model_path, preprocessor_path=None):
        """Initialize predictor with trained model and preprocessor."""
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
        self.preprocessor = None
        if preprocessor_path:
            print(f"Loading preprocessor from {preprocessor_path}...")
            from data_preprocessing import DataPreprocessor
            self.preprocessor = DataPreprocessor()
            self.preprocessor.load_preprocessor(preprocessor_path)
    
    def load_data(self, filepath):
        """Load data for prediction."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} samples")
        return df
    
    def preprocess_data(self, df):
        """Preprocess data if preprocessor is available."""
        if self.preprocessor:
            print("Preprocessing data...")
            X, _ = self.preprocessor.preprocess(df, target_col='Churn', fit=False)
            return X
        return df
    
    def predict(self, X):
        """Make predictions."""
        print("Making predictions...")
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        return predictions, probabilities
    
    def predict_single(self, features_dict):
        """Make prediction for a single customer."""
        df = pd.DataFrame([features_dict])
        
        if self.preprocessor:
            X, _ = self.preprocessor.preprocess(df, target_col='Churn', fit=False)
        else:
            X = df
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0, 1]
        
        return {
            'churn_prediction': int(prediction),
            'churn_probability': float(probability),
            'churn_risk': self._get_risk_level(probability)
        }
    
    def _get_risk_level(self, probability):
        """Categorize churn risk level."""
        if probability < 0.3:
            return 'Low'
        elif probability < 0.6:
            return 'Medium'
        else:
            return 'High'
    
    def save_predictions(self, df, predictions, probabilities, output_path):
        """Save predictions to file."""
        df['churn_prediction'] = predictions
        df['churn_probability'] = probabilities
        df['churn_risk'] = df['churn_probability'].apply(self._get_risk_level)
        
        df.to_csv(output_path, index=False)
        print(f"\nPredictions saved to {output_path}")
        
        # Print summary
        print("\nPrediction Summary:")
        print(f"Total customers: {len(df)}")
        print(f"Predicted churners: {predictions.sum()}")
        print(f"Predicted retention: {len(df) - predictions.sum()}")
        print(f"\nRisk Distribution:")
        print(df['churn_risk'].value_counts())


def main():
    """Main function to run predictions from command line."""
    parser = argparse.ArgumentParser(description='Make churn predictions')
    parser.add_argument('--model', type=str, required=True, help='Trained model file path')
    parser.add_argument('--input', type=str, required=True, help='Input data CSV file path')
    parser.add_argument('--output', type=str, help='Output file path for predictions')
    parser.add_argument('--preprocessor', type=str, help='Preprocessor directory path')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = ChurnPredictor(args.model, args.preprocessor)
    
    # Load data
    df = predictor.load_data(args.input)
    
    # Preprocess if preprocessor available
    if predictor.preprocessor:
        X = predictor.preprocess_data(df)
    else:
        # Remove target column if exists
        if 'Churn' in df.columns:
            X = df.drop(columns=['Churn'])
        else:
            X = df
    
    # Make predictions
    predictions, probabilities = predictor.predict(X)
    
    # Save predictions
    if args.output:
        predictor.save_predictions(df.copy(), predictions, probabilities, args.output)
    else:
        # Print first few predictions
        print("\nSample Predictions:")
        for i in range(min(5, len(predictions))):
            print(f"Customer {i+1}: Churn={predictions[i]}, "
                  f"Probability={probabilities[i]:.4f}, "
                  f"Risk={predictor._get_risk_level(probabilities[i])}")
    
    print("\nPrediction complete!")


if __name__ == '__main__':
    main()
