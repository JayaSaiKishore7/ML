"""
Model training module for churn prediction.
Trains multiple ML models and saves the best performing one.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import argparse
import os
import joblib
import json
from datetime import datetime


class ChurnModelTrainer:
    """Trains and manages churn prediction models."""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def load_data(self, filepath):
        """Load training data."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Separate features and target
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        
        print(f"Loaded {len(df)} samples with {X.shape[1]} features")
        print(f"Class distribution: {dict(y.value_counts())}")
        
        return X, y
    
    def initialize_models(self):
        """Initialize different ML models."""
        print("Initializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                class_weight='balanced'
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=1
            )
        }
        
        print(f"Initialized {len(self.models)} models")
        
    def train_model(self, model, X_train, y_train):
        """Train a single model."""
        print(f"Training {type(model).__name__}...")
        model.fit(X_train, y_train)
        return model
    
    def evaluate_model(self, model, X, y):
        """Evaluate model performance."""
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1_score': f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y, y_pred_proba)
        }
        
        return metrics
    
    def train_all_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models and track performance."""
        self.initialize_models()
        results = {}
        
        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training: {name}")
            print(f"{'='*50}")
            
            # Train model
            trained_model = self.train_model(model, X_train, y_train)
            
            # Evaluate on training set
            train_metrics = self.evaluate_model(trained_model, X_train, y_train)
            print(f"\nTraining Metrics:")
            for metric_name, value in train_metrics.items():
                print(f"  {metric_name}: {value:.4f}")
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate_model(trained_model, X_val, y_val)
                print(f"\nValidation Metrics:")
                for metric_name, value in val_metrics.items():
                    print(f"  {metric_name}: {value:.4f}")
                
                # Track best model based on validation F1 score
                if val_metrics['f1_score'] > self.best_score:
                    self.best_score = val_metrics['f1_score']
                    self.best_model = trained_model
                    self.best_model_name = name
                
                results[name] = {
                    'train_metrics': train_metrics,
                    'val_metrics': val_metrics
                }
            else:
                # Track best model based on training F1 score
                if train_metrics['f1_score'] > self.best_score:
                    self.best_score = train_metrics['f1_score']
                    self.best_model = trained_model
                    self.best_model_name = name
                
                results[name] = {
                    'train_metrics': train_metrics
                }
        
        print(f"\n{'='*50}")
        print(f"Best Model: {self.best_model_name}")
        print(f"Best F1 Score: {self.best_score:.4f}")
        print(f"{'='*50}")
        
        return results
    
    def save_model(self, output_path, metadata=None):
        """Save the best model and metadata."""
        os.makedirs(output_path, exist_ok=True)
        
        # Save model
        model_path = os.path.join(output_path, 'churn_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Save metadata
        if metadata:
            metadata_path = os.path.join(output_path, 'model_metadata.json')
            metadata['best_model_name'] = self.best_model_name
            metadata['best_score'] = self.best_score
            metadata['timestamp'] = datetime.now().isoformat()
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path):
        """Load a trained model."""
        self.best_model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")


def main():
    """Main function to run training from command line."""
    parser = argparse.ArgumentParser(description='Train churn prediction model')
    parser.add_argument('--data', type=str, required=True, help='Training data CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory for model')
    parser.add_argument('--val-data', type=str, help='Optional validation data CSV file path')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ChurnModelTrainer()
    
    # Load training data
    X_train, y_train = trainer.load_data(args.data)
    
    # Load validation data if provided
    X_val, y_val = None, None
    if args.val_data:
        X_val, y_val = trainer.load_data(args.val_data)
    
    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_val, y_val)
    
    # Save best model
    trainer.save_model(args.output, metadata={'results': results})
    
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
