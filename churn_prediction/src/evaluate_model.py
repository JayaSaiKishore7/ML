"""
Model evaluation module for churn prediction.
Provides comprehensive evaluation metrics and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import argparse
import os
import joblib


class ChurnModelEvaluator:
    """Evaluates churn prediction model performance."""
    
    def __init__(self, model_path):
        """Initialize evaluator with trained model."""
        print(f"Loading model from {model_path}...")
        self.model = joblib.load(model_path)
        
    def load_data(self, filepath):
        """Load test data."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Separate features and target
        X = df.drop(columns=['Churn'])
        y = df['Churn']
        
        print(f"Loaded {len(df)} samples with {X.shape[1]} features")
        
        return X, y
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics."""
        print("\nCalculating metrics...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics."""
        print("\n" + "="*50)
        print("Model Evaluation Metrics")
        print("="*50)
        for metric_name, value in metrics.items():
            print(f"{metric_name.capitalize():20s}: {value:.4f}")
        print("="*50)
    
    def print_classification_report(self, y_true, y_pred):
        """Print detailed classification report."""
        print("\n" + "="*50)
        print("Classification Report")
        print("="*50)
        print(classification_report(y_true, y_pred, target_names=['Not Churned', 'Churned']))
    
    def plot_confusion_matrix(self, y_true, y_pred, output_path=None):
        """Plot confusion matrix."""
        print("\nGenerating confusion matrix...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Not Churned', 'Churned'],
                    yticklabels=['Not Churned', 'Churned'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'confusion_matrix.png'), 
                       bbox_inches='tight', dpi=300)
            print(f"Confusion matrix saved to {output_path}/confusion_matrix.png")
        
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, output_path=None):
        """Plot ROC curve."""
        print("\nGenerating ROC curve...")
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'roc_curve.png'), 
                       bbox_inches='tight', dpi=300)
            print(f"ROC curve saved to {output_path}/roc_curve.png")
        
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, output_path=None):
        """Plot precision-recall curve."""
        print("\nGenerating precision-recall curve...")
        
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.grid(alpha=0.3)
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'precision_recall_curve.png'), 
                       bbox_inches='tight', dpi=300)
            print(f"Precision-recall curve saved to {output_path}/precision_recall_curve.png")
        
        plt.close()
    
    def plot_feature_importance(self, feature_names, output_path=None, top_n=20):
        """Plot feature importance if available."""
        print("\nGenerating feature importance plot...")
        
        # Check if model has feature_importances_
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importances = np.abs(self.model.coef_[0])
        else:
            print("Model does not support feature importance.")
            return
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['importance'])
        plt.yticks(range(len(importance_df)), importance_df['feature'])
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if output_path:
            plt.savefig(os.path.join(output_path, 'feature_importance.png'), 
                       bbox_inches='tight', dpi=300)
            print(f"Feature importance saved to {output_path}/feature_importance.png")
        
        plt.close()
    
    def evaluate(self, X_test, y_test, output_path=None):
        """Complete evaluation pipeline."""
        print("\nEvaluating model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Print results
        self.print_metrics(metrics)
        self.print_classification_report(y_test, y_pred)
        
        # Generate plots if output path provided
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            self.plot_confusion_matrix(y_test, y_pred, output_path)
            self.plot_roc_curve(y_test, y_pred_proba, output_path)
            self.plot_precision_recall_curve(y_test, y_pred_proba, output_path)
            self.plot_feature_importance(X_test.columns, output_path)
        
        return metrics


def main():
    """Main function to run evaluation from command line."""
    parser = argparse.ArgumentParser(description='Evaluate churn prediction model')
    parser.add_argument('--model', type=str, required=True, help='Trained model file path')
    parser.add_argument('--data', type=str, required=True, help='Test data CSV file path')
    parser.add_argument('--output', type=str, help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ChurnModelEvaluator(args.model)
    
    # Load test data
    X_test, y_test = evaluator.load_data(args.data)
    
    # Evaluate model
    metrics = evaluator.evaluate(X_test, y_test, args.output)
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
