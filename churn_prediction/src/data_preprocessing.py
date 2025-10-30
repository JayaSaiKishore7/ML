"""
Data preprocessing module for churn prediction.
Handles data cleaning, feature engineering, and train-test split.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import argparse
import os
import joblib


class DataPreprocessor:
    """Preprocesses data for churn prediction model."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self, filepath):
        """Load data from CSV file."""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        print("Handling missing values...")
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                
        print(f"Missing values after handling: {df.isnull().sum().sum()}")
        return df
    
    def encode_categorical_features(self, df, fit=True):
        """Encode categorical features using label encoding."""
        print("Encoding categorical features...")
        
        categorical_cols = df.select_dtypes(include=['object']).columns
        categorical_cols = [col for col in categorical_cols if col != 'Churn']
        
        for col in categorical_cols:
            if fit:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    le = self.label_encoders[col]
                    df[col] = le.transform(df[col].astype(str))
                    
        return df
    
    def create_features(self, df):
        """Create additional features for better prediction."""
        print("Creating additional features...")
        
        # Example feature engineering (adapt based on actual data)
        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
            df['tenure_monthly_ratio'] = df['tenure'] / (df['MonthlyCharges'] + 1)
            
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
            df['avg_monthly_charges'] = df['TotalCharges'] / (df['tenure'] + 1)
            
        return df
    
    def scale_features(self, df, target_col='Churn', fit=True):
        """Scale numerical features."""
        print("Scaling features...")
        
        feature_cols = [col for col in df.columns if col != target_col]
        
        if fit:
            df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        else:
            df[feature_cols] = self.scaler.transform(df[feature_cols])
            
        return df
    
    def preprocess(self, df, target_col='Churn', fit=True):
        """Complete preprocessing pipeline."""
        df = df.copy()
        
        # Separate target if exists
        if target_col in df.columns:
            y = df[target_col]
            X = df.drop(columns=[target_col])
        else:
            y = None
            X = df
        
        # Preprocess features
        X = self.handle_missing_values(X)
        X = self.encode_categorical_features(X, fit=fit)
        X = self.create_features(X)
        X = self.scale_features(X, target_col=target_col, fit=fit)
        
        # Encode target if exists
        if y is not None:
            if fit:
                le_target = LabelEncoder()
                y = le_target.fit_transform(y)
                self.label_encoders['target'] = le_target
            else:
                if 'target' in self.label_encoders:
                    y = self.label_encoders['target'].transform(y)
            
            return X, y
        else:
            return X, None
    
    def save_preprocessor(self, output_path):
        """Save preprocessor objects."""
        os.makedirs(output_path, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(output_path, 'scaler.pkl'))
        joblib.dump(self.label_encoders, os.path.join(output_path, 'label_encoders.pkl'))
        print(f"Preprocessor saved to {output_path}")
    
    def load_preprocessor(self, input_path):
        """Load preprocessor objects."""
        self.scaler = joblib.load(os.path.join(input_path, 'scaler.pkl'))
        self.label_encoders = joblib.load(os.path.join(input_path, 'label_encoders.pkl'))
        print(f"Preprocessor loaded from {input_path}")


def main():
    """Main function to run preprocessing from command line."""
    parser = argparse.ArgumentParser(description='Preprocess churn data')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random-state', type=int, default=42, help='Random state (default: 42)')
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Load and preprocess data
    df = preprocessor.load_data(args.input)
    X, y = preprocessor.preprocess(df, target_col='Churn', fit=True)
    
    # Split data
    print(f"Splitting data (test_size={args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )
    
    # Save processed data
    os.makedirs(args.output, exist_ok=True)
    
    train_df = pd.DataFrame(X_train, columns=X.columns)
    train_df['Churn'] = y_train
    train_df.to_csv(os.path.join(args.output, 'train.csv'), index=False)
    
    test_df = pd.DataFrame(X_test, columns=X.columns)
    test_df['Churn'] = y_test
    test_df.to_csv(os.path.join(args.output, 'test.csv'), index=False)
    
    # Save preprocessor
    preprocessor.save_preprocessor(args.output)
    
    print(f"\nPreprocessing complete!")
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    print(f"Files saved to {args.output}")


if __name__ == '__main__':
    main()
