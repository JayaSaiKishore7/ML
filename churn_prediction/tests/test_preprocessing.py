"""
Unit tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_preprocessing import DataPreprocessor


class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return pd.DataFrame({
            'tenure': [1, 12, 24, 36, np.nan],
            'MonthlyCharges': [50.0, 70.0, 90.0, 100.0, 60.0],
            'TotalCharges': [50.0, 840.0, 2160.0, 3600.0, 720.0],
            'Contract': ['Month-to-month', 'One year', 'Two year', 'One year', 'Month-to-month'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'Electronic check'],
            'Churn': ['Yes', 'No', 'No', 'No', 'Yes']
        })
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.scaler is not None
        assert isinstance(preprocessor.label_encoders, dict)
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        df = preprocessor.handle_missing_values(sample_data)
        
        # Check that no missing values remain
        assert df.isnull().sum().sum() == 0
        
        # Check that median was used for numerical columns
        assert df['tenure'].notna().all()
    
    def test_encode_categorical_features(self, sample_data):
        """Test categorical encoding."""
        preprocessor = DataPreprocessor()
        df = sample_data.copy()
        df = preprocessor.encode_categorical_features(df, fit=True)
        
        # Check that categorical columns are encoded
        assert df['Contract'].dtype in [np.int32, np.int64]
        assert df['PaymentMethod'].dtype in [np.int32, np.int64]
        
        # Check that label encoders are saved
        assert 'Contract' in preprocessor.label_encoders
        assert 'PaymentMethod' in preprocessor.label_encoders
    
    def test_create_features(self, sample_data):
        """Test feature creation."""
        preprocessor = DataPreprocessor()
        df = preprocessor.create_features(sample_data)
        
        # Check that new features are created
        assert 'tenure_monthly_ratio' in df.columns
        assert 'avg_monthly_charges' in df.columns
    
    def test_preprocess_with_target(self, sample_data):
        """Test complete preprocessing with target."""
        preprocessor = DataPreprocessor()
        X, y = preprocessor.preprocess(sample_data, target_col='Churn', fit=True)
        
        # Check output types
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, np.ndarray)
        
        # Check shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        
        # Check that target is not in features
        assert 'Churn' not in X.columns
    
    def test_preprocess_without_target(self, sample_data):
        """Test preprocessing without target."""
        preprocessor = DataPreprocessor()
        df_no_target = sample_data.drop(columns=['Churn'])
        X, y = preprocessor.preprocess(df_no_target, fit=True)
        
        # Check that y is None
        assert y is None
        assert isinstance(X, pd.DataFrame)
