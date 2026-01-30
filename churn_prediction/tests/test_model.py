"""
Unit tests for model training module.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from train_model import ChurnModelTrainer


class TestChurnModelTrainer:
    """Test cases for ChurnModelTrainer class."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100
        
        X = pd.DataFrame({
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.randn(n_samples),
            'feature3': np.random.randn(n_samples)
        })
        y = np.random.randint(0, 2, n_samples)
        
        return X, y
    
    def test_initialization(self):
        """Test trainer initialization."""
        trainer = ChurnModelTrainer()
        assert trainer.models == {}
        assert trainer.best_model is None
        assert trainer.best_score == 0
    
    def test_initialize_models(self):
        """Test model initialization."""
        trainer = ChurnModelTrainer()
        trainer.initialize_models()
        
        # Check that models are initialized
        assert len(trainer.models) > 0
        assert 'Logistic Regression' in trainer.models
        assert 'Random Forest' in trainer.models
        assert 'XGBoost' in trainer.models
    
    def test_train_model(self, sample_training_data):
        """Test single model training."""
        trainer = ChurnModelTrainer()
        trainer.initialize_models()
        
        X, y = sample_training_data
        model = trainer.models['Logistic Regression']
        
        trained_model = trainer.train_model(model, X, y)
        assert trained_model is not None
        
        # Test that model can make predictions
        predictions = trained_model.predict(X)
        assert len(predictions) == len(y)
    
    def test_evaluate_model(self, sample_training_data):
        """Test model evaluation."""
        trainer = ChurnModelTrainer()
        trainer.initialize_models()
        
        X, y = sample_training_data
        model = trainer.train_model(trainer.models['Logistic Regression'], X, y)
        
        metrics = trainer.evaluate_model(model, X, y)
        
        # Check that all metrics are present
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'roc_auc' in metrics
        
        # Check that metrics are in valid range
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1
    
    def test_train_all_models(self, sample_training_data):
        """Test training all models."""
        trainer = ChurnModelTrainer()
        X, y = sample_training_data
        
        results = trainer.train_all_models(X, y)
        
        # Check that results contain all models
        assert len(results) > 0
        
        # Check that best model is selected
        assert trainer.best_model is not None
        assert trainer.best_model_name is not None
        assert trainer.best_score > 0
