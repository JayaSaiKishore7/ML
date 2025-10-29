# Customer Churn Prediction

## Project Overview
This project implements an end-to-end machine learning solution for predicting customer churn. The model helps businesses identify customers who are likely to discontinue their services, enabling proactive retention strategies.

## Project Structure
```
churn_prediction/
├── data/               # Dataset storage
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── models/            # Saved model files
├── notebooks/         # Jupyter notebooks for exploration
├── src/               # Source code
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── predict.py
├── tests/             # Unit tests
├── deployment/        # Deployment files
│   ├── app.py        # Flask API
│   └── Dockerfile    # Docker configuration
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Features
- Data preprocessing and feature engineering
- Multiple ML model training (Logistic Regression, Random Forest, XGBoost)
- Model evaluation with comprehensive metrics
- Model serialization and versioning
- REST API for real-time predictions
- Docker containerization for easy deployment

## Setup

### Prerequisites
- Python 3.8+
- pip

### Installation
```bash
cd churn_prediction
pip install -r requirements.txt
```

## Usage

### 1. Data Preprocessing
```bash
python src/data_preprocessing.py --input data/raw/customer_data.csv --output data/processed/
```

### 2. Train Model
```bash
python src/train_model.py --data data/processed/train.csv --output models/
```

### 3. Evaluate Model
```bash
python src/evaluate_model.py --model models/churn_model.pkl --data data/processed/test.csv
```

### 4. Make Predictions
```bash
python src/predict.py --model models/churn_model.pkl --input data/new_customers.csv
```

### 5. Deploy API
```bash
cd deployment
python app.py
```

The API will be available at `http://localhost:5000`

#### API Endpoints
- `GET /health` - Health check
- `POST /predict` - Predict churn probability
  ```json
  {
    "features": {
      "tenure": 12,
      "monthly_charges": 70.5,
      "total_charges": 846.0,
      "contract_type": "Month-to-month",
      "payment_method": "Electronic check"
    }
  }
  ```

### 6. Docker Deployment
```bash
cd deployment
docker build -t churn-prediction-api .
docker run -p 5000:5000 churn-prediction-api
```

## Model Performance
The model achieves the following performance metrics on the test set:
- Accuracy: ~80%
- Precision: ~75%
- Recall: ~70%
- F1-Score: ~72%
- ROC-AUC: ~85%

## Future Enhancements
- Implement hyperparameter tuning
- Add more advanced models (Neural Networks, LightGBM)
- Create web dashboard for visualization
- Implement model monitoring and retraining pipeline
- Add CI/CD pipeline
- Deploy to cloud platforms (AWS, Azure, GCP)
