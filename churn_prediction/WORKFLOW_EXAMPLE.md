# Complete Workflow Example

This document demonstrates a complete end-to-end workflow for the churn prediction project.

## Step-by-Step Example

### 1. Generate Sample Data

First, generate sample customer data:

```bash
cd churn_prediction/data
python generate_sample_data.py
```

**Output:**
```
Generating 2000 sample customer records...
Sample data saved to data/raw/customer_data.csv
Total samples: 2000
Features: 17
Churn distribution:
No     1179
Yes     821
Churn rate: 41.05%
```

### 2. Data Preprocessing

Preprocess the raw data for model training:

```bash
cd churn_prediction
python src/data_preprocessing.py \
    --input data/raw/customer_data.csv \
    --output data/processed/ \
    --test-size 0.2 \
    --random-state 42
```

**Output:**
```
Loading data from data/raw/customer_data.csv...
Loaded 2000 rows and 17 columns
Handling missing values...
Encoding categorical features...
Creating additional features...
Scaling features...
Splitting data (test_size=0.2)...

Preprocessing complete!
Train set: 1600 samples
Test set: 400 samples
Files saved to data/processed/
```

**Generated Files:**
- `data/processed/train.csv` - Training data
- `data/processed/test.csv` - Test data
- `data/processed/scaler.pkl` - Feature scaler
- `data/processed/label_encoders.pkl` - Label encoders

### 3. Model Training

Train multiple ML models and save the best one:

```bash
python src/train_model.py \
    --data data/processed/train.csv \
    --val-data data/processed/test.csv \
    --output models/
```

**Output:**
```
Loading data from data/processed/train.csv...
Loaded 1600 samples with 18 features

Training: Logistic Regression
Training Metrics:
  accuracy: 0.7144
  precision: 0.6299
  recall: 0.7382
  f1_score: 0.6797
  roc_auc: 0.7970

Validation Metrics:
  accuracy: 0.6925
  precision: 0.6010
  recall: 0.7439
  f1_score: 0.6649
  roc_auc: 0.7836

Training: Random Forest
...

Training: XGBoost
...

Best Model: Logistic Regression
Best F1 Score: 0.6649

Model saved to models/churn_model.pkl
```

**Generated Files:**
- `models/churn_model.pkl` - Trained model
- `models/model_metadata.json` - Model metadata and metrics

### 4. Model Evaluation

Evaluate the model and generate visualizations:

```bash
mkdir -p results
python src/evaluate_model.py \
    --model models/churn_model.pkl \
    --data data/processed/test.csv \
    --output results/
```

**Output:**
```
Model Evaluation Metrics
==================================================
Accuracy            : 0.6925
Precision           : 0.6010
Recall              : 0.7439
F1_score            : 0.6649
Roc_auc             : 0.7836

Classification Report
              precision    recall  f1-score   support

 Not Churned       0.79      0.66      0.72       236
     Churned       0.60      0.74      0.66       164

    accuracy                           0.69       400

Confusion matrix saved to results/confusion_matrix.png
ROC curve saved to results/roc_curve.png
Precision-recall curve saved to results/precision_recall_curve.png
Feature importance saved to results/feature_importance.png
```

**Generated Files:**
- `results/confusion_matrix.png` - Confusion matrix visualization
- `results/roc_curve.png` - ROC curve
- `results/precision_recall_curve.png` - Precision-recall curve
- `results/feature_importance.png` - Feature importance plot

### 5. Making Predictions

Make predictions on new customer data:

```bash
python src/predict.py \
    --model models/churn_model.pkl \
    --preprocessor data/processed/ \
    --input data/raw/customer_data.csv \
    --output predictions.csv
```

**Output:**
```
Loading model...
Loading preprocessor...
Loading data...
Preprocessing data...
Making predictions...

Prediction Summary:
Total customers: 2000
Predicted churners: 821
Predicted retention: 1179

Risk Distribution:
Low       734
Medium    645
High      621
```

**Generated Files:**
- `predictions.csv` - Predictions with churn probability and risk level

### 6. Deploy API

Deploy the model as a REST API:

```bash
cd deployment
export MODEL_PATH=../models/churn_model.pkl
export PREPROCESSOR_PATH=../data/processed/
python app.py
```

**Output:**
```
Loading model from ../models/churn_model.pkl...
Model loaded successfully!
Loading preprocessor from ../data/processed/...
Preprocessor loaded successfully!
 * Running on http://0.0.0.0:5000
```

### 7. Test the API

Test the health endpoint:

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

Test making a prediction:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "gender": "Male",
      "SeniorCitizen": 0,
      "Partner": "Yes",
      "Dependents": "No",
      "tenure": 12,
      "PhoneService": "Yes",
      "MultipleLines": "No",
      "InternetService": "Fiber optic",
      "OnlineSecurity": "No",
      "TechSupport": "No",
      "Contract": "Month-to-month",
      "PaperlessBilling": "Yes",
      "PaymentMethod": "Electronic check",
      "MonthlyCharges": 70.5,
      "TotalCharges": 846.0
    }
  }'
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.7234,
  "churn_risk": "High"
}
```

### 8. Docker Deployment

Build and run the Docker container:

```bash
cd deployment
docker build -t churn-prediction-api:latest .
docker run -d -p 5000:5000 --name churn-api churn-prediction-api:latest
```

Check the logs:

```bash
docker logs -f churn-api
```

Test the containerized API:

```bash
curl http://localhost:5000/health
```

## Project Structure After Workflow

```
churn_prediction/
├── data/
│   ├── raw/
│   │   └── customer_data.csv        # Generated sample data
│   └── processed/
│       ├── train.csv                # Preprocessed training data
│       ├── test.csv                 # Preprocessed test data
│       ├── scaler.pkl               # Feature scaler
│       └── label_encoders.pkl       # Label encoders
├── models/
│   ├── churn_model.pkl              # Trained model
│   └── model_metadata.json          # Model metadata
├── results/
│   ├── confusion_matrix.png         # Confusion matrix
│   ├── roc_curve.png               # ROC curve
│   ├── precision_recall_curve.png   # PR curve
│   └── feature_importance.png       # Feature importance
└── predictions.csv                  # Predictions output
```

## Key Metrics from Example Run

| Metric | Value |
|--------|-------|
| Accuracy | 69.25% |
| Precision | 60.10% |
| Recall | 74.39% |
| F1-Score | 66.49% |
| ROC-AUC | 78.36% |

## Notes

- The model achieves good recall (74%), meaning it catches most potential churners
- Precision is moderate (60%), so there are some false positives
- The ROC-AUC of 78% indicates good model discrimination
- Risk levels help prioritize customer retention efforts:
  - **High Risk (>60%)**: Immediate intervention needed
  - **Medium Risk (30-60%)**: Monitor and engage
  - **Low Risk (<30%)**: Standard service

## Next Steps

1. **Hyperparameter Tuning**: Optimize model parameters
2. **Feature Engineering**: Create more predictive features
3. **Model Ensemble**: Combine multiple models
4. **Production Monitoring**: Track model performance
5. **Automated Retraining**: Set up periodic model updates
