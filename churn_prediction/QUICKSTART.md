# Quick Start Guide - Churn Prediction

This guide will help you get started with the churn prediction project in 5 minutes.

## Step 1: Install Dependencies

```bash
cd churn_prediction
pip install -r requirements.txt
```

## Step 2: Generate Sample Data

```bash
cd data
python generate_sample_data.py
cd ..
```

This creates a sample dataset at `data/raw/customer_data.csv` with 2000 customer records.

## Step 3: Preprocess Data

```bash
python src/data_preprocessing.py \
    --input data/raw/customer_data.csv \
    --output data/processed/
```

This will:
- Clean the data
- Encode categorical features
- Create new features
- Split into train/test sets
- Save preprocessed data and preprocessor objects

## Step 4: Train Model

```bash
python src/train_model.py \
    --data data/processed/train.csv \
    --val-data data/processed/test.csv \
    --output models/
```

This will:
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Evaluate each model
- Save the best performing model

## Step 5: Evaluate Model

```bash
python src/evaluate_model.py \
    --model models/churn_model.pkl \
    --data data/processed/test.csv \
    --output results/
```

This will:
- Generate comprehensive evaluation metrics
- Create visualizations (confusion matrix, ROC curve, etc.)
- Save results to the `results/` directory

## Step 6: Make Predictions

```bash
python src/predict.py \
    --model models/churn_model.pkl \
    --preprocessor data/processed/ \
    --input data/raw/customer_data.csv \
    --output predictions.csv
```

## Step 7: Deploy API

```bash
cd deployment
export MODEL_PATH=../models/churn_model.pkl
export PREPROCESSOR_PATH=../data/processed/
python app.py
```

The API will be available at `http://localhost:5000`

## Test the API

### Health Check
```bash
curl http://localhost:5000/health
```

### Make Prediction
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

## Docker Deployment (Optional)

```bash
cd deployment
docker build -t churn-prediction-api .
docker run -p 5000:5000 churn-prediction-api
```

## Next Steps

- Explore the Jupyter notebook: `notebooks/exploratory_analysis.ipynb`
- Experiment with different model parameters
- Try hyperparameter tuning
- Add more features to improve accuracy
- Deploy to cloud platforms (AWS, Azure, GCP)

## Troubleshooting

### Issue: Missing dependencies
**Solution:** Run `pip install -r requirements.txt`

### Issue: Data not found
**Solution:** Make sure you generated sample data: `python data/generate_sample_data.py`

### Issue: Model not found
**Solution:** Train the model first: `python src/train_model.py ...`

### Issue: API not responding
**Solution:** Check if the model and preprocessor paths are correct

## Need Help?

Check the main README or open an issue in the repository.
