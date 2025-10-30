# Deployment Guide

This guide covers various deployment options for the churn prediction model.

## Table of Contents

1. [Local Deployment](#local-deployment)
2. [Docker Deployment](#docker-deployment)
3. [Cloud Deployment](#cloud-deployment)
4. [Production Considerations](#production-considerations)

## Local Deployment

### Development Server

For development and testing:

```bash
cd deployment
export MODEL_PATH=../models/churn_model.pkl
export PREPROCESSOR_PATH=../data/processed/
python app.py
```

The API will be available at `http://localhost:5000`

### Production Server with Gunicorn

For production deployment:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Options:
- `-w 4`: Use 4 worker processes
- `-b 0.0.0.0:5000`: Bind to all interfaces on port 5000
- `--timeout 120`: Set timeout to 120 seconds

## Docker Deployment

### Build Docker Image

```bash
cd deployment
docker build -t churn-prediction-api:latest .
```

### Run Container

```bash
docker run -d \
  --name churn-api \
  -p 5000:5000 \
  -e MODEL_PATH=/app/models/churn_model.pkl \
  -e PREPROCESSOR_PATH=/app/data/processed/ \
  churn-prediction-api:latest
```

### View Logs

```bash
docker logs -f churn-api
```

### Stop Container

```bash
docker stop churn-api
docker rm churn-api
```

## Cloud Deployment

### AWS Elastic Beanstalk

1. Install EB CLI:
```bash
pip install awsebcli
```

2. Initialize EB application:
```bash
eb init -p python-3.9 churn-prediction-api
```

3. Create environment and deploy:
```bash
eb create churn-prediction-env
eb deploy
```

4. Open application:
```bash
eb open
```

### AWS Lambda + API Gateway

1. Package the application:
```bash
pip install -r requirements.txt -t package/
cp -r src package/
cp deployment/app.py package/
cd package && zip -r ../deployment.zip . && cd ..
```

2. Create Lambda function in AWS Console
3. Upload deployment.zip
4. Configure API Gateway to trigger Lambda

### Google Cloud Run

1. Build and push to Google Container Registry:
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/churn-prediction-api
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy churn-prediction-api \
  --image gcr.io/PROJECT_ID/churn-prediction-api \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Azure App Service

1. Create Azure Container Registry:
```bash
az acr create --resource-group myResourceGroup \
  --name myregistry --sku Basic
```

2. Build and push image:
```bash
az acr build --registry myregistry \
  --image churn-prediction-api:latest .
```

3. Deploy to App Service:
```bash
az webapp create --resource-group myResourceGroup \
  --plan myAppServicePlan --name churn-api \
  --deployment-container-image-name myregistry.azurecr.io/churn-prediction-api:latest
```

### Heroku

1. Create Heroku app:
```bash
heroku create churn-prediction-api
```

2. Add container registry:
```bash
heroku container:login
```

3. Build and push:
```bash
heroku container:push web
heroku container:release web
```

4. Open application:
```bash
heroku open
```

## Production Considerations

### Security

1. **API Authentication**: Add API key or OAuth authentication
2. **HTTPS**: Use SSL/TLS certificates
3. **Rate Limiting**: Implement rate limiting to prevent abuse
4. **Input Validation**: Validate all input data
5. **Environment Variables**: Store secrets in environment variables

### Monitoring

1. **Logging**: Implement structured logging
2. **Metrics**: Track API metrics (latency, requests, errors)
3. **Health Checks**: Regular health check endpoints
4. **Alerts**: Set up alerts for errors and performance issues

### Performance

1. **Caching**: Cache predictions for common inputs
2. **Load Balancing**: Use load balancer for multiple instances
3. **Auto-scaling**: Implement auto-scaling based on traffic
4. **Model Optimization**: Optimize model size and inference time

### Model Management

1. **Versioning**: Version control for models
2. **A/B Testing**: Test new models alongside old ones
3. **Rollback**: Easy rollback to previous model versions
4. **Monitoring**: Monitor model performance in production
5. **Retraining**: Regular model retraining with new data

### Example Production Configuration

```python
# production_config.py
import os

class Config:
    # Security
    SECRET_KEY = os.environ.get('SECRET_KEY')
    API_KEY = os.environ.get('API_KEY')
    
    # Model
    MODEL_PATH = os.environ.get('MODEL_PATH')
    PREPROCESSOR_PATH = os.environ.get('PREPROCESSOR_PATH')
    
    # Performance
    CACHE_TYPE = 'redis'
    CACHE_REDIS_URL = os.environ.get('REDIS_URL')
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'app.log')
    
    # Rate Limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_DEFAULT = "100 per hour"
```

## Testing Deployment

### Health Check

```bash
curl http://your-api-url/health
```

### Test Prediction

```bash
curl -X POST http://your-api-url/predict \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

### Load Testing

```bash
# Using Apache Bench
ab -n 1000 -c 10 -p test_data.json \
  -T application/json \
  http://your-api-url/predict
```

## Troubleshooting

### Common Issues

1. **Model not loading**: Check MODEL_PATH environment variable
2. **High latency**: Optimize model or use model serving platforms
3. **Memory issues**: Reduce model size or increase instance memory
4. **Connection timeouts**: Increase timeout settings

### Debug Mode

Enable debug mode for development:

```bash
export DEBUG=True
python app.py
```

**Warning**: Never enable debug mode in production!

## Support

For issues or questions, please open an issue in the repository.
