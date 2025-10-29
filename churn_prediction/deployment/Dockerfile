# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY ../requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . /app/deployment/
COPY ../src /app/src/
COPY ../models /app/models/
COPY ../data/processed /app/data/processed/

# Set environment variables
ENV MODEL_PATH=/app/models/churn_model.pkl
ENV PREPROCESSOR_PATH=/app/data/processed/
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 5000

# Run the application
CMD ["python", "deployment/app.py"]
