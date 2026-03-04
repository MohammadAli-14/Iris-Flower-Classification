# Use official Python runtime
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Copy your model files (adjust filenames as needed)
COPY iris_classification_model_20251228_161424.pkl .
COPY scaler_20251228_161424.pkl .
COPY label_encoder_20251228_161424.pkl .
COPY results_summary_20251228_161424.json .

# Create templates directory and copy HTML
RUN mkdir -p templates
COPY templates/index.html templates/

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Run the application with increased timeout for ML model loading
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]