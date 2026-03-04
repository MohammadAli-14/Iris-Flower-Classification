```markdown
# Iris Flower Classification 🌸

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

A production-ready web application and REST API that serves a trained Iris flower classifier. The model (SVM) achieves **96.7% accuracy** on the classic Iris dataset. The app features a polished web interface for single predictions and supports batch predictions via JSON API.

![Web UI Screenshot](https://via.placeholder.com/800x400?text=Iris+Classification+UI) <!-- Replace with actual screenshot -->

## Features ✨

- **Interactive Web UI** – Enter sepal/petal measurements and get instant predictions with confidence scores.
- **REST API** – JSON endpoints for single and batch predictions, model metadata, and health checks.
- **High Accuracy** – SVM model with 96.7% accuracy, trained on the Iris dataset.
- **Model Artifacts** – Pre-trained model, scaler, and label encoder included; easily retrain using the provided Jupyter notebook.
- **Docker Support** – Containerize the app for consistent deployment.
- **Production Ready** – Uses Gunicorn as WSGI server; ready for deployment on platforms like Heroku, Render, or AWS.

## Project Structure 📁

```
.
├── app.py                          # Flask application with routes and prediction logic
├── dockerfile                       # Docker configuration (update base image to Python 3.12+)
├── requirements.txt                 # Python dependencies
├── iris-flower-classification.ipynb # Jupyter notebook for model training
├── Procfile                         # For Heroku deployment (Gunicorn)
├── README.md                        # This file
├── templates/
│   └── index.html                   # Web UI template
└── model_artifacts/                  # (Generated files – placed in root for simplicity)
    ├── iris_classification_model_20251228_161424.pkl
    ├── scaler_20251228_161424.pkl
    ├── label_encoder_20251228_161424.pkl
    └── results_summary_20251228_161424.json
```

## Model Information 📊

| Property          | Value                                      |
|-------------------|--------------------------------------------|
| **Best Model**    | SVM (Support Vector Machine)               |
| **Accuracy**      | 96.67%                                     |
| **Features**      | sepal length (cm), sepal width (cm), petal length (cm), petal width (cm) |
| **Classes**       | setosa, versicolor, virginica              |
| **Training Date** | 2025-12-28                                  |

The model was trained on the well-known [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris) using scikit-learn. The training notebook (`iris-flower-classification.ipynb`) includes data preprocessing, model comparison, and artifact generation.

## Prerequisites 📋

- Python **3.13** (or 3.11/3.12) – the dependencies are compatible with these versions.
- pip (latest version recommended)
- Docker (optional, for containerized deployment)

## Installation ⚙️

### Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/MohammadAli-14/Iris-Flower-Classification.git
   cd Iris-Flower-Classification
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Linux/Mac
   .\.venv\Scripts\Activate.ps1   # Windows PowerShell
   ```

3. **Upgrade pip and install dependencies**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python app.py
   ```

   The app will be available at `http://localhost:5000`.

### Docker

The provided `dockerfile` uses `python:3.9-slim`; for best compatibility with the current dependencies, update the base image to `python:3.12-slim` or `python:3.13-slim`.

1. **Build the Docker image**
   ```bash
   docker build -t iris-classifier .
   ```

2. **Run the container**
   ```bash
   docker run -p 5000:5000 iris-classifier
   ```

## Usage 🚀

### Web Interface

Visit `http://localhost:5000` in your browser. Use the form to input the four feature values. You can also load sample values for each Iris species using the provided buttons. After clicking **Predict Species**, the result and class probabilities are displayed.

### API Endpoints

All endpoints return JSON responses.

#### `GET /api/health`
Health check to verify that the model is loaded.
```bash
curl http://localhost:5000/api/health
```

#### `GET /api/info`
Returns metadata about the model (features, classes, model type).
```bash
curl http://localhost:5000/api/info
```

#### `POST /predict`
Predict a single sample. Accepts form data (from web UI) or JSON.
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

#### `POST /batch_predict`
Predict multiple samples at once. Expects a JSON object with a `samples` key containing an array of feature arrays.
```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"samples": [[5.1, 3.5, 1.4, 0.2], [6.0, 2.9, 4.5, 1.5]]}'
```

### API Response Format

**Single Prediction (Success)**
```json
{
  "success": true,
  "prediction": {
    "class": "setosa",
    "class_encoded": 0,
    "confidence": 0.98,
    "probabilities": {
      "setosa": 0.98,
      "versicolor": 0.02,
      "virginica": 0.0
    }
  },
  "input_features": {
    "sepal length (cm)": 5.1,
    "sepal width (cm)": 3.5,
    "petal length (cm)": 1.4,
    "petal width (cm)": 0.2
  }
}
```

**Batch Prediction (Success)**
```json
{
  "success": true,
  "predictions": [
    {
      "sample_id": 0,
      "prediction": "setosa",
      "prediction_encoded": 0,
      "features": { ... }
    },
    ...
  ],
  "count": 2
}
```

## Training the Model 🧠

If you wish to retrain the model or experiment with different algorithms, use the Jupyter notebook:

```bash
jupyter notebook iris-flower-classification.ipynb
```

The notebook:
- Loads the Iris dataset from scikit-learn.
- Splits data into training and test sets.
- Scales features using `StandardScaler`.
- Trains and evaluates multiple classifiers (Logistic Regression, Decision Tree, Random Forest, SVM, KNN, Gaussian NB, Gradient Boosting).
- Selects the best model based on accuracy.
- Saves the model, scaler, label encoder, and a results summary with timestamps.

**Note:** Update the filenames in `app.py` if you regenerate artifacts with new timestamps.

## Deployment ☁️

### Using Gunicorn (Production)

For production, the app uses Gunicorn as defined in the `Procfile`:
```
web: gunicorn app:app
```

You can also run it manually:
```bash
gunicorn --bind 0.0.0.0:5000 app:app
```

### Deploy to Heroku / Render / etc.

1. Ensure the `Procfile` is present.
2. Set the environment to Python 3.13 (or compatible) on the platform.
3. Push the code; the platform will automatically install dependencies and start the web process.

## Technologies Used 🛠️

- **Backend**: Flask, Gunicorn
- **Machine Learning**: scikit-learn, joblib, numpy, pandas
- **Frontend**: HTML, CSS, JavaScript (vanilla)
- **Containerization**: Docker
- **Development**: Jupyter Notebook, Python 3.13

## License 📄

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Acknowledgements 🙏

- The Iris dataset – originally introduced by Ronald Fisher in 1936.
- scikit-learn community for providing excellent machine learning tools.
- Flask for making web development in Python simple and elegant.

---

⭐ If you find this project useful, please consider giving it a star on GitHub!
```