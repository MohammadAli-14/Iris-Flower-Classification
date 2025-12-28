# Iris Flower Classification

A Flask web app and REST API that serves a trained Iris classifier (SVM, ~96.7% accuracy) with a polished web UI and ready-to-use model artifacts.

## Project Structure
- [app.py](app.py): Flask app, API routes, model loading, batch predict.
- [templates/index.html](templates/index.html): Web UI for entering features and viewing predictions.
- [requirements.txt](requirements.txt): Python 3.13-compatible dependency set.
- [iris-flower-classification.ipynb](iris-flower-classification.ipynb): Training notebook (generates artifacts).
- Model artifacts: [iris_classification_model_20251228_161424.pkl](iris_classification_model_20251228_161424.pkl), [scaler_20251228_161424.pkl](scaler_20251228_161424.pkl), [label_encoder_20251228_161424.pkl](label_encoder_20251228_161424.pkl), [results_summary_20251228_161424.json](results_summary_20251228_161424.json).
- Deployment: [dockerfile](dockerfile) (update base image to Python ≥3.11/3.12 for current deps).

## Model Summary
- Best model: SVM
- Accuracy: 0.9667 (from [results_summary_20251228_161424.json](results_summary_20251228_161424.json))
- Features: sepal length (cm), sepal width (cm), petal length (cm), petal width (cm)
- Classes: setosa, versicolor, virginica

## Prerequisites
- Python 3.13 (or 3.11/3.12) with pip
- The three model artifacts (.pkl) placed in the repository root

## Setup (local)
```powershell
# From project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run the app
```powershell
python app.py
```
- Web UI: http://localhost:5000
- API health: http://localhost:5000/api/health

## API Endpoints
- GET `/`: Web form UI
- POST `/predict`: Predict single sample (form or JSON)
- POST `/batch_predict`: Predict multiple samples (JSON array)
- GET `/api/info`: Model metadata
- GET `/api/health`: Health check

### Example: single prediction (JSON)
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

### Example: batch prediction
```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '{"samples": [[5.1,3.5,1.4,0.2],[6.0,2.9,4.5,1.5]]}'
```

## Docker (update base image first)
- The current dockerfile uses `python:3.9-slim`; change to `python:3.12-slim` (or 3.13) to match the dependency floor in [requirements.txt](requirements.txt).
```bash
docker build -t iris-classifier .
docker run -p 5000:5000 iris-classifier
```

## Notes
- Do not run via Streamlit; this app is Flask-based. Use `python app.py` or Gunicorn/Waitress in production.
- If pip reports missing wheels for numpy/scikit-learn, ensure Python ≥3.12 and that pip is upgraded.
- Retraining/regeneration lives in [iris-flower-classification.ipynb](iris-flower-classification.ipynb); keep artifact filenames consistent with those loaded in [app.py](app.py).
