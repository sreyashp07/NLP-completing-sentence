# CustomerIntent AI

### AI-Powered Customer Support Intent Classification and Smart Ticket Routing

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Predict customer intent before they finish typing. Route tickets automatically. Built for scale.

---

## What is CustomerIntent AI?

CustomerIntent AI is a production-grade NLP system that classifies customer support messages into intent categories and automatically routes them to the correct department in real time.

This is the same type of system used internally by companies like Stripe, Zendesk, Intercom, and Freshworks to handle thousands of support tickets per hour.

**Example:**

```
Input:  "My payment failed but money got deducted from my account"

Output:
  Intent:     Payment Issue (87.3% confidence)
  Department: Billing Team
  Priority:   CRITICAL
  Keywords:   payment, failed, deducted, account
```

---

## Features

| Feature | Description |
|---|---|
| Intent Classification | 9-class NLP classification with confidence scores |
| Smart Ticket Routing | Auto-assigns tickets to the correct department |
| Priority Prediction | Critical / High / Medium / Low urgency scoring |
| Real-Time Prediction | Live typing mode — predicts as you type |
| Analytics Dashboard | Intent distribution and priority breakdown charts |
| Ticket History | Full history with color-coded priority UI |
| Keyword Explainability | Shows which words drove the prediction |
| REST API | FastAPI backend with Swagger docs at /docs |
| Docker Ready | One-command deployment |

---

## Supported Intents

| Intent | Department | Priority |
|---|---|---|
| payment_issue | Billing Team | Critical |
| account_locked | Security Team | Critical |
| refund_request | Billing Team | High |
| technical_bug | Engineering Team | High |
| subscription_cancel | Retention Team | High |
| invoice_problem | Finance Team | Medium |
| shipping_delay | Logistics Team | Medium |
| feature_request | Product Team | Low |
| general_inquiry | General Support | Low |

---

## Architecture

```
Streamlit Frontend
  Live Classifier | Analytics | Ticket History
         |
         | HTTP REST
         |
FastAPI Backend
  GET /health | POST /predict | POST /predict/batch
         |
Model Service Layer
  Request validation + routing
         |
Inference Engine
  IntentRouter -> BaselinePredictor
         |
NLP Preprocessing Pipeline
  Clean -> Tokenize -> Lemmatize -> TF-IDF -> Classify
```

---

## Project Structure

```
CustomerIntent/
│
├── app/
│   ├── api/
│   │   └── v1/
│   │       └── endpoints/
│   │           ├── health.py             # GET /health
│   │           └── predictions.py        # POST /predict, /predict/batch
│   ├── core/
│   │   ├── config.py                     # Pydantic Settings (env-driven)
│   │   └── logging.py                    # Loguru structured logging
│   ├── schemas/
│   │   └── prediction.py                 # Pydantic v2 request/response models
│   ├── services/
│   │   └── model_service.py              # Model lifecycle + inference orchestration
│   └── main.py                           # App factory + lifespan
│
├── ml/
│   ├── preprocessing/
│   │   └── text_cleaner.py               # Full NLP cleaning pipeline
│   ├── training/
│   │   └── train_baseline.py             # TF-IDF + LR training script
│   ├── inference/
│   │   └── predictor.py                  # IntentRouter + BaselinePredictor
│   ├── evaluation/                       # Metrics and evaluation scripts
│   └── saved_models/
│       ├── baseline/                     # pipeline.pkl, label_encoder.pkl
│       └── transformer/                  # Phase 2: DistilBERT
│
├── streamlit_app/
│   └── app.py                            # Full Streamlit UI (3 pages)
│
├── data/
│   ├── raw/                              # Raw CSV datasets
│   ├── processed/                        # Preprocessed data
│   └── generate_dataset.py              # Synthetic data generator
│
├── configs/
│   ├── app_config.yaml                   # App settings
│   ├── intent_config.yaml                # Intent to department/priority mapping
│   └── model_config.yaml                 # Model parameters
│
├── tests/
│   ├── unit/                             # Unit tests
│   └── integration/                      # Integration tests
│
├── notebooks/                            # EDA and experiment notebooks
├── logs/                                 # Application logs
├── Dockerfile                            # Container definition
├── docker-compose.yml                    # Multi-service orchestration
├── requirements.txt                      # Pinned dependencies
├── .env.example                          # Environment variable template
└── pyproject.toml                        # Project metadata and tooling config
```

---

## Quick Start

### Prerequisites

- Python 3.10 or higher
- Git

### 1. Clone the repository

```bash
git clone https://github.com/sreyashp07/NLP-completing-sentence.git
cd NLP-completing-sentence
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download NLTK data

```bash
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger
```

### 5. Set up environment variables

```bash
cp .env.example .env
```

### 6. Generate training data

```bash
python data/generate_dataset.py
```

### 7. Train the baseline model

```bash
python ml/training/train_baseline.py
```

### 8. Start the FastAPI backend

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 9. Start the Streamlit frontend (open a new terminal)

```bash
streamlit run streamlit_app/app.py
```

Open your browser at http://localhost:8501

---

## API Reference

### Health Check

```
GET /api/v1/health
```

Response:

```json
{
  "status": "healthy",
  "app_name": "CustomerIntent",
  "version": "1.0.0",
  "model_loaded": true,
  "environment": "development"
}
```

### Single Prediction

```
POST /api/v1/predict
Content-Type: application/json
```

Request body:

```json
{
  "text": "My payment failed but money got deducted",
  "model_type": "baseline"
}
```

Response:

```json
{
  "success": true,
  "text": "My payment failed but money got deducted",
  "primary_intent": {
    "intent": "payment_issue",
    "confidence": 0.873,
    "department": "Billing Team",
    "priority": "critical",
    "display_label": "Payment Issue"
  },
  "top_intents": [
    { "intent": "payment_issue", "confidence": 0.873, "display_label": "Payment Issue" },
    { "intent": "refund_request", "confidence": 0.081, "display_label": "Refund Request" },
    { "intent": "invoice_problem", "confidence": 0.024, "display_label": "Invoice Problem" }
  ],
  "keywords": ["payment", "failed", "deducted", "money", "account"],
  "model_used": "TF-IDF + Logistic Regression",
  "processing_time_ms": 12.4
}
```

### Batch Prediction

```
POST /api/v1/predict/batch
Content-Type: application/json
```

Request body:

```json
{
  "texts": [
    "My payment failed but money was deducted",
    "I want to cancel my subscription",
    "App keeps crashing on the dashboard"
  ],
  "model_type": "baseline"
}
```

Full interactive API docs are available at http://localhost:8000/docs once the backend is running.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly |
| Backend | FastAPI, Uvicorn |
| ML / NLP | scikit-learn, NLTK, spaCy |
| Deep Learning | PyTorch, HuggingFace Transformers (Phase 2) |
| Config | Pydantic Settings, PyYAML |
| Logging | Loguru |
| MLOps | MLflow |
| Containerization | Docker, docker-compose |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Docker

```bash
docker-compose up --build
```

API will be available at http://localhost:8000

UI will be available at http://localhost:8501

---

## Windows Users Note

If `python` command is not found, use `py` instead:

```bash
py data/generate_dataset.py
py ml/training/train_baseline.py
py -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
py -m streamlit run streamlit_app/app.py
```

---

## Roadmap

- [x] TF-IDF + Logistic Regression baseline
- [x] FastAPI backend with REST endpoints
- [x] Streamlit UI with live prediction
- [x] Intent to department routing
- [x] Priority scoring
- [ ] DistilBERT transformer model
- [ ] MLflow experiment tracking
- [ ] SHAP explainability
- [ ] GitHub Actions CI/CD
- [ ] Hugging Face Spaces deployment

---

## Author

Sreyash P — https://github.com/sreyashp07

---

## License

MIT License. See LICENSE file for details.
## Running with Docker

### Prerequisites
- Docker Desktop installed and running

### One-command start

```bash
# Train model first (required before Docker)
py ml/training/train_baseline.py

# Build and start all services
docker-compose up --build
```

Services:
- API: http://localhost:8000
- UI:  http://localhost:8501
- API Docs: http://localhost:8000/docs

### Individual containers

```bash
# API only
docker-compose up api

# Stop everything
docker-compose down
```
