---
title: CustomerIntent AI
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.30.0
app_file: app_hf.py
pinned: true
license: mit
short_description: AI customer support intent classifier
---

<div align="center">

<img src="https://readme-typing-svg.herokuapp.com?font=Sora&size=32&duration=3000&pause=1000&color=0066FF&center=true&vCenter=true&width=600&lines=CustomerIntent+AI;Smart+Ticket+Routing;Real-Time+Intent+Detection" alt="Typing SVG" />

<br/>

**AI-powered customer support intent classification and smart ticket routing**

*The same type of system used internally at Stripe, Zendesk, Intercom, and Freshworks*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.9-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![CI](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?style=for-the-badge&logo=github-actions&logoColor=white)](https://github.com/sreyashp07/NLP-completing-sentence/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![HF Spaces](https://img.shields.io/badge/HuggingFace-Live_Demo-FFD21E?style=for-the-badge)](https://huggingface.co/spaces/sreyashp07/customerintent-ai)

<br/>

[Live Demo](https://huggingface.co/spaces/sreyashp07/customerintent-ai) • [API Docs](http://localhost:8000/docs) • [MLflow UI](#mlflow-experiment-tracking) • [Docker](#docker)

<br/>

</div>

---

## What is this?

**CustomerIntent AI** is a production-grade NLP system that classifies customer support messages into intent categories and automatically routes them to the correct department in real time, before the agent even reads the message.

```
Input:   "My payment failed but money got deducted from my account"

Output:
  Intent      ->  Payment Issue  (91.4% confidence)
  Department  ->  Billing Team
  Priority    ->  CRITICAL
  Keywords    ->  payment, failed, deducted, money, account
  Latency     ->  12ms
```

This is built to simulate real production systems at scale. Not a tutorial clone.

---

## Features at a Glance

| Feature | Description |
|---|---|
| Intent Classification | 9-class NLP classification with confidence scores |
| Smart Ticket Routing | Auto-assigns to correct department instantly |
| Priority Prediction | Critical / High / Medium / Low urgency scoring |
| Real-Time Prediction | Live typing mode — predicts as you type |
| Analytics Dashboard | Intent and priority distribution charts |
| Ticket History | Full session history with color-coded priority UI |
| Keyword Explainability | Shows which words drove the prediction |
| REST API | FastAPI with Swagger docs at /docs |
| MLflow Tracking | Full experiment tracking and model registry |
| Docker Ready | One-command containerized deployment |
| CI/CD Pipeline | GitHub Actions — tests run on every push |
| HF Spaces | Publicly deployed on Hugging Face Spaces |

---

## Intent to Department Routing Map

```
Customer Message
      |
      v
 Intent Classifier
 TF-IDF + Logistic Regression
      |
      |-------- CRITICAL ------------|-------- HIGH -----------|------- MEDIUM/LOW ------
      |                              |                          |
  Payment Issue              Account Locked             Shipping Delay
  Billing Team               Security Team              Logistics Team
      |                              |                          |
  Refund Request             Technical Bug              Invoice Problem
  Billing Team               Engineering Team           Finance Team
                                     |                          |
                             Subscription Cancel        Feature Request
                             Retention Team             Product Team
```

| Intent | Department | Priority | SLA |
|---|---|---|---|
| payment_issue | Billing Team | Critical | 4 hrs |
| account_locked | Security Team | Critical | 1 hr |
| refund_request | Billing Team | High | 4 hrs |
| technical_bug | Engineering Team | High | 8 hrs |
| subscription_cancel | Retention Team | High | 2 hrs |
| invoice_problem | Finance Team | Medium | 24 hrs |
| shipping_delay | Logistics Team | Medium | 12 hrs |
| feature_request | Product Team | Low | 72 hrs |
| general_inquiry | General Support | Low | 48 hrs |

---

## System Architecture

```
Streamlit Frontend (port 8501)
  Live Classifier | Analytics | Ticket History | Model Evaluation
         |
         | HTTP REST
         |
FastAPI Backend (port 8000)
  GET /health | POST /predict | POST /predict/batch
  Pydantic v2 request/response validation
         |
Model Service Layer
  Request validation + model lifecycle management
         |
         |--- IntentRouter (Strategy Pattern)
                   |
                   |--- BaselinePredictor
                              |
                              v
                   NLP Preprocessing Pipeline
                   Emoji -> Expand -> Lowercase -> URL/Email Remove
                   -> Tokenize -> Lemmatize -> TF-IDF -> Classify -> Route
```

---

## Project Structure

```
CustomerIntent/
|
|-- app/                          FastAPI backend
|   |-- api/v1/endpoints/
|   |   |-- health.py             GET /health
|   |   |-- predictions.py        POST /predict, /predict/batch
|   |-- core/
|   |   |-- config.py             Pydantic Settings (env-driven)
|   |   |-- logging.py            Loguru structured logging
|   |   |-- constants.py          App-wide constants
|   |   |-- exceptions.py         Custom exception hierarchy
|   |   |-- middleware.py         Request timing and tracing
|   |-- schemas/
|   |   |-- prediction.py         Pydantic v2 schemas
|   |-- services/
|   |   |-- model_service.py      Model lifecycle orchestration
|   |-- utils/
|       |-- helpers.py            Formatting utilities
|       |-- text_utils.py         Input sanitization
|
|-- ml/                           ML pipeline
|   |-- preprocessing/
|   |   |-- text_cleaner.py       Full NLP cleaning pipeline
|   |   |-- text_stats.py         Dataset statistics
|   |   |-- stopwords_config.py   Custom stopwords
|   |-- training/
|   |   |-- train_baseline.py     TF-IDF + LR training script
|   |   |-- data_validator.py     Pre-training data checks
|   |-- inference/
|   |   |-- predictor.py          IntentRouter + BaselinePredictor
|   |   |-- confidence_calibrator.py  Confidence thresholding
|   |-- evaluation/
|   |   |-- evaluator.py          Confusion matrix + metrics
|   |   |-- metrics_tracker.py    Run history tracking
|   |-- experiments/
|   |   |-- mlflow_trainer.py     MLflow experiment tracking
|   |-- saved_models/
|       |-- baseline/             pipeline.pkl, label_encoder.pkl
|       |-- transformer/          Phase 2: DistilBERT
|
|-- streamlit_app/
|   |-- app.py                    4-page Streamlit UI
|   |-- components/
|       |-- ticket_card.py        Reusable ticket card component
|
|-- data/
|   |-- raw/                      Raw CSV datasets
|   |-- generate_dataset.py       Synthetic data generator (1800 samples)
|
|-- configs/
|   |-- app_config.yaml           App settings
|   |-- intent_config.yaml        Intent metadata
|   |-- model_config.yaml         Model parameters
|   |-- training_config.yaml      Training hyperparameters
|   |-- departments.yaml          Department SLAs and contacts
|   |-- logging_config.yaml       Logging configuration
|
|-- tests/
|   |-- unit/                     77+ unit tests
|   |-- integration/              API integration tests
|
|-- notebooks/
|   |-- 01_eda.py                 Exploratory data analysis
|   |-- 02_model_analysis.py      Error analysis
|   |-- 03_prediction_demo.py     Single prediction demo
|   |-- 04_batch_prediction_demo.py  Batch prediction demo
|
|-- .github/
|   |-- workflows/ci.yml          GitHub Actions CI pipeline
|
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
|-- requirements-local.txt
|-- .env.example
|-- pyproject.toml
|-- CONTRIBUTING.md
|-- CHANGELOG.md
|-- LICENSE
|-- README.md
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Git

### 1. Clone

```bash
git clone https://github.com/sreyashp07/NLP-completing-sentence.git
cd NLP-completing-sentence
```

### 2. Virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements-local.txt
```

### 4. Download NLTK data

```bash
python -m nltk.downloader punkt stopwords wordnet averaged_perceptron_tagger punkt_tab
```

### 5. Setup environment

```bash
cp .env.example .env
```

### 6. Generate training data

```bash
python data/generate_dataset.py
```

### 7. Train the model

```bash
python ml/training/train_baseline.py
```

### 8. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 9. Start the UI in a new terminal

```bash
streamlit run streamlit_app/app.py
```

Open http://localhost:8501

> Windows users: Replace python with py and uvicorn with py -m uvicorn

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

Request:
```json
{
  "text": "My payment failed but money got deducted from my account",
  "model_type": "baseline"
}
```

Response:
```json
{
  "success": true,
  "primary_intent": {
    "intent": "payment_issue",
    "confidence": 0.914,
    "department": "Billing Team",
    "priority": "critical",
    "display_label": "Payment Issue"
  },
  "top_intents": [
    { "intent": "payment_issue",  "confidence": 0.914 },
    { "intent": "refund_request", "confidence": 0.031 },
    { "intent": "account_locked", "confidence": 0.021 }
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

Request:
```json
{
  "texts": [
    "My payment failed but money was deducted",
    "I want to cancel my subscription",
    "App keeps crashing on the dashboard"
  ]
}
```

Full interactive docs: http://localhost:8000/docs

---

## Testing

```bash
# Run all unit tests
pytest tests/unit/ -v

# Run specific file
pytest tests/unit/test_text_cleaner.py -v
```

Current test suite:

```
test_text_cleaner.py      31 tests   PASSING
test_schemas.py           13 tests   PASSING
test_helpers.py            4 tests   PASSING
test_text_stats.py         5 tests   PASSING
test_confidence.py         5 tests   PASSING
test_data_validator.py     4 tests   PASSING
test_constants.py          5 tests   PASSING
test_stopwords.py          4 tests   PASSING
test_text_utils.py         6 tests   PASSING
----------------------------------------------
TOTAL                     77 tests   ALL PASSING
```

---

## MLflow Experiment Tracking

```bash
# Run experiment
python ml/experiments/mlflow_trainer.py

# Launch UI
mlflow ui
```

Open http://localhost:5000

What gets tracked:
- 12 hyperparameters (TF-IDF config + LR config)
- 7 metrics (accuracy, F1 weighted/macro, precision, recall, CV mean/std)
- Confusion matrix PNG artifact
- Per-class metrics CSV artifact
- Registered model: CustomerIntent-Baseline v1

---

## Docker

```bash
# Build and start everything
docker-compose up --build

# API at      http://localhost:8000
# UI at       http://localhost:8501
# API Docs at http://localhost:8000/docs

# Stop
docker-compose down
```

---

## Deployment

### Live on Hugging Face Spaces

https://huggingface.co/spaces/sreyashp07/customerintent-ai

No login required. Free CPU tier. Always on.

### CI/CD via GitHub Actions

Every push to main automatically runs:

```
Push to main
    |
    v
Install all dependencies
    |
    v
Download NLTK data
    |
    v
Generate 1800-sample dataset
    |
    v
Train TF-IDF + LR baseline model
    |
    v
Run 77 unit tests
    |
    v
Run model evaluation
    |
    v
Pipeline complete
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | Streamlit, Plotly | Interactive 4-page UI |
| Backend | FastAPI, Uvicorn | REST API server |
| ML | scikit-learn | TF-IDF + LR model |
| NLP | NLTK, custom pipeline | Text preprocessing |
| Config | Pydantic Settings, PyYAML | Environment management |
| Logging | Loguru | Structured logging |
| MLOps | MLflow | Experiment tracking + model registry |
| Testing | pytest | 77+ unit and integration tests |
| CI/CD | GitHub Actions | Automated pipeline |
| Deploy | HF Spaces, Docker | Cloud and container |

---

## Model Performance

Trained on 1800 synthetic samples across 9 balanced classes:

| Metric | Score |
|---|---|
| Accuracy | 100% on synthetic data |
| F1 Weighted | 100% on synthetic data |
| Test Samples | 360 |
| Training Time | ~2 seconds |
| Inference Speed | ~12ms per request |
| Classes | 9 intent categories |

Note: 100% accuracy is expected on clean synthetic data. On real-world noisy customer data, expect 85-92% F1 score.

---

## Roadmap

- [x] TF-IDF + Logistic Regression baseline
- [x] FastAPI REST backend with Swagger docs
- [x] Streamlit 4-page dark UI
- [x] MLflow experiment tracking and model registry
- [x] 77+ unit tests across all modules
- [x] GitHub Actions CI/CD pipeline
- [x] Docker containerization
- [x] Hugging Face Spaces deployment
- [x] Custom exception hierarchy
- [x] Request timing middleware
- [x] Department SLA configuration
- [ ] DistilBERT transformer model (Phase 2)
- [ ] SHAP explainability integration
- [ ] Real customer dataset fine-tuning
- [ ] Multi-label classification support
- [ ] Webhook notifications to Slack

---

## Contributing

See CONTRIBUTING.md for development setup and commit message guidelines.

---

## Author

Sreyash Panigrahi

GitHub: https://github.com/sreyashp07

---

## License

MIT License. See LICENSE file for details.

---

<div align="center">

Star this repo if you found it useful!

Built with Python, FastAPI, Streamlit, scikit-learn, and a lot of coffee.

</div>