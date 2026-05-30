# Deployment Guide

## Local Development

```bash
# 1. Generate data
python data/generate_dataset.py

# 2. Train model
python ml/training/train_baseline.py

# 3. Start API (terminal 1)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 4. Start UI (terminal 2)
streamlit run streamlit_app/app.py
```

## Hugging Face Spaces

The app auto-trains the model on first startup.
Model artifacts are cached in /tmp for subsequent restarts.

Live URL: https://huggingface.co/spaces/sreyashp07/customerintent-ai

Push to HF:
```bash
git push hf main
```

## Docker

```bash
# Build and run
docker-compose up --build

# API: http://localhost:8000
# UI:  http://localhost:8501
```

## Environment Variables

Copy .env.example to .env and configure:
```bash
cp .env.example .env
```

Key variables:
- APP_ENV: development or production
- LOG_LEVEL: INFO or DEBUG
- DEFAULT_MODEL: baseline or transformer
- CONFIDENCE_THRESHOLD: 0.6 (default)
