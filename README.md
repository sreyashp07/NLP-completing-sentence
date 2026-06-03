---
title: CustomerIntent AI
emoji: ◉
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: true
license: mit
short_description: Production-grade NLP customer support intent classifier
---

# CustomerIntent AI

Production-grade NLP system for automated customer support ticket classification and smart routing.

Live demo: https://huggingface.co/spaces/sreyashp07/customerintent-ai

## Tech Stack

- ML: scikit-learn TF-IDF + Logistic Regression
- NLP: NLTK with custom preprocessing pipeline
- Frontend: Streamlit with custom dark UI
- Backend: FastAPI with Pydantic v2 validation
- Containerization: Docker
- CI/CD: GitHub Actions with 200+ tests
- Experiment tracking: MLflow

## Features

- 9-class intent classification (payment, refund, account locked, etc.)
- Smart department routing
- Priority prediction (Critical/High/Medium/Low)
- Real-time confidence scores
- Keyword explainability

## Repository

https://github.com/sreyashp07/NLP-completing-sentence
