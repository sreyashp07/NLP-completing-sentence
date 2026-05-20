# Changelog

## v1.0.0 (2026-05-20)

### Added
- 9-class intent classification with TF-IDF + Logistic Regression
- Smart ticket routing to correct department
- Priority prediction (Critical/High/Medium/Low)
- FastAPI backend with /predict and /predict/batch endpoints
- Streamlit frontend with 4 pages (Classifier, Analytics, History, Evaluation)
- MLflow experiment tracking with model registry
- Docker + docker-compose setup
- GitHub Actions CI pipeline
- Hugging Face Spaces deployment
- 70+ unit tests across all modules
- NLP preprocessing pipeline (cleaning, lemmatization, emoji handling)
- Synthetic dataset generator (1800 samples, 9 classes)
- Model evaluation with confusion matrix and per-class metrics
- Confidence calibration utilities
- Dataset validation module

### Models
- Baseline: TF-IDF + Logistic Regression (accuracy: 1.0 on synthetic data)
- Transformer: DistilBERT (Phase 2 - coming soon)
