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

# CustomerIntent AI

AI-powered customer support intent classification and smart ticket routing system.

Built with FastAPI, Streamlit, scikit-learn, and NLTK.

## Features
- 9-class intent classification
- Smart department routing
- Priority prediction (Critical/High/Medium/Low)
- Real-time confidence scores
- Keyword explainability

## Intents Supported
- Payment Issue, Refund Request, Account Locked
- Technical Bug, Feature Request, Subscription Cancel
- Invoice Problem, Shipping Delay, General Inquiry

## Tech Stack
- Frontend: Streamlit
- ML: scikit-learn TF-IDF + Logistic Regression
- NLP: NLTK, custom preprocessing pipeline
