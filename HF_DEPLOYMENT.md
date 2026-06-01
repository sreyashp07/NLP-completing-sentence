# Hugging Face Spaces Deployment

## Current Setup

- SDK: Docker
- App port: 7860
- App file: app_hf.py
- Pre-trains model at Docker build time
- No first-time training delay for users

## URL

https://huggingface.co/spaces/sreyashp07/customerintent-ai

## Build Process

1. Docker pulls python:3.11-slim base image
2. Installs system dependencies (gcc, build-essential)
3. Installs Python packages from requirements.txt
4. Downloads NLTK data
5. Pre-trains baseline model (1800 samples, 9 classes)
6. Copies app code
7. Starts Streamlit on port 7860

## Troubleshooting

If the Space shows runtime error:
1. Check Container Logs in the Logs tab
2. Look for Python tracebacks
3. Verify all imports succeed
4. Ensure NLTK data downloaded successfully

## Push to HF

```bash
git push hf main
```

## Force Rebuild

In HF Space Settings > Factory rebuild
