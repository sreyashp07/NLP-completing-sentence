# ── Stage 1: Base Python image ─────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing .pyc files and enable stdout logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Install Python dependencies ────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Download NLTK data at build time ──────────────────────────────────────────
RUN python -c "import nltk; \
    nltk.download('punkt', quiet=True); \
    nltk.download('stopwords', quiet=True); \
    nltk.download('wordnet', quiet=True); \
    nltk.download('averaged_perceptron_tagger', quiet=True); \
    nltk.download('punkt_tab', quiet=True)"

# ── Copy project files ─────────────────────────────────────────────────────────
COPY app/ ./app/
COPY ml/ ./ml/
COPY configs/ ./configs/
COPY data/ ./data/
COPY streamlit_app/ ./streamlit_app/
COPY .env.example .env

# ── Expose ports ───────────────────────────────────────────────────────────────
# 8000 = FastAPI backend
# 8501 = Streamlit frontend
EXPOSE 8000 8501

# ── Default command: start FastAPI ─────────────────────────────────────────────
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
