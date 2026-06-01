FROM python:3.11-slim

WORKDIR /code

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/code \
    HOME=/code \
    NLTK_DATA=/code/nltk_data

RUN apt-get update && apt-get install -y \
    build-essential gcc python3-dev curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('punkt', download_dir='/code/nltk_data', quiet=True); nltk.download('stopwords', download_dir='/code/nltk_data', quiet=True); nltk.download('wordnet', download_dir='/code/nltk_data', quiet=True); nltk.download('punkt_tab', download_dir='/code/nltk_data', quiet=True); nltk.download('averaged_perceptron_tagger', download_dir='/code/nltk_data', quiet=True)"

COPY . .

RUN chmod -R 777 /code && chmod +x docker-startup.sh

# Pre-train the model at build time
RUN python data/generate_dataset.py && \
    python ml/training/train_baseline.py

EXPOSE 7860

CMD ["./docker-startup.sh"]
