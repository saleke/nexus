FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev postgresql-client && rm -rf /var/lib/apt/lists/*
COPY pyproject.toml README.md ./
COPY post_clustering_pipeline ./post_clustering_pipeline
RUN pip install --no-cache-dir . && python -m spacy download en_core_web_sm

EXPOSE 8000
CMD ["python", "-m", "post_clustering_pipeline"]
