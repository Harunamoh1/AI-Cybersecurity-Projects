# syntax=docker/dockerfile:1.7
FROM python:3.13-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

# Unprivileged runtime user; the only writable location is /data (scan history).
RUN useradd --system --uid 10001 --home /app phishlens && mkdir /data && chown phishlens /data

COPY phishlens ./phishlens
COPY models ./models

ENV PHISHLENS_MODEL_DIR=/app/models \
    PHISHLENS_DB_PATH=/data/phishlens.db

USER phishlens
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "phishlens.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
