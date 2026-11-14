# syntax=docker/dockerfile:1.7
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp: OpenMP runtime required by the XGBoost wheel
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Unprivileged runtime user; the only writable location is /data (detection history).
RUN useradd --system --uid 10001 --home /app sentinel && mkdir /data && chown sentinel /data && chmod 700 /data

COPY sentinelai ./sentinelai
COPY models/sentinel_xgb.ubj models/model_meta.json models/SHA256SUMS models/metrics.json ./models/
COPY samples ./samples

ENV SENTINEL_MODEL_DIR=/app/models \
    SENTINEL_SAMPLES_DIR=/app/samples \
    SENTINEL_DB_PATH=/data/sentinel.db \
    MPLCONFIGDIR=/tmp \
    NUMBA_CACHE_DIR=/tmp

USER sentinel
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"

CMD ["uvicorn", "sentinelai.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-server-header"]
