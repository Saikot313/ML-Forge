FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MLFORGE_CHECKPOINT=/app/model/best_model.pt

WORKDIR /app

# CPU-only PyTorch wheel (much smaller than the default CUDA build).
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml ./
COPY src ./src
RUN pip install .

RUN useradd --create-home --uid 1000 mlforge && mkdir -p /app/model && chown -R mlforge /app
USER mlforge

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# The app loads the checkpoint at $MLFORGE_CHECKPOINT on start-up. If it is missing,
# /health reports "degraded" and prediction endpoints return 503.
CMD ["uvicorn", "mlforge.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]