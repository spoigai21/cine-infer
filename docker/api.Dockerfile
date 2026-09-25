FROM python:3.11-slim
# LightGBM's OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY docker/requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt
COPY src/__init__.py src/serve.py src/serving.py src/
# The model bundle (make export) is mounted at runtime, never baked into the image.
ENV CINEINFER_BUNDLE=/app/models/serving CINEINFER_THREADS=1
EXPOSE 8000
CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
