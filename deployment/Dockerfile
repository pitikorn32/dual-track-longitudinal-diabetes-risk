# DigiHealth Risk Score API — standalone deployment image.
#
# Build context is this folder. Export the model artifacts first
# (python export_models.py) to bake them into the image, or mount them
# at runtime with -v "$(pwd)/models:/app/models".
FROM python:3.11-slim

# libgomp1 is required by the xgboost runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
