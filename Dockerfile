FROM python:3.11-slim
WORKDIR /app

# OpenCV's Python wheel needs these shared libraries at runtime.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       libgl1 \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
       libxcb1 \
    && rm -rf /var/lib/apt/lists/*

COPY . /app
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e '.[ml,dev]'
EXPOSE 8000
# Render provides PORT at runtime (commonly 10000); retain 8000 as the local default.
CMD ["sh", "-c", "exec uvicorn fireguard.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
