FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -U pip && pip install --no-cache-dir -e '.[ml,dev]'
EXPOSE 8000
CMD ["sh", "-c", "uvicorn fireguard.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
