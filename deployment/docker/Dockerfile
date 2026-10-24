FROM python:3.11-slim

WORKDIR /app

COPY deployment/docker/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

ENV PYTHONUNBUFFERED=1
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=4)"

CMD [ "python", "-m", "src.api.app" ]
