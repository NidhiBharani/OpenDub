# Build the web UI, then run the FastAPI server which serves both API and UI on :8000.
FROM node:20-slim AS webbuild
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install
COPY web/ ./
RUN npm run build

FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY server/pyproject.toml server/
COPY server/app server/app
RUN pip install --no-cache-dir -e ./server
# Optional heavy OSS model extras — uncomment to bake into the image (large!):
# RUN pip install --no-cache-dir -e './server[asr,separation,tts]'
COPY configs/providers.example.yaml configs/
COPY --from=webbuild /web/dist web/dist
ENV OPENDUB_ROOT=/app
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "server", "--host", "0.0.0.0", "--port", "8000"]
