FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md ./
COPY askdocs ./askdocs
RUN pip install --no-cache-dir -e ".[api]"
COPY configs ./configs
COPY data ./data
COPY eval ./eval
# Build the offline index at image build time so the container starts instantly.
RUN askdocs ingest --config configs/offline.yaml
EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1
CMD ["askdocs", "serve", "--config", "configs/offline.yaml", "--host", "0.0.0.0", "--port", "8000"]
