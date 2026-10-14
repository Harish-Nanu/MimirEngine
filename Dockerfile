FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/models
WORKDIR /app

# CPU-only torch keeps the image ~2GB smaller than the default CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY mimir ./mimir
RUN pip install ".[local]"

# Bake the default model into the image so scaled-out workers don't each download it.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"
# Use the baked-in copy; don't call out to the Hugging Face Hub on every start.
ENV HF_HUB_OFFLINE=1

ENTRYPOINT ["mimir"]
CMD ["api"]
