FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FASTEMBED_CACHE_DIR=/opt/models

ARG EXTRAS="[anthropic]"
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".${EXTRAS}"

# Bake the embedding model into the image so the container starts without network access.
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir='/opt/models')"

COPY sample_docs ./sample_docs
COPY evals ./evals

RUN useradd --create-home --uid 1000 app && chown -R app /opt/models
USER app

EXPOSE 8000
CMD ["uvicorn", "docs_qa.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
