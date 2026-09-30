FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[ml]"
# The index is built outside the image and mounted at /app/data (see README)
ENV RAG_DATA_DIR=/app/data
EXPOSE 8000
CMD ["uvicorn", "legalrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
