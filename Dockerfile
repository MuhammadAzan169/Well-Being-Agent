# Dockerfile for Well Being Agent — Hugging Face deployment
FROM python:3.12-slim

WORKDIR /app

# System dependencies (minimal — no audio libs needed)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install Python deps (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create runtime directories
RUN mkdir -p static/audio cache DataSet

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Health check - use the correct port
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start command for Hugging Face - FIXED PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]