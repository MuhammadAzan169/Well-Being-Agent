# Dockerfile for deploying the Hugging Face application
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies including audio libraries
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    ffmpeg \
    portaudio19-dev \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p static/audio config cache DataSet

# Expose port (Hugging Face uses 7860)
EXPOSE 7860

# Health check - use the correct port
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start command for Hugging Face - FIXED PORT
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]