# Use official slim Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face Spaces (UID 1000)
RUN useradd -m -u 1000 user
WORKDIR $HOME/app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache RoBERTa model weights inside Docker image
RUN python -c "from transformers import AutoTokenizer, AutoModelForSequenceClassification; \
    AutoTokenizer.from_pretrained('Shirisha-23/movie-review-roberta'); \
    AutoModelForSequenceClassification.from_pretrained('Shirisha-23/movie-review-roberta')"

# Copy application files and grant permissions
COPY --chown=user:user . $HOME/app
USER user

# Expose port (7860 is default for Hugging Face Spaces)
EXPOSE 7860

# Start Gunicorn server (binds to $PORT on Railway or 7860 on Hugging Face Spaces)
CMD exec gunicorn app_flask:app --bind 0.0.0.0:${PORT:-7860} --workers 1 --threads 4 --timeout 120
