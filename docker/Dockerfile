FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install -e .

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Create non-root user
RUN useradd --create-home --shell /bin/bash bird_vision
USER bird_vision

# Set default command
CMD ["python", "-m", "bird_vision.cli", "--help"]