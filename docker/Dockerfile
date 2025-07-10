FROM python:3.10-slim

# Set environment variables for Python and virtual environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv $VIRTUAL_ENV

# Upgrade pip within virtual environment
RUN pip install --upgrade pip

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies in virtual environment
# Use development dependencies for Docker builds
RUN pip install -e ".[dev]"

# Copy source code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/
COPY tests/ tests/

# Create non-root user and change ownership
RUN useradd --create-home --shell /bin/bash bird_vision && \
    chown -R bird_vision:bird_vision /app /opt/venv

USER bird_vision

# Verify installation
RUN python -c "import bird_vision; print('Bird Vision installed successfully')" && \
    bird-vision --help

# Set default command
CMD ["python", "-m", "bird_vision.cli", "--help"]