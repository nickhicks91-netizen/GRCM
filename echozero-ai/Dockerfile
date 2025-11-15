FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Install package in editable mode
RUN pip install -e .

# Create directories
RUN mkdir -p experiments data

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "scripts/train.py"]
