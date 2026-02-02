FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create directories
RUN mkdir -p uploads cache

# Expose ports
EXPOSE 8000 8501

# Default to API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
