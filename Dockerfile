FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get upgrade -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Copy entrypoint script
RUN chmod +x docker/entrypoint.sh

# Set environment variables
ENV PYTHONUNBUFFERED=1

# ENTRYPOINT ["docker/entrypoint.sh"]
ENTRYPOINT ["python", "zotero_llm/main.py"]
