# Use an official lightweight Python image
# Python 3.11 offers solid compatibility with common data science & ML packages
FROM python:3.11-slim

# Set environment variables:
# - PYTHONDONTWRITEBYTECODE: prevents Python from writing .pyc files
# - PYTHONUNBUFFERED: ensures stdout/stderr are flushed immediately for Docker logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container
WORKDIR /app

# Install essential system packages (clean up apt cache afterwards to keep image small)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Layer caching: copy requirements first so rebuilds with code changes don't reinstall dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Default command (can be customized or overridden in docker-compose.yml)
CMD ["python"]

