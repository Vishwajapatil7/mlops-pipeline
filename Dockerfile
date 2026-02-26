# Use lightweight Python image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy all project files into container
COPY . .

# Install system dependencies (needed for sklearn, pandas, etc.)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install extra MLOps + API tools
RUN pip install --no-cache-dir dvc fastapi uvicorn scikit-learn pandas

# Expose API port
EXPOSE 5000

# Run DVC pipeline first, then start API server
CMD dvc repro && uvicorn app.main:app --host 0.0.0.0 --port 5000