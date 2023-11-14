# Use a base image that supports Python
FROM python:3.11.4-slim

# Set a working directory
WORKDIR /app

# Install system dependencies required for building certain Python packages
RUN apt-get update \
    && apt-get install -y gcc python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files into the Docker image except those in .dockerignore
COPY . /app

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Command to run your script
CMD ["python", "/app/main.py"]
