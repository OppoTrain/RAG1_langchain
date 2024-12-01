# Docker configuration file 
# Use a slim Python image with version 3.11
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's caching mechanism
COPY requirements.txt /app/

# Install system dependencies and Python dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the remaining application code into the container
COPY . /app/

# Expose the port FastAPI will run on (default is 8000)
EXPOSE 8000

# Set environment variables for your application (optional, recommended to pass via runtime)
ENV PYTHONUNBUFFERED=1

# Command to run the FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
