# Multi-stage build for Streamlit app
FROM python:3.11-slim AS builder
WORKDIR /app
# Copy dependencies
COPY requirements.txt .
# Install dependencies in a temporary container
RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get remove -y gcc && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
# Final runtime image
FROM python:3.11-slim
WORKDIR /app
# Copy installed dependencies from the builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# Copy the application code
COPY . .
# Expose the port
EXPOSE 8501
# Start the Streamlit app
CMD ["streamlit", "run", "streamlit_app.py"]