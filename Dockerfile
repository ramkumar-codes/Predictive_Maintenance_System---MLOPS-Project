FROM python:3.11-slim
WORKDIR /app

# Install build deps and Python packages
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Expose ports
EXPOSE 5001 8501
ENV STREAMLIT_SERVER_HEADLESS=true

# Launch MLflow server + Streamlit
CMD ["sh","-c", "\
    mlflow server \
      --backend-store-uri /app/mlruns \
      --default-artifact-root /app/mlruns \
      --host 0.0.0.0 \
      --port 5001 & \
    streamlit run app/streamlit_app.py \
      --server.address=0.0.0.0 \
      --server.port=8501 \
"]


