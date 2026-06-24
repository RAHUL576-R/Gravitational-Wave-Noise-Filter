# Dockerfile for Railway — GW Denoiser
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required by gwpy and scipy
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    gfortran \
    libhdf5-dev \
    libfftw3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch nightly first (closest to local 2.12.1 nightly build)
RUN pip install --no-cache-dir --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/cpu

# Copy requirements and install everything else
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]