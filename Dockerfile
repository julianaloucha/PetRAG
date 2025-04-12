# Use a base image with Python and CUDA support (optional depending on GPU needs)
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy your application code
COPY working_app.py /app/working_app.py

# Optional: copy requirements if you have one
# COPY requirements.txt /app/requirements.txt
COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install streamlit transformers sentence-transformers faiss-cpu pandas torch groq supabase

# Expose the Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "working_app.py", "--server.port=8501", "--server.enableCORS=false"]
