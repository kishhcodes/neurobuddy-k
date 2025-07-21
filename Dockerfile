FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV and dlib
RUN apt-get update && apt-get install -y \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create directory for face recognition data
RUN mkdir -p known_faces

# Expose port for FastAPI
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
