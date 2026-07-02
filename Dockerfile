# Start with Python 3.11 - stable version for all our packages
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Copy requirements first (Docker caching optimization)
COPY requirements.txt .

# Install all Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files into the container
COPY dashboard/ ./dashboard/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY start.sh .

# Make the startup script executable
RUN chmod +x start.sh

# Expose ports for Streamlit and FastAPI
EXPOSE 8501 8000

# Run both services via startup script
CMD ["./start.sh"]