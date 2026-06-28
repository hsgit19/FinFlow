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


# Expose port 8501 for Streamlit
EXPOSE 8501

# Command to run when container starts
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]