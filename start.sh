#!/bin/bash
# Start FastAPI on port 8000
cd /app && uvicorn src.api:app --host 0.0.0.0 --port 8000 &
# Start Streamlit on port 8501
streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0