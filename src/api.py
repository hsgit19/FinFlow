"""
src/api.py
FinFlow FastAPI — ML model serving API
Exposes fraud detection, forecasting, and summary endpoints
"""

import os
import json
import boto3
import pickle
import joblib
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import psycopg2

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
S3_BUCKET   = "finflow-data-152125349659"
RDS_HOST    = "database-1.c8t4u68s25xa.us-east-1.rds.amazonaws.com"
RDS_PORT    = 5432
RDS_DB      = "finflow"
RDS_USER    = "postgres"
RDS_PASSWORD = os.environ.get("RDS_PASSWORD")

# ─────────────────────────────────────────────
# Model store (loaded once at startup)
# ─────────────────────────────────────────────
models = {}

def load_models():
    """Load all ML models from disk into memory once at startup."""
    base = os.path.join(os.path.dirname(__file__), "models")
    models["lgb"]       = joblib.load(os.path.join(base, "lgb_fraud_model.pkl"))
    models["iso"]       = joblib.load(os.path.join(base, "isolation_forest.pkl"))
    models["prophet"]   = joblib.load(os.path.join(base, "prophet_forecaster.pkl"))
    models["le_cat"]    = joblib.load(os.path.join(base, "le_fraud_category.pkl"))
    with open(os.path.join(base, "fraud_feature_cols.pkl"), "rb") as f:
        models["feature_cols"] = pickle.load(f)
    print("✅ Models loaded successfully")

# ─────────────────────────────────────────────
# Lifespan — runs on startup and shutdown
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    print("API shutting down")

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="FinFlow API",
    description="ML model serving API for FinFlow personal finance intelligence",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────
class FraudRequest(BaseModel):
    amount: float
    category: str
    transaction_type: str   # "purchase", "transfer", etc.
    hour: int = 12          # hour of day (0-23)

class FraudResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    risk_level: str         # "Low", "Medium", "High"
    model_version: str = "lgb_v1"

class ForecastPoint(BaseModel):
    date: str
    predicted_spend: float
    lower_bound: float
    upper_bound: float

class SummaryResponse(BaseModel):
    total_transactions: int
    total_expenses: float
    total_income: float
    top_category: str
    date_range_start: str
    date_range_end: str
    source: str = "RDS PostgreSQL"

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def get_rds_connection():
    """Return a live psycopg2 connection to RDS."""
    if not RDS_PASSWORD:
        raise HTTPException(status_code=500, detail="RDS_PASSWORD environment variable not set")
    return psycopg2.connect(
        host=RDS_HOST, port=RDS_PORT, dbname=RDS_DB,
        user=RDS_USER, password=RDS_PASSWORD, sslmode="require"
    )

def read_s3_csv(key: str, parse_dates=None) -> pd.DataFrame:
    """Read a CSV from S3 into a DataFrame."""
    s3 = boto3.client("s3", region_name="us-east-1")
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    return pd.read_csv(obj["Body"], parse_dates=parse_dates)

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Confirms the API is running and models are loaded."""
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/predict/fraud", response_model=FraudResponse)
def predict_fraud(req: FraudRequest):
    """
    Takes transaction details and returns a fraud probability score.
    Uses the LightGBM fraud classifier trained on 1.3M transactions.
    """
    try:
        # Encode category using the saved label encoder
        try:
            cat_encoded = models["le_cat"].transform([req.category])[0]
        except ValueError:
            # Unknown category — use -1 as fallback
            cat_encoded = -1

        # Build feature vector matching training columns
        feature_dict = {
            "amount": req.amount,
            "category_encoded": cat_encoded,
            "hour": req.hour,
        }

        # Fill any remaining expected columns with 0
        row = {col: feature_dict.get(col, 0) for col in models["feature_cols"]}
        X = pd.DataFrame([row])[models["feature_cols"]]

        prob = float(models["lgb"].predict_proba(X)[0][1])
        is_fraud = prob >= 0.5

        if prob < 0.3:
            risk = "Low"
        elif prob < 0.7:
            risk = "Medium"
        else:
            risk = "High"

        return FraudResponse(
            fraud_probability=round(prob, 4),
            is_fraud=is_fraud,
            risk_level=risk
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/forecast", response_model=list[ForecastPoint])
def get_forecast(periods: int = 30):
    """
    Returns spending forecast for the next N days (default 30).
    Uses the Prophet model trained on historical BudgetWise data.
    """
    try:
        future = models["prophet"].make_future_dataframe(periods=periods)
        forecast = models["prophet"].predict(future)

        # Return only future dates
        result = forecast.tail(periods)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

        return [
            ForecastPoint(
                date=str(row["ds"].date()),
                predicted_spend=round(max(row["yhat"], 0), 2),
                lower_bound=round(max(row["yhat_lower"], 0), 2),
                upper_bound=round(max(row["yhat_upper"], 0), 2),
            )
            for _, row in result.iterrows()
        ]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/summary", response_model=SummaryResponse)
def get_summary():
    """
    Returns key financial summary stats pulled live from RDS PostgreSQL.
    """
    try:
        conn = get_rds_connection()
        cur = conn.cursor()

        cur.execute("SELECT COUNT(*) FROM transactions")
        total = cur.fetchone()[0]

        cur.execute("SELECT COALESCE(SUM(amount),0) FROM transactions WHERE transaction_type = 'Expense'")
        expenses = float(cur.fetchone()[0])

        cur.execute("SELECT COALESCE(SUM(amount),0) FROM transactions WHERE transaction_type = 'Income'")
        income = float(cur.fetchone()[0])

        cur.execute("""
            SELECT category FROM transactions
            WHERE transaction_type = 'Expense'
            GROUP BY category ORDER BY SUM(amount) DESC LIMIT 1
        """)
        top_cat = cur.fetchone()
        top_cat = top_cat[0] if top_cat else "N/A"

        cur.execute("SELECT MIN(date), MAX(date) FROM transactions")
        date_min, date_max = cur.fetchone()

        cur.close()
        conn.close()

        return SummaryResponse(
            total_transactions=total,
            total_expenses=round(expenses, 2),
            total_income=round(income, 2),
            top_category=top_cat,
            date_range_start=str(date_min),
            date_range_end=str(date_max)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))