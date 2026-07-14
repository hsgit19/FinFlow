"""
src/api.py
FinFlow FastAPI — ML model serving API
Exposes fraud detection, forecasting, summary, and RAG-based Q&A endpoints
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

# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────
S3_BUCKET   = "finflow-data-152125349659"
RDS_HOST    = "database-1.c8t4u68s25xa.us-east-1.rds.amazonaws.com"
RDS_PORT    = 5432
RDS_DB      = "finflow"
RDS_USER    = "postgres"
RDS_PASSWORD = os.environ.get("RDS_PASSWORD")

EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
CHAT_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
bedrock_runtime = boto3.client("bedrock-runtime", region_name="us-east-1")

# ─────────────────────────────────────────────────
# Model store (loaded once at startup)
# ─────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────
# Lifespan — runs on startup and shutdown
# ─────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield
    print("API shutting down")

# ─────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────
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

class AskRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    type: str
    text: str
    similarity: float

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]

# ─────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────
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

def get_embedding(text: str) -> list:
    """Convert text into a 1024-dim embedding using Bedrock Titan."""
    body = json.dumps({"inputText": text})
    response = bedrock_runtime.invoke_model(
        modelId=EMBED_MODEL_ID, body=body,
        contentType="application/json", accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["embedding"]

def retrieve_relevant_chunks(conn, query_embedding: list, top_k: int = 5) -> list:
    """Find the top_k most similar chunks using pgvector cosine similarity."""
    cur = conn.cursor()
    embedding_str = str(query_embedding)
    cur.execute(
        """
        SELECT content_type, content_text, 1 - (embedding <=> %s::vector) AS similarity
        FROM finflow_embeddings
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (embedding_str, embedding_str, top_k)
    )
    rows = cur.fetchall()
    cur.close()
    return [{"type": r[0], "text": r[1], "similarity": float(r[2])} for r in rows]

def generate_grounded_answer(question: str, chunks: list) -> str:
    """Send question + retrieved context to Claude Haiku for a grounded answer."""
    context = "\n".join([f"- {c['text']}" for c in chunks])
    prompt = f"""You are FinFlow's financial assistant. Answer the user's question using ONLY the context below. If the context doesn't contain enough information to answer, say so honestly rather than guessing.

Context:
{context}

Question: {question}

Answer:"""

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 500,
        "messages": [{"role": "user", "content": prompt}]
    })

    response = bedrock_runtime.invoke_model(
        modelId=CHAT_MODEL_ID, body=body,
        contentType="application/json", accept="application/json",
    )
    result = json.loads(response["body"].read())
    return result["content"][0]["text"]

# ─────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────
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
        try:
            cat_encoded = models["le_cat"].transform([req.category])[0]
        except ValueError:
            cat_encoded = -1

        feature_dict = {
            "amount": req.amount,
            "category_encoded": cat_encoded,
            "hour": req.hour,
        }

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


@app.post("/ask", response_model=AskResponse)
def ask_finflow(req: AskRequest):
    """
    RAG endpoint — answers a natural language question about the user's
    financial data, grounded in stored transaction summaries and EDA insights.
    """
    try:
        conn = get_rds_connection()

        query_embedding = get_embedding(req.question)
        chunks = retrieve_relevant_chunks(conn, query_embedding, top_k=5)
        answer = generate_grounded_answer(req.question, chunks)

        conn.close()

        return AskResponse(
            question=req.question,
            answer=answer,
            sources=[SourceChunk(type=c["type"], text=c["text"], similarity=c["similarity"]) for c in chunks]
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))