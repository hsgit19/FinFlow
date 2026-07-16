FinFlow — AI-Powered Personal Finance Intelligence Platform


Built by Harini Lingampelli | Deployed on AWS EC2



Live Demo: http://35.171.107.227:8501
API Docs: http://35.171.107.227:8000/docs


What is FinFlow?

FinFlow is an end-to-end capstone project demonstrating skills across data engineering, machine learning, cloud infrastructure, and applied LLM engineering. It ingests personal finance transaction data, detects fraud, forecasts spending, and answers natural-language questions about financial patterns — grounded in real data via a custom-built RAG pipeline — all served through a production-style AWS deployment with automated monitoring and CI/CD.


Live Dashboard

Visit the live Streamlit dashboard:

http://35.171.107.227:8501

5 pages:


Spending Overview — category breakdown, monthly trends, payment analysis
Fraud Detection — anomaly analysis across 1.3M transactions
AI Insights — Claude-powered recommendations via Amazon Bedrock
Forecasting — Prophet time-series 6-month spending forecast
Ask FinFlow — RAG-powered chat interface for natural-language Q&A over transaction data and insights



Tech Stack

LayerTechnologyData cleaningPython, pandas, fuzzy matchingML modelingLightGBM, Isolation Forest, ProphetLLM / AIAmazon Bedrock, Claude Haiku 4.5, Titan Text Embeddings V2RAG / Vector searchpgvector (PostgreSQL extension), cosine similarityAPI layerFastAPI, UvicornDashboardStreamlit, PlotlyBI DashboardPower BIDatabaseAmazon RDS (PostgreSQL 18.3)StorageAmazon S3ComputeAWS EC2, Docker, Amazon ECRServerlessAWS Lambda, Amazon EventBridgeMonitoringAmazon CloudWatch (alarms, dashboards, SNS alerting)CI/CDGitHub ActionsVersion controlGit, GitHub


Project Architecture

Raw Data (Kaggle + FRED API)
    ↓
ETL Pipeline (pandas + fuzzy matching)
    ↓
Feature Engineering (12 features)
    ↓
S3 (raw + processed data) ──→ Amazon RDS PostgreSQL (transactions, 14.5K rows)
    ↓
ML Models:
  - LightGBM Fraud Classifier (ROC-AUC: 0.9978)
  - Isolation Forest Anomaly Detector
  - Prophet Spending Forecaster
    ↓
Prompt Engineering (61.1% token reduction)
    ↓
FastAPI (containerized, Docker → ECR → EC2)
  ├── /health, /predict/fraud, /forecast, /summary
  └── /ask  ──→  RAG Pipeline:
                   Question → Titan Embedding → pgvector similarity search
                   → top-5 chunks → Claude Haiku (grounded generation)
    ↓
Streamlit Dashboard (5 pages, incl. Ask FinFlow chat) → AWS EC2 (Live)

Parallel automated pipeline:
EventBridge (daily cron) → Lambda (FX rate refresh) → S3
CloudWatch Alarms (Lambda errors, EC2 CPU, RDS CPU) → SNS → Email alerts
GitHub Actions → Docker build → ECR push → SSH deploy → EC2


RAG Pipeline (Retrieval-Augmented Generation)

Built a RAG system to answer natural-language questions about financial data, grounded in real transaction history and analytical insights — rather than relying on the LLM's general knowledge.


Vector storage: pgvector extension enabled directly on the existing Amazon RDS PostgreSQL instance, avoiding the need for separate vector database infrastructure
Embeddings: Amazon Titan Text Embeddings V2 (1024-dim) via Bedrock, applied to 206 chunks — 6 EDA insights + 200 aggregated transaction summaries (grouped by category, month, and transaction type)
Retrieval: cosine similarity search (<=> operator) over pgvector, returning top-5 most relevant chunks per query
Generation: Claude Haiku via Bedrock, with a grounding-constrained prompt engineered to explicitly acknowledge insufficient context rather than hallucinate an answer
API: exposed as a POST /ask endpoint in FastAPI, reusing the existing RDS connection pattern
UI: interactive chat interface in Streamlit (st.chat_message, session-state-backed conversation history), displaying retrieved source chunks and similarity scores alongside each answer for transparency


Architectural note: chose to extend the existing RDS instance with pgvector rather than provisioning a dedicated vector database service — reduces infrastructure surface area while still supporting production-grade similarity search.


LLM Evaluation Framework

Built an automated evaluation pipeline to systematically score RAG output quality, using an LLM-as-judge methodology — a widely-used technique for evaluating generative systems where no simple ground-truth metric exists.


Test set: 10 questions spanning direct-match, ambiguous, and adversarial/out-of-domain cases (including a deliberately unrelated control question)
Metrics scored (1-5 scale, via Claude Haiku as judge):

Groundedness — is every claim in the answer actually supported by the retrieved source chunks?
Relevance — does the answer actually address the question asked?
Appropriate refusal (boolean) — when context was insufficient, did the system correctly acknowledge that rather than guess?



Storage: results persisted to a dedicated eval_results table in RDS, keyed by run_id, supporting comparison across multiple evaluation runs over time
Result: 100% groundedness maintained across all 10 test questions, including cases with insufficient or entirely out-of-domain context (e.g., correctly declining to answer "what color should I paint my kitchen")



Key Results

ModelMetricScoreLightGBM Fraud ClassifierROC-AUC0.9978LightGBM Fraud ClassifierRecall97.7%Isolation ForestRecall16.1%Prophet ForecasterTraining months48Prompt optimizationToken reduction61.1%RAG pipelineGroundedness (10-question eval)100%


Datasets

DatasetSourceRowsPurposeBudgetWiseKaggle15,836Spending analysis + forecastingFraud DetectionKaggle (Sparkov)1,296,675Anomaly detectionPersonal FinanceKaggle806Category classificationFRED Economic DataFederal Reserve API—External regressors


Prompt Engineering

Three prompt types built with token optimization:

PromptV1 TokensV2 TokensReductionSpending insight1084855.6%Fraud alert1064260.4%Merchant classifier842669.0%Total29811661.1%

Techniques used: few-shot prompting, chain-of-thought, output format specification, token budget enforcement. The RAG grounding prompt applies a related principle — explicit constraint enforcement — to prevent hallucination rather than reduce token count.


Data Engineering Highlights


Currency conversion: Live exchange rates via ExchangeRate-API, refreshed daily and automated end-to-end (see Automated Infrastructure below)
Fuzzy category matching: Reduced 184 misspelled categories to 13 clean categories using thefuzz library
Mixed date formats: Handled 5 different date formats using dateutil flexible parser
Feature engineering: 12 features for spending model, 8 features for fraud model
Database migration: Migrated transaction storage from local/EC2 disk to Amazon S3, then to Amazon RDS PostgreSQL for structured querying via FastAPI



Automated Infrastructure


Daily FX rate refresh: finflow-fx-refresh Lambda function, triggered daily at 00:00 UTC via an EventBridge scheduled rule (cron(0 0 * * ? *)), fetches live exchange rates and writes to S3 — fully serverless, no manual refresh needed
Monitoring & alerting: 3 CloudWatch alarms (Lambda errors, EC2 high CPU, RDS high CPU) wired to an SNS topic with email notification, plus a FinFlow-Overview CloudWatch Dashboard visualizing Lambda invocations/errors, EC2 CPU, and RDS CPU in one view
CI/CD: GitHub Actions pipeline — on push to main, builds a Docker image, pushes to Amazon ECR, then SSHes into EC2 to pull and redeploy, with automated container cleanup to prevent stale-image and port-conflict issues



Known Limitations

Currency normalization: Dataset contains mixed USD and INR values. Normalized using live FX rates via ExchangeRate-API with full auditability (original currency, rate used, conversion date stored). Now refreshed automatically daily via the Lambda + EventBridge pipeline described above.

Forecast data: Model trained on 2019-2022 data. In production, Prophet would retrain monthly on new transaction data via a scheduled Lambda, following the same pattern already built for FX rate refreshing.

Instance sizing under concurrent AI load: The EC2 instance (t2.micro/t3.micro, burstable class) can experience CPU credit exhaustion under sustained concurrent load — e.g., running Streamlit, FastAPI, and multiple simultaneous Bedrock calls (embedding + generation) at once. This was directly observed and diagnosed via CloudWatch's CPU Credit Balance metric during RAG pipeline load testing. Production deployment would use a non-burstable instance class (e.g., t3.medium or higher, or a compute-optimized type) to guarantee sustained throughput under AI inference workloads.

RAG retrieval scope: The vector store currently covers 206 chunks (6 insights + 200 aggregated transaction summaries). Individual raw transactions are not embedded directly — retrieval operates over category/month-level aggregates, which is intentional for relevance but means highly granular single-transaction questions may return lower-confidence matches.


## Project Structure

```
FinFlow/
├── notebooks/
│   ├── 01_data_inspection.ipynb    # Data loading + health check
│   ├── 02_EDA.ipynb                # Exploratory data analysis
│   ├── 03_ml_modeling.ipynb        # Feature engineering + ML models
│   └── 04_llm_layer.ipynb          # Prompt engineering + Bedrock
├── dashboard/
│   └── app.py                      # Streamlit dashboard (5 pages, incl. Ask FinFlow)
├── src/
│   ├── __init__.py
│   ├── api.py                      # FastAPI app (5 endpoints, incl. /ask RAG endpoint)
│   ├── prompts.py                  # Prompt library (V1 + V2)
│   └── models/                     # Saved ML models (.pkl)
├── lamda/
│   └── fx_refresh.py               # Daily exchange rate refresh Lambda
├── scripts/
│   └── migrate_to_rds.py           # RDS migration utility
├── enable_pgvector.py              # One-time pgvector extension setup
├── generate_embeddings.py          # Embeds transactions + insights into pgvector
├── rag_query.py                    # Standalone RAG pipeline test script
├── run_evaluation.py               # LLM-as-judge evaluation framework
├── .github/
│   └── workflows/
│       └── deploy.yml              # CI/CD: build → ECR → EC2 deploy
├── docs/
│   └── FinFlow_Executive_Dashboard.pdf  # Power BI dashboard
├── Dockerfile
├── start.sh                        # Launches FastAPI + Streamlit in one container
├── requirements.txt
├── .env                            # API keys (never committed)
└── README.md
```

Setup Instructions

bash# Clone the repo
git clone https://github.com/hsgit19/FinFlow.git
cd FinFlow

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Add your API keys to .env
# AWS_ACCESS_KEY_ID=...
# AWS_SECRET_ACCESS_KEY=...
# FRED_API_KEY=...
# FX_API_KEY=...
# RDS_PASSWORD=...

# Run dashboard locally (RDS-dependent features require VPC access or a public RDS endpoint)
cd dashboard
streamlit run app.py

# Or run the API locally
uvicorn src.api:app --reload --port 8000

Note: the RDS instance is intentionally private (VPC-only access, not publicly accessible) for security. Local development against live RDS data requires running from within the same VPC (e.g., via EC2) — see scripts/ for examples of running setup/maintenance scripts directly inside the deployed container.


Author

Harini Lingampelli


GitHub: github.com/hsgit19
Project: github.com/hsgit19/FinFlow
Live demo: http://35.171.107.227:8501
API docs: http://35.171.107.227:8000/docs