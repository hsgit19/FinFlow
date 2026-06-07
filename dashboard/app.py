# dashboard/app.py
# FinFlow - AI-Powered Personal Finance Dashboard
# Built by Harini Lingampelli

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import boto3
import json
import os
import sys
from dotenv import load_dotenv

load_dotenv('../.env')

st.set_page_config(
    page_title="FinFlow - Personal Finance Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #1e2130;
    }
    .stMetric {
        background: #1e2130;
        border-radius: 8px;
        padding: 12px;
    }
    h1 { color: #4a9eff; }
    h2 { color: #ffffff; }
    h3 { color: #a8c8f0; }
    .metric-card {
        background: #1e2130;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #185FA5;
    }
    .alert-box {
        background: #2d1515;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #E24B4A;
        margin: 0.5rem 0;
        color: #ffaaaa;
    }
    .insight-box {
        background: #0d2018;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #1D9E75;
        margin: 0.5rem 0;
        color: #aaffcc;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data():
    bw = pd.read_csv('../Data/processed/budgetwise_usd_converted.csv',
                     parse_dates=['date'])
    fraud = pd.read_csv('../Data/processed/fraud_train_cleaned.csv',
                        parse_dates=['date'])
    forecast = pd.read_csv('../Data/processed/spending_forecast.csv',
                           parse_dates=['ds'])
    return bw, fraud, forecast

bw, fraud, forecast = load_data()

expenses = bw[bw['transaction_type'] == 'Expense'].copy()
income = bw[bw['transaction_type'] == 'Income'].copy()

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("""
<div style="background:#185FA5;padding:12px 16px;border-radius:8px;margin-bottom:8px">
    <div style="color:white;font-size:22px;font-weight:600;letter-spacing:1px">FinFlow</div>
    <div style="color:#a8c8f0;font-size:11px">AI-Powered Finance Intelligence</div>
    <div style="color:#ffffff;font-size:11px;margin-top:6px;opacity:0.7">by Harini Lingampelli</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Spending Overview", "Fraud Detection", "AI Insights", "Forecasting"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Data summary**")
st.sidebar.metric("Total transactions", f"{bw.shape[0]:,}")
st.sidebar.metric("Date range", "2019 - 2022")
st.sidebar.metric("Clean categories", "13")

# ============================================================
# PAGE 1 - SPENDING OVERVIEW
# ============================================================
if page == "Spending Overview":
    st.title("Spending Overview")
    st.markdown("Understanding where your money goes — by category, time, and payment method.")

    total_spend = expenses['amount'].sum()
    total_income = income['amount'].sum()
    savings_rate = ((total_income - total_spend) / total_income) * 100
    avg_transaction = expenses['amount'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total spent", f"${total_spend:,.0f}")
    with col2:
        st.metric("Total income", f"${total_income:,.0f}")
    with col3:
        st.metric("Savings rate", f"{savings_rate:.1f}%",
                  delta="Above average" if savings_rate > 20 else "Below average")
    with col4:
        st.metric("Avg transaction", f"${avg_transaction:,.0f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Total spend by category")
        cat_data = (expenses.groupby('category')['amount']
                    .sum()
                    .sort_values(ascending=True)
                    .reset_index())
        fig = px.bar(cat_data, x='amount', y='category',
                     orientation='h',
                     color='amount',
                     color_continuous_scale='Blues',
                     labels={'amount': 'Total spend ($)', 'category': ''})
        fig.update_layout(showlegend=False, coloraxis_showscale=False,
                          height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Spending distribution")
        fig = px.pie(cat_data, values='amount', names='category',
                     color_discrete_sequence=px.colors.qualitative.Set3)
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly spending trend")
    expenses['year_month'] = expenses['date'].dt.to_period('M').astype(str)
    monthly = (expenses.groupby('year_month')['amount']
               .sum().reset_index())
    monthly = monthly[monthly['amount'] > 10000]
    fig = px.line(monthly, x='year_month', y='amount',
                  markers=True,
                  labels={'amount': 'Total spend ($)', 'year_month': 'Month'},
                  color_discrete_sequence=['#185FA5'])
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(tickangle=45)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Spending by day of week")
        day_order = ['Monday','Tuesday','Wednesday',
                     'Thursday','Friday','Saturday','Sunday']
        expenses['day'] = expenses['date'].dt.day_name()
        day_data = (expenses.groupby('day')['amount']
                    .mean().reindex(day_order).reset_index())
        fig = px.bar(day_data, x='day', y='amount',
                     color='amount',
                     color_continuous_scale='Blues',
                     labels={'amount': 'Avg spend ($)', 'day': ''})
        fig.update_layout(height=300, showlegend=False,
                          coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Payment method breakdown")
        payment_data = (expenses.groupby('payment_mode')['amount']
                        .sum().sort_values(ascending=False)
                        .head(8).reset_index())
        fig = px.bar(payment_data, x='payment_mode', y='amount',
                     color='amount',
                     color_continuous_scale='Teal',
                     labels={'amount': 'Total spend ($)', 'payment_mode': ''})
        fig.update_layout(height=300, showlegend=False,
                          coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 - FRAUD DETECTION
# ============================================================
elif page == "Fraud Detection":
    st.title("Fraud Detection")
    st.markdown("Anomaly analysis across 1.3M credit card transactions.")

    total_fraud = fraud['is_fraud'].sum()
    fraud_rate = fraud['is_fraud'].mean() * 100
    avg_fraud_amt = fraud[fraud['is_fraud']==1]['amount'].mean()
    avg_normal_amt = fraud[fraud['is_fraud']==0]['amount'].mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total fraud cases", f"{total_fraud:,}")
    with col2:
        st.metric("Fraud rate", f"{fraud_rate:.2f}%")
    with col3:
        st.metric("Avg fraud amount", f"${avg_fraud_amt:.2f}")
    with col4:
        st.metric("Avg normal amount", f"${avg_normal_amt:.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fraud rate by category")
        fraud_cat = (fraud.groupby('category')
                     .agg(total=('is_fraud','count'),
                          fraudulent=('is_fraud','sum'))
                     .assign(fraud_rate=lambda x: x['fraudulent']/x['total']*100)
                     .sort_values('fraud_rate', ascending=True)
                     .reset_index())
        fig = px.bar(fraud_cat, x='fraud_rate', y='category',
                     orientation='h',
                     color='fraud_rate',
                     color_continuous_scale='Reds',
                     labels={'fraud_rate': 'Fraud rate (%)', 'category': ''})
        fig.update_layout(height=420, showlegend=False,
                          coloraxis_showscale=False,
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Fraud amount vs normal amount")
        fig = go.Figure()
        normal_sample = fraud[fraud['is_fraud']==0]['amount'].sample(5000, random_state=42)
        fraud_sample = fraud[fraud['is_fraud']==1]['amount']
        fig.add_trace(go.Histogram(x=normal_sample, name='Normal',
                                   nbinsx=50, opacity=0.7,
                                   marker_color='#185FA5'))
        fig.add_trace(go.Histogram(x=fraud_sample, name='Fraud',
                                   nbinsx=50, opacity=0.7,
                                   marker_color='#E24B4A'))
        fig.update_layout(barmode='overlay', height=420,
                          xaxis_title='Transaction amount ($)',
                          yaxis_title='Count',
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Fraud by hour of day")
    fraud['hour'] = fraud['date'].dt.hour
    fraud_hour = (fraud.groupby('hour')['is_fraud']
                  .mean() * 100).reset_index()
    fraud_hour.columns = ['hour', 'fraud_rate']
    fig = px.line(fraud_hour, x='hour', y='fraud_rate',
                  markers=True,
                  color_discrete_sequence=['#E24B4A'],
                  labels={'fraud_rate': 'Fraud rate (%)', 'hour': 'Hour of day'})
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sample flagged transactions")
    flagged = fraud[fraud['is_fraud']==1][
        ['date', 'merchant', 'category', 'amount', 'city', 'state']
    ].head(10)
    st.dataframe(flagged, use_container_width=True)

# ============================================================
# PAGE 3 - AI INSIGHTS
# ============================================================
elif page == "AI Insights":
    st.title("AI Insights")
    st.markdown("Claude-powered financial recommendations via Amazon Bedrock.")

    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1',
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
    )
    MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    expenses_local = bw[bw['transaction_type']=='Expense'].copy()
    cat_spend = (expenses_local.groupby('category')['amount']
                 .sum().sort_values(ascending=False).head(5))
    spending_summary = "\n".join([f"{c}: ${a:,.0f}" for c, a in cat_spend.items()])
    total_inc = income['amount'].sum()
    total_exp = expenses_local['amount'].sum()
    savings_rate = ((total_inc - total_exp) / total_inc) * 100
    top_cat = cat_spend.index[0]

    st.subheader("Your spending profile")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top 5 spending categories:**")
        for cat, amt in cat_spend.items():
            st.markdown(f"- **{cat}**: ${amt:,.0f}")
    with col2:
        st.metric("Savings rate", f"{savings_rate:.1f}%")
        st.metric("Top category", top_cat)

    st.markdown("---")
    st.subheader("AI-generated recommendations")

    if st.button("Generate insights with Claude"):
        with st.spinner("Calling Claude via Amazon Bedrock..."):
            prompt = f"""You are FinFlow AI. savings_rate={savings_rate:.1f}%, top_category={top_cat}
Monthly spending vs budget:
{spending_summary}
6-month forecast: Spending predicted to continue at current levels.
Generate exactly 3 recommendations. Format:
[Category] | [Issue] | [Action] | [Est. Monthly Saving $]
Max 120 words total."""

            response = bedrock.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 200,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            result = json.loads(response['body'].read())
            insight = result['content'][0]['text']
            tokens_used = result['usage']['input_tokens']

            st.markdown(f"""
            <div class="insight-box">
            {insight.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            st.caption(f"Tokens used: {tokens_used} | "
                      f"Model: Claude Haiku 4.5 | "
                      f"Cost: ~${tokens_used * 0.00000025:.6f}")

    st.markdown("---")
    st.subheader("Fraud alert explainer")
    st.markdown("Enter a transaction to check if it looks suspicious.")

    col1, col2, col3 = st.columns(3)
    with col1:
        txn_amount = st.number_input("Transaction amount ($)", value=500.0, min_value=0.0)
    with col2:
        txn_category = st.selectbox("Category", [
            'shopping_net', 'grocery_pos', 'misc_net',
            'gas_transport', 'entertainment', 'food_dining',
            'health_fitness', 'travel', 'home', 'kids_pets'
        ])
    with col3:
        txn_hour = st.slider("Hour of day", 0, 23, 2)

    if st.button("Analyze this transaction"):
        with st.spinner("Analyzing with Claude..."):
            user_avg = fraud[fraud['is_fraud']==0]['amount'].mean()
            cat_fraud_rate = fraud.groupby('category')['is_fraud'].mean().get(txn_category, 0.005) * 100
            time_label = "night" if (txn_hour >= 22 or txn_hour <= 6) else "business hours"
            amount_signal = f"{txn_amount/user_avg:.1f}x above user average"
            anomaly_score = min(0.99, (txn_amount/user_avg) * 0.1 +
                              (0.3 if txn_hour >= 22 or txn_hour <= 6 else 0))

            prompt = f"""FinFlow fraud analyst.
Flagged transaction:
- Amount: ${txn_amount:.2f} (user avg: ${user_avg:.2f})
- Category: {txn_category} (fraud rate: {cat_fraud_rate:.2f}%)
- Time: {txn_hour}:00 ({time_label})
- Anomaly score: {anomaly_score:.2f}/1.0
Think step by step:
1. Amount signal: {amount_signal}
2. Time signal: {txn_hour}:00 {time_label}
Write a 2-sentence alert explaining the top 2 risk factors. Then suggest 1 action. Max 60 words."""

            response = bedrock.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 150,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            result = json.loads(response['body'].read())
            alert = result['content'][0]['text']

            if anomaly_score > 0.5:
                st.markdown(f"""
                <div class="alert-box">
                <strong>Fraud alert</strong><br>
                {alert.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="insight-box">
                <strong>Transaction looks normal</strong><br>
                {alert.replace(chr(10), '<br>')}
                </div>
                """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Merchant category classifier")
    merchant_input = st.text_input(
        "Enter a merchant name to classify:",
        placeholder="e.g. WHOLEFDS MKT 10417"
    )
    if merchant_input:
        with st.spinner("Classifying..."):
            prompt = f"""Classify merchant into one category.
Categories: Food|Rent|Travel|Utilities|Entertainment|Healthcare|Education|Savings|Others
Examples: WHOLEFDS->Food, DELTA AIR->Travel, NETFLIX->Entertainment, CVS->Healthcare
Merchant: {merchant_input}
Category (one word only):"""
            response = bedrock.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps({
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": 5,
                    "messages": [{"role": "user", "content": prompt}]
                })
            )
            result = json.loads(response['body'].read())
            category = result['content'][0]['text'].strip().split()[0]
            st.success(f"**{merchant_input}** → **{category}**")

# ============================================================
# PAGE 4 - FORECASTING
# ============================================================
elif page == "Forecasting":
    st.title("Spending Forecast")
    st.markdown("Prophet time-series model predicting future spending patterns.")

    st.subheader("6-month spending forecast")

    historical = forecast[forecast['ds'] <= '2022-12-31'].copy()
    future = forecast[forecast['ds'] > '2022-12-31'].copy()
    future['yhat'] = future['yhat'].clip(lower=0)
    future['yhat_lower'] = future['yhat_lower'].clip(lower=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=historical['ds'], y=historical['yhat'],
        name='Historical fit',
        line=dict(color='#185FA5', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=future['ds'], y=future['yhat'],
        name='Forecast',
        line=dict(color='#E24B4A', width=2, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([future['ds'], future['ds'][::-1]]),
        y=pd.concat([future['yhat_upper'], future['yhat_lower'][::-1]]),
        fill='toself',
        fillcolor='rgba(226,75,74,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% confidence interval'
    ))
    fig.add_vline(x='2023-01-01', line_dash='dash',
                  line_color='gray', opacity=0.5,
                  annotation_text="Forecast start")
    fig.update_layout(height=450,
                      xaxis_title='Month',
                      yaxis_title='Total spend ($)',
                      margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast values — next 6 months")
    forecast_display = future[['ds', 'yhat',
                                'yhat_lower', 'yhat_upper']].copy()
    forecast_display.columns = ['Month', 'Predicted spend',
                                 'Lower bound', 'Upper bound']
    forecast_display['Month'] = forecast_display['Month'].dt.strftime('%B %Y')
    forecast_display['Predicted spend'] = forecast_display[
        'Predicted spend'].apply(lambda x: f"${x:,.0f}")
    forecast_display['Lower bound'] = forecast_display[
        'Lower bound'].apply(lambda x: f"${x:,.0f}")
    forecast_display['Upper bound'] = forecast_display[
        'Upper bound'].apply(lambda x: f"${x:,.0f}")
    st.dataframe(forecast_display, use_container_width=True)

    st.markdown("---")
    st.subheader("Model details")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model", "Prophet")
    with col2:
        st.metric("Training months", "48")
    with col3:
        st.metric("Forecast horizon", "6 months")
    with col4:
        st.metric("Seasonality mode", "Multiplicative")

    st.markdown("---")
    st.subheader("Trend component")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=forecast['ds'], y=forecast['trend'],
        name='Spending trend',
        line=dict(color='#1D9E75', width=2)
    ))
    fig.update_layout(height=300,
                      xaxis_title='Month',
                      yaxis_title='Trend value ($)',
                      margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)
    