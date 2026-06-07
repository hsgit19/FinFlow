content = '''# src/prompts.py
# FinFlow Prompt Library

def count_tokens_approx(text):
    return int(len(text.split()) / 0.75)

SPENDING_INSIGHT_V1 = """
You are a financial advisor. Look at this user spending data
and give them some advice about their finances.
Spending data: {spending_data}
Forecast: {forecast_data}
Give me some recommendations.
"""

SPENDING_INSIGHT_V2 = """You are FinFlow, a personal finance AI.
User profile: savings_rate={savings_rate}%, top_category={top_category}
Monthly spending vs budget:
{spending_summary}
6-month forecast: {forecast_summary}
Generate exactly 3 recommendations. Format:
[Category] | [Issue] | [Action] | [Est. Monthly Saving $]
Max 120 words total."""

def get_spending_prompt(spending_data, forecast_data, savings_rate, top_category, spending_summary, forecast_summary, version="v2"):
    if version == "v1":
        return SPENDING_INSIGHT_V1.format(spending_data=spending_data, forecast_data=forecast_data)
    else:
        return SPENDING_INSIGHT_V2.format(savings_rate=savings_rate, top_category=top_category, spending_summary=spending_summary, forecast_summary=forecast_summary)

FRAUD_ALERT_V1 = """
You are a fraud detection system. This transaction was flagged as suspicious.
Transaction details: {transaction}
User history: {user_history}
Please explain this to the user.
"""

FRAUD_ALERT_V2 = """You are FinFlow fraud analyst.
Flagged transaction:
- Amount: ${amount} (your avg: ${user_avg})
- Category: {category} (fraud rate: {category_fraud_rate}%)
- Time: {hour}:00 ({time_label})
- Distance from home: {distance:.0f} miles
- Anomaly score: {anomaly_score:.2f}/1.0
Think step by step:
1. Amount signal: {amount_signal}
2. Location signal: {location_signal}
3. Time signal: {time_signal}
Write a 2-sentence alert for the user explaining the top 2 risk factors. Then suggest 1 action. Max 60 words."""

def get_fraud_prompt(transaction, user_history, amount, user_avg, category, category_fraud_rate, hour, time_label, distance, anomaly_score, amount_signal, location_signal, time_signal, version="v2"):
    if version == "v1":
        return FRAUD_ALERT_V1.format(transaction=transaction, user_history=user_history)
    else:
        return FRAUD_ALERT_V2.format(amount=amount, user_avg=user_avg, category=category, category_fraud_rate=category_fraud_rate, hour=hour, time_label=time_label, distance=distance, anomaly_score=anomaly_score, amount_signal=amount_signal, location_signal=location_signal, time_signal=time_signal)

CATEGORY_CLASSIFIER_V1 = """
What category does this merchant belong to?
Merchant: {merchant}
Categories: Food, Rent, Travel, Utilities, Entertainment, Healthcare, Education, Savings, Others
Answer:
"""

CATEGORY_CLASSIFIER_V2 = """Classify merchant into one category.
Categories: Food|Rent|Travel|Utilities|Entertainment|Healthcare|Education|Savings|Others
Examples:
WHOLEFDS MKT 10417 -> Food
DELTA AIR 0062341 -> Travel
NETFLIX.COM -> Entertainment
CVS PHARMACY 4521 -> Healthcare
SHELL OIL 57234 -> Travel
SPOTIFY USA -> Entertainment
RENT PAYMENT -> Rent
CON EDISON -> Utilities
Merchant: {merchant}
Category:"""

def get_classifier_prompt(merchant, version="v2"):
    if version == "v1":
        return CATEGORY_CLASSIFIER_V1.format(merchant=merchant)
    else:
        return CATEGORY_CLASSIFIER_V2.format(merchant=merchant)

def generate_token_report():
    sample_spending = "Food: $450, Rent: $1200, Entertainment: $300"
    sample_forecast = "Next month: $2100 predicted"
    sample_transaction = "Amount: $850, Merchant: AMZN, Time: 2am"
    sample_history = "Average transaction: $67, Usually shops locally"
    sample_merchant = "WHOLEFDS MKT 10417"
    prompts = {
        "Spending V1": SPENDING_INSIGHT_V1.format(spending_data=sample_spending, forecast_data=sample_forecast),
        "Spending V2": SPENDING_INSIGHT_V2.format(savings_rate=59.6, top_category="Food", spending_summary=sample_spending, forecast_summary=sample_forecast),
        "Fraud V1": FRAUD_ALERT_V1.format(transaction=sample_transaction, user_history=sample_history),
        "Fraud V2": FRAUD_ALERT_V2.format(amount=850, user_avg=67, category="shopping_net", category_fraud_rate=1.75, hour=2, time_label="night", distance=450, anomaly_score=0.89, amount_signal="12.7x above average", location_signal="450 miles from home", time_signal="2am unusual hour"),
        "Classifier V1": CATEGORY_CLASSIFIER_V1.format(merchant=sample_merchant),
        "Classifier V2": CATEGORY_CLASSIFIER_V2.format(merchant=sample_merchant)
    }
    report = []
    for name, prompt in prompts.items():
        tokens = count_tokens_approx(prompt)
        report.append({"prompt": name, "tokens": tokens})
    return report
'''

with open('src/prompts.py', 'w') as f:
    f.write(content)

import os
print(f"File written: {os.path.getsize('src/prompts.py')} bytes")