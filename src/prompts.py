# src/prompts.py
# FinFlow Prompt Library

def count_tokens_approx(text):
    return int(len(text.split()) / 0.75)

SPENDING_INSIGHT_V1 = 'You are a financial advisor. Spending data: {spending_data}. Forecast: {forecast_data}. Give recommendations.'

SPENDING_INSIGHT_V2 = 'You are FinFlow AI. savings_rate={savings_rate}%, top_category={top_category}. Spending: {spending_summary}. Forecast: {forecast_summary}. Give 3 recommendations: [Category] | [Issue] | [Action] | [Saving]. Max 120 words.'

def get_spending_prompt(spending_data, forecast_data, savings_rate, top_category, spending_summary, forecast_summary, version='v2'):
    if version == 'v1':
        return SPENDING_INSIGHT_V1.format(spending_data=spending_data, forecast_data=forecast_data)
    return SPENDING_INSIGHT_V2.format(savings_rate=savings_rate, top_category=top_category, spending_summary=spending_summary, forecast_summary=forecast_summary)

FRAUD_ALERT_V1 = 'You are fraud detection. Transaction: {transaction}. History: {user_history}. Explain to user.'

FRAUD_ALERT_V2 = 'FinFlow fraud analyst. Amount: {amount} vs avg {user_avg}. Category: {category} fraud rate {category_fraud_rate}%. Time: {hour}:00 {time_label}. Distance: {distance:.0f} miles. Score: {anomaly_score:.2f}. Signals: {amount_signal}, {location_signal}, {time_signal}. Write 2-sentence alert and 1 action. Max 60 words.'

def get_fraud_prompt(transaction, user_history, amount, user_avg, category, category_fraud_rate, hour, time_label, distance, anomaly_score, amount_signal, location_signal, time_signal, version='v2'):
    if version == 'v1':
        return FRAUD_ALERT_V1.format(transaction=transaction, user_history=user_history)
    return FRAUD_ALERT_V2.format(amount=amount, user_avg=user_avg, category=category, category_fraud_rate=category_fraud_rate, hour=hour, time_label=time_label, distance=distance, anomaly_score=anomaly_score, amount_signal=amount_signal, location_signal=location_signal, time_signal=time_signal)

CATEGORY_CLASSIFIER_V1 = 'What category is this merchant? Merchant: {merchant}. Categories: Food, Rent, Travel, Utilities, Entertainment, Healthcare, Education, Savings, Others.'

CATEGORY_CLASSIFIER_V2 = 'Classify merchant. Categories: Food|Rent|Travel|Utilities|Entertainment|Healthcare|Education|Savings|Others. Examples: WHOLEFDS->Food, DELTA AIR->Travel, NETFLIX->Entertainment, CVS->Healthcare. Merchant: {merchant}. Category:'

def get_classifier_prompt(merchant, version='v2'):
    if version == 'v1':
        return CATEGORY_CLASSIFIER_V1.format(merchant=merchant)
    return CATEGORY_CLASSIFIER_V2.format(merchant=merchant)

def generate_token_report():
    prompts = {
        'Spending V1': SPENDING_INSIGHT_V1.format(spending_data='Food: 450', forecast_data='2100'),
        'Spending V2': SPENDING_INSIGHT_V2.format(savings_rate=59.6, top_category='Food', spending_summary='Food: 450', forecast_summary='2100'),
        'Fraud V1': FRAUD_ALERT_V1.format(transaction='850 AMZN 2am', user_history='avg 67'),
        'Fraud V2': FRAUD_ALERT_V2.format(amount=850, user_avg=67, category='shopping_net', category_fraud_rate=1.75, hour=2, time_label='night', distance=450, anomaly_score=0.89, amount_signal='12x avg', location_signal='450 miles', time_signal='2am'),
        'Classifier V1': CATEGORY_CLASSIFIER_V1.format(merchant='WHOLEFDS MKT'),
        'Classifier V2': CATEGORY_CLASSIFIER_V2.format(merchant='WHOLEFDS MKT')
    }
    return [{'prompt': k, 'tokens': count_tokens_approx(v)} for k, v in prompts.items()]
