"""
One-time migration script: loads FinFlow transaction data from S3
into the RDS PostgreSQL 'transactions' table.
"""
import os
import boto3
import pandas as pd
import psycopg2
from io import StringIO

# --- Configuration ---
S3_BUCKET = 'finflow-data-152125349659'
S3_KEY = 'budgetwise_usd_converted.csv'

RDS_HOST = 'database-1.c8t4u68s25xa.us-east-1.rds.amazonaws.com'
RDS_PORT = 5432
RDS_DB = 'finflow'
RDS_USER = 'postgres'
RDS_PASSWORD = os.environ.get('myFinflowRDS') 

if not RDS_PASSWORD:
    raise ValueError("RDS_PASSWORD environment variable is not set")

def main():
    print("Reading data from S3...")
    s3 = boto3.client('s3', region_name='us-east-1')
    obj = s3.get_object(Bucket=S3_BUCKET, Key=S3_KEY)
    df = pd.read_csv(obj['Body'], parse_dates=['date'])
    print(f"Loaded {len(df)} rows from S3")

    print("Connecting to RDS...")
    conn = psycopg2.connect(
        host=RDS_HOST, port=RDS_PORT, dbname=RDS_DB,
        user=RDS_USER, password=RDS_PASSWORD
    )
    cur = conn.cursor()

    print("Inserting rows...")
    for _, row in df.iterrows():
        cur.execute(
            """INSERT INTO transactions (date, category, amount, transaction_type)
               VALUES (%s, %s, %s, %s)""",
            (row['date'], row.get('category'), row['amount'], row['transaction_type'])
        )

    conn.commit()
    cur.close()
    conn.close()
    print(f"Done! Inserted {len(df)} rows into RDS.")

if __name__ == '__main__':
    main()