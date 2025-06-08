import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import os
import glob
import pickle
import replicate
from datetime import datetime
from google.cloud import bigquery


# ---- CONFIG ----
project_id = "eccbd-final-project"
bq_table = "seattle_parking.predictions"
model_bundle_path = "/Users/sanjaydv/Downloads/model_bundle.pkl"
csv_file_path = "./sample_data.csv"

# ---- Load Model Preprocessing Bundle ----
with open(model_bundle_path, "rb") as f:
    bundle = pickle.load(f)
encoder = bundle["encoder"]
scaler = bundle["scaler"]
categorical_cols = bundle["categorical_cols"]
numeric_cols = bundle["numeric_cols"]
extra_feats = bundle["extra_feats"]

# ---- Setup BigQuery and Replicate ----
bq_client = bigquery.Client(project=project_id)
replicate_client = replicate.Client(api_token=os.getenv("REPLICATE_API_TOKEN"))
deployment = replicate.deployments.get("sanjaydv/predictor")

# ---- Streamlit UI ----
st.set_page_config(page_title="Parking Prediction Dashboard", layout="wide")
st.title("🚗 Parking Pricing Prediction: CSV ➝ Replicate ➝ BigQuery")

st.auto_refresh


def preprocess_row(row):
    X_cat = encoder.transform([[row[col] for col in categorical_cols]])
    X_num = scaler.transform([[row[col] for col in numeric_cols]])
    X_extra = np.array([[row[col] for col in extra_feats]])
    return np.hstack([X_cat, X_num, X_extra])[0].tolist()

def start_csv_stream_and_predict():
    df = pd.read_csv(csv_file_path)

    # Feature engineering
    df['timestamp'] = pd.to_datetime(df['OccupancyDateTime'], errors='coerce')
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek + 1
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['dayofweek'].isin([1, 7]).astype(int)
    df['is_peak_hour'] = df['hour'].between(8, 11) | df['hour'].between(16, 19)
    df['is_peak_hour'] = df['is_peak_hour'].astype(int)
    df['occupancy_rate'] = df['PaidOccupancy'] / df['ParkingSpaceCount']

    # Initialize session state tracker
    if "last_processed_index" not in st.session_state:
        st.session_state.last_processed_index = 0

    for i in range(st.session_state.last_processed_index, len(df)):
        if not st.session_state.streaming:
            break

        row = df.iloc[i]
        try:
            features = preprocess_row(row)
            prediction = deployment.predictions.create(input={"features": features})
            prediction.wait()
            base_price = prediction.output

            occupancy = row["occupancy_rate"]
            is_peak = row["is_peak_hour"]

            if occupancy > 0.8 and is_peak:
                surged = base_price * 0.5
                surge_tag = "high-surge"
            elif occupancy > 0.6:
                surged = base_price * 0.2
                surge_tag = "surge"
            else:
                surged = 0.0
                surge_tag = "no surge"

            row_to_insert = {
                "location": str(row.get("BlockfaceName")),
                "block_id": str(row.get("PaidParkingBlockface")),
                "hour": int(row.get("hour")),
                "day_of_week": str(int(row.get("dayofweek"))),
                "is_holiday": False,
                "prediction": float(base_price),
                "SurgedPrice": float(surged),
                "TotalPrice": float(base_price + surged),
                "surge_applied": str(surge_tag),
                "timestamp": datetime.utcnow().isoformat()
            }

            errors = bq_client.insert_rows_json(bq_table, [row_to_insert])
            if errors:
                st.error(f"❌ BQ insert error for row {i}: {errors}")
            else:
                st.success(f"✅ Row {i} inserted with prediction: ${base_price:.2f}")

            # Update last processed index
            st.session_state.last_processed_index = i + 1
            time.sleep(1)

        except Exception as e:
            st.error(f"❌ Row {i} failed: {e}")

# ---- SESSION STATE FOR STREAMING ----
if "streaming" not in st.session_state:
    st.session_state.streaming = False

# ---- Streaming Controls ----
st.header("⚙ Streaming Control")

col1, col2 = st.columns(2)
with col1:
    if st.button("🚀 Start CSV Streaming and Prediction", disabled=st.session_state.streaming):
        st.session_state.streaming = True

with col2:
    if st.button("🛑 Stop Streaming", disabled=not st.session_state.streaming):
        st.session_state.streaming = False

# ---- Streaming Execution ----
if st.session_state.streaming:
    with st.spinner("Streaming in progress..."):
        start_csv_stream_and_predict()

# ---- Live Dashboard ----
st.header("📈 Live Dashboard: BigQuery Stats")

def fetch_bq_data():
    query = f"""
        SELECT * FROM {project_id}.{bq_table}
        ORDER BY timestamp DESC
        LIMIT 500
    """
    return bq_client.query(query).to_dataframe()

placeholder = st.empty()

while True:
    with placeholder.container():
        df_bq = fetch_bq_data()

        if df_bq.empty:
            st.info("No data available yet.")
        else:
            # Metrics
            st.subheader("📊 Aggregated Stats")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔢 Rows Inserted", len(df_bq))
            col2.metric("💵 Avg Base Price", f"${df_bq['prediction'].mean():.2f}")
            col3.metric("🔥 Avg Surged Price", f"${df_bq['SurgedPrice'].mean():.2f}")

            # Surge Tag Bar Chart
            st.subheader("🚦 Surge Tag Distribution")
            st.bar_chart(df_bq['surge_applied'].value_counts())

            # Recent Data
            st.subheader("🕒 Most Recent Rows")
            st.dataframe(df_bq.head(5), use_container_width=True)

            df_bq['timestamp'] = pd.to_datetime(df_bq['timestamp'])
            df_bq_sorted = df_bq.sort_values("timestamp")
            st.subheader("📈 Prediction Over Time")
            st.line_chart(df_bq_sorted.set_index('timestamp')[['prediction', 'SurgedPrice']])

            st.subheader("⏰ Surge Tags by Hour of Day")
            surge_by_hour = df_bq.groupby(['hour', 'surge_applied']).size().unstack(fill_value=0)
            st.bar_chart(surge_by_hour)

            st.subheader("📅 Avg Total Price by Day of Week")
            price_by_day = df_bq.groupby('day_of_week')['TotalPrice'].mean()
            st.bar_chart(price_by_day)

            st.subheader("📍 Top 5 Locations by Avg Total Price")
            top_locs = df_bq.groupby('location')['TotalPrice'].mean().nlargest(5)
            st.bar_chart(top_locs)

    time.sleep(20)
