# Seattle Dynamic Parking Management System (SDPMS)

A real-time machine learning system for predicting dynamic parking prices based on occupancy patterns in Seattle. This project leverages historical data, surge pricing logic, and cloud infrastructure to optimize curb space utilization and reduce congestion. Designed for both analytical insight and production deployment, the system includes a model training pipeline and an interactive Streamlit dashboard with BigQuery integration.

---

## Objective

The goal of this project is to support the Seattle Department of Transportation (SDOT) with a dynamic pricing system that:

- Predicts the base hourly parking rate based on historical patterns
- Applies real-time surge pricing logic during high-demand periods
- Ingests data continuously from CSV or live sources
- Streams enriched results to BigQuery for centralized analysis
- Offers an interactive dashboard to monitor prices and trends

---

## Key Features

- **Model Training Pipeline**: Includes feature engineering and training a Linear Regression model.
- **Surge Pricing Logic**: Applies a pricing premium based on occupancy and time-of-day.
- **Real-Time Dashboard**: Built in Streamlit with interactive charts and controls.
- **BigQuery Integration**: Streams live predictions and metadata into a structured cloud data warehouse.
- **Replicate Deployment**: Model is deployed to an API endpoint for scalable, secure inference.

---

## 🛠️ Tech Stack

| Layer                | Tools / Services                          |
|---------------------|-------------------------------------------|
| Language            | Python                                    |
| ML Framework        | scikit-learn                              |
| Visualization       | Streamlit                                 |
| Model Deployment    | Replicate API                             |
| Data Streaming      | Streamlit session + CSV simulation        |
| Cloud Storage       | Google Cloud BigQuery                     |
| Data Processing     | Pandas, NumPy                             |
| Serialization       | Pickle (`model_bundle.pkl`)               |



---

## 🧠 Model Development

### Dataset
- **Source**: Seattle Open Data Portal
- **Size**: ~1 million rows sampled
- **Key Fields**: Occupancy datetime, parking rate, space count, block ID, location metadata

### Feature Engineering
- Time-based: `hour`, `dayofweek`, `month`, `is_weekend`, `is_peak_hour`
- Derived: `occupancy_rate = PaidOccupancy / ParkingSpaceCount`
- One-hot encoding of location-based categorical columns
- Scaling of numerical features

### Model
- Algorithm: `LinearRegression` from scikit-learn
- Evaluation: (R² ≈ moderate, RMSE ≈ 4, MAE ≈ 1.1)
- Output: Base price prediction

### Surge Logic
```python
if occupancy_rate > 0.8 and is_peak_hour:
    surge_price = base_price * 0.5  # High surge
elif occupancy_rate > 0.6:
    surge_price = base_price * 0.2  # Moderate surge
else:
    surge_price = 0.0               # No surge


📈 Future Improvements
- Switch to more robust models (e.g., XGBoost, ElasticNet)
- Use GCP Pub/Sub or Cloud Functions for true real-time ingestion
- Add authentication layer for app access control
- Enable map-based visualization of pricing zones