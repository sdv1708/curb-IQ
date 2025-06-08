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

## Dataset

### Source
The dataset used for training and streaming is the **2020 Paid Parking Occupancy** dataset published by the Seattle Department of Transportation (SDOT). It is publicly available through the City of Seattle Open Data Portal:

- **Link**: [Seattle Parking Occupancy Dataset](https://catalog.data.gov/dataset/2020-paid-parking-occupancy)

---

### Description
This is a time-series geospatial dataset that captures detailed parking occupancy and pricing patterns across Seattle’s paid parking zones throughout the year 2020.

- **Format**: CSV (and streaming as batch slices)
- **Records**: ~1 million entries
- **Frequency**: 15-minute intervals per block
- **Granularity**: Blockface-level data (location + side of street)

---

### Key Fields

| Column               | Description                                                  |
|----------------------|--------------------------------------------------------------|
| `OccupancyDateTime`  | Timestamp of the occupancy reading                           |
| `PaidOccupancy`      | Number of paid parking spaces occupied                       |
| `ParkingSpaceCount`  | Total number of spaces available                             |
| `PaidParkingRate`    | Rate charged at the time                                     |
| `BlockfaceName`      | Street and direction of the parking block                    |
| `SideOfStreet`       | Which side of the street the blockface is located on         |
| `PaidParkingArea`    | Zone/neighborhood where the parking spot is located          |
| `ParkingTimeLimitCategory` | Time limit rules for that zone (e.g., 2h, 4h)         |
| `ParkingCategory`    | Type of parking area (e.g., commercial, mixed use)           |

---

### Feature Engineering

From the raw data, the following engineered features were derived:

- **Temporal Variables**:
  - `hour`, `dayofweek`, `month` from `OccupancyDateTime`
  - `is_weekend`: binary flag for weekend hours
  - `is_peak_hour`: flag for morning/evening rush hours

- **Occupancy Rate**:
  - `occupancy_rate = PaidOccupancy / ParkingSpaceCount`

These transformations allow the model to capture hourly trends, weekly patterns, and dynamic demand.

---

### Streaming Configuration

- A subset of the dataset (last 4 months of 2020) is used to simulate real-time streaming in the Streamlit dashboard.
- Each row represents a unique location-time combination and is processed one-by-one to simulate live inference.
- Data is streamed into the app, sent to the Replicate API for prediction, and logged to Google BigQuery for downstream analytics.

---

### Use Case Justification

This dataset is ideally suited for developing and validating a dynamic pricing system because:

- It provides high temporal and spatial resolution
- Includes real occupancy and pricing behaviors across multiple urban zones
- Allows for simulating real-world surge pricing scenarios based on congestion, peak hours, and location demand

This foundation enables training a robust, interpretable base pricing model, and testing real-time surge pricing logic in a controlled, data-driven environment.

## 🧠 Model Development
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
