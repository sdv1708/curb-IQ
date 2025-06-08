import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import pickle


dataLoc = "/content/drive/MyDrive/BUDT737/exported.csv"

# Loading dataset
df = pd.read_csv(dataLoc, nrows = 1000000)

# ---------------------------------------------
# Feature Engineering
# ---------------------------------------------
df['timestamp'] = pd.to_datetime(df['OccupancyDateTime'], format="%m/%d/%Y %I:%M:%S %p")
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek + 1
df['month'] = df['timestamp'].dt.month
df['is_weekend'] = df['dayofweek'].isin([1, 7]).astype(int)
df['is_peak_hour'] = df['hour'].between(8, 11) | df['hour'].between(16, 19)
df['is_peak_hour'] = df['is_peak_hour'].astype(int)
df['occupancy_rate'] = df['PaidOccupancy'] / df['ParkingSpaceCount']

# Columns
categorical_cols = ["BlockfaceName", "SideOfStreet", "PaidParkingArea", "PaidParkingSubArea", "ParkingCategory"]
numeric_cols = ["PaidOccupancy", "ParkingTimeLimitCategory", "ParkingSpaceCount", "occupancy_rate"]
extra_feats = ["hour", "dayofweek", "is_weekend", "is_peak_hour"]

X = df[categorical_cols + numeric_cols + extra_feats]
y = df["PaidParkingRate"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# ---------------------------------------------
# Manual Preprocessing
# ---------------------------------------------
# Categorical Encoding
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = encoder.fit_transform(X_train[categorical_cols])

# Numeric Scaling
scaler = StandardScaler()
X_num = scaler.fit_transform(X_train[numeric_cols])

# Combine all
X_final_train = np.hstack([X_cat, X_num, X_train[extra_feats].values])

# Model Training
model = LinearRegression()
model.fit(X_final_train, y_train)


model_bundle = {
    "model": model,
    "encoder": encoder,
    "scaler": scaler,
    "categorical_cols": categorical_cols,
    "numeric_cols": numeric_cols,
    "extra_feats": extra_feats
}

with open("model_bundle.pkl", "wb") as f:
    pickle.dump(model_bundle, f)