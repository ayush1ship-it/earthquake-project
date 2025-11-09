import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import warnings

warnings.filterwarnings("ignore")

# ======================================
# Step 1: Load & Set Features and Target
# ======================================

# Load your CSV
#df = pd.read_csv("USGS_processed_1.csv")
df = pd.read_csv("USGS_processed_2.csv")
#print(df.head())

# Features and targets
X = df[['Month', 'Latitude', 'Longitude']]
y_mag = df['Magnitude']

# ===========================
# Step 2: Train Models
# ===========================

# Magnitude prediction model
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y_mag, test_size=0.2, random_state=42)
mag_model = RandomForestRegressor(n_estimators=100, random_state=42)
mag_model.fit(X_train1, y_train1)
y_mag_pred = mag_model.predict(X_test1)

# ===========================
# Step 3: Evaluate Models
# ===========================

print("\nModel Evaluation:")

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test1, y_mag_pred)
print(f"Earthquake Magnitude MAE: {mae:.3f}")

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test1, y_mag_pred)
print(f"Earthquake Magnitude MSE: {mse:.3f}")

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Earthquake Magnitude RMSE: {rmse:.3f}")

# ===========================
# Step 4: Save Models (Optional)
# ===========================
joblib.dump(mag_model, "random_forest_regressor.pkl")

'''
Jan 2011 to Jun 2025
Model Evaluation:
Earthquake Magnitude MAE: 0.326
Earthquake Magnitude MSE: 0.205
Earthquake Magnitude RMSE: 0.453
'''

'''
Jan 2015 to Jun 2025
Model Evaluation:
Earthquake Magnitude MAE: 0.320
Earthquake Magnitude MSE: 0.198
Earthquake Magnitude RMSE: 0.445
'''
