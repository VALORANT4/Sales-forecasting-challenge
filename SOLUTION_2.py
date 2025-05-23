import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import xgboost as xgb
from datetime import timedelta
from sklearn.metrics import mean_squared_error

# Load data
file_path = r"C:\VIBES\sales-forcecasting\data.csv"
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# Preprocessing
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']).dt.date
df['Revenue'] = df['Quantity'] * df['UnitPrice']
daily_revenue = df.groupby('InvoiceDate')['Revenue'].sum().reset_index()
daily_revenue.columns = ['ds', 'y']

# Prophet Forecast
prophet_model = Prophet()
prophet_model.fit(daily_revenue)
future = prophet_model.make_future_dataframe(periods=7)
forecast = prophet_model.predict(future)

# ---- XGBoost Forecast ----

# Step 1: Create time-based features
xgb_df = daily_revenue.copy()
xgb_df['ds'] = pd.to_datetime(xgb_df['ds'])
xgb_df['day'] = xgb_df['ds'].dt.day
xgb_df['month'] = xgb_df['ds'].dt.month
xgb_df['dayofweek'] = xgb_df['ds'].dt.dayofweek
xgb_df['lag_1'] = xgb_df['y'].shift(1)
xgb_df['lag_7'] = xgb_df['y'].shift(7)
xgb_df = xgb_df.dropna()

# Step 2: Train-test split
train = xgb_df.iloc[:-7]
test = xgb_df.iloc[-7:]

features = ['day', 'month', 'dayofweek', 'lag_1', 'lag_7']
X_train, y_train = train[features], train['y']
X_test, y_test = test[features], test['y']

# Step 3: Train XGBoost
model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model_xgb.fit(X_train, y_train)

# Step 4: Predict next 7 days
xgb_forecast = model_xgb.predict(X_test)
test_dates = test['ds'].dt.date.reset_index(drop=True)
xgb_results = pd.DataFrame({'ds': test_dates, 'xgb_forecast': xgb_forecast})

# ---- Plot Both Models ----

plt.figure(figsize=(14, 6))

# Prophet
plt.plot(daily_revenue['ds'], daily_revenue['y'], label='Actual Revenue')
plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Forecast', color='orange')

# XGBoost
plt.plot(xgb_results['ds'], xgb_results['xgb_forecast'], label='XGBoost Forecast', color='green', marker='o')

plt.axvline(x=daily_revenue['ds'].max(), color='gray', linestyle='--', label='Forecast Start')
plt.title('7-Day Revenue Forecast: Prophet vs XGBoost')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.tight_layout()
plt.show()

# Optional: Save XGBoost results
xgb_results.to_csv("xgboost_7_day_forecast.csv", index=False)
