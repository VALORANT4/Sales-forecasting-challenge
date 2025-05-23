# Sales-forecasting-challenge

├── README.md
├── CODE/
│   └── SOLUTION_1.PY
│   └── SOLUTION_2.PY
├── DATA/
│   └── (https://drive.google.com/drive/folders/1vBFWWdpflo3VsXTqX78qaKdTY1ODt4_I?usp=sharing)
├── OUTPUT/
│   └── SOLUTION_1.png
│   └── SOLUTION_2.png















📊 Sales Forecasting Report
Forecasting Daily Revenue for the Next 7 Days
Techniques Used: Prophet (Time Series) & XGBoost (Machine Learning)
Data Source: data.csv – Online retail transactions

1. 🧹 Data Preprocessing
Columns used: InvoiceDate, Quantity, UnitPrice

Derived feature:

Revenue = Quantity × UnitPrice

Date formatted to daily granularity (removed time).

Aggregated total revenue per day.

2. 🔮 Method 1 – Prophet (Time Series Model)
Description:
Developed by Facebook.

Automatically detects trend, seasonality, and holidays.

Ideal for business and daily-level forecasting.

Implementation:
Trained on historical daily revenue.

Generated forecasts for the next 7 days.

Produced point forecasts along with confidence intervals (yhat_lower, yhat_upper).

Visual Output:
Line chart of actual revenue vs. forecast.

Shaded region indicating prediction uncertainty.

3. 🤖 Method 2 – XGBoost (Feature-Based ML Model)
Description:
A powerful tree-boosting algorithm.

Requires manual feature engineering (unlike Prophet).

Features Created:
Calendar-based: day, month, day of week

Lag-based: Revenue from 1 day ago, Revenue from 7 days ago

Implementation:
Trained on all data except the last 7 days.

Forecasted the next 7 days’ revenue using engineered features.

Visual Output:
Overlay of XGBoost forecast with Prophet and actuals.

Highlights XGBoost’s ability to adapt to recent trends.

4. 📈 Forecast Comparison
Aspect	Prophet	XGBoost
Model Type	Additive Time Series Model	Tree-Based Regressor
Handles Seasonality	Yes (automatic)	Only with manual features
Forecasting Approach	Trend + Seasonality + Noise	Learns from historical patterns
Forecast Horizon	7 Days	7 Days

5. 📁 Output Summary
✅ Prophet Forecast: Includes yhat, yhat_lower, yhat_upper

✅ XGBoost Forecast: Saved to xgboost_7_day_forecast.csv

✅ Comparison Plot: Shows all three lines – actual, Prophet, XGBoost

