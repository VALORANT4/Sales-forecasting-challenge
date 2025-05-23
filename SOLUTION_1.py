


import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

data = pd.read_csv("C:\VIBES\sales-forcecasting\DATA\data.csv", encoding='ISO-8859-1')  # or use .xlsx with pd.read_excel

# Display the first 5 rows
print(data.head())

# Summary statistics
print(data.info())
print(data.describe())

# Check missing values
print(data.isnull().sum())

# Drop rows where CustomerID is null
data = data.dropna(subset=['CustomerID'])

# Reset index
data.reset_index(drop=True, inplace=True)

# Convert to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
2
# Calculate revenue
data['Revenue'] = data['Quantity'] * data['UnitPrice']

# Group by date
daily_sales = data.groupby(data['InvoiceDate'].dt.date)['Revenue'].sum().reset_index()

# Rename columns
daily_sales.columns = ['Date', 'DailyRevenue']
print(daily_sales.head())

import numpy as np
date_rng = pd.date_range(start='2024-01-01', end='2024-04-30', freq='D')
np.random.seed(42)
revenue_data = np.random.normal(loc=2000, scale=300, size=len(date_rng))

df = pd.DataFrame({'ds': date_rng, 'y': revenue_data})

# Step 2: Initialize and fit the Prophet model
model = Prophet()
model.fit(df)

# Step 3: Create a dataframe to hold future dates
future = model.make_future_dataframe(periods=7)

# Step 4: Predict future revenue
forecast = model.predict(future)

# Step 5: Visualize actual vs forecasted sales

plt.figure(figsize=(12, 6))
plt.plot(df['ds'], df['y'], label='Actual Sales')
plt.plot(forecast['ds'], forecast['yhat'], label='Forecasted Sales', color='orange')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
plt.axvline(x=df['ds'].max(), color='gray', linestyle='--', label='Forecast Start')
plt.title('Actual vs Forecasted Daily Revenue')
plt.xlabel('Date')
plt.ylabel('Revenue')
plt.legend()
plt.tight_layout()
plt.show()

