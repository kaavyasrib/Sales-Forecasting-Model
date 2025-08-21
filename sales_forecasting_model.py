import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')  # Suppress ARIMA convergence warnings

# Create output folder if it doesn't exist
os.makedirs('output', exist_ok=True)

# Sample data generation (simulates daily sales; replace with pd.read_csv('your_data.csv') for real data)
dates = pd.date_range(start='2023-01-01', periods=365, freq='D')  # 1 year of data
sales = np.cumsum(np.random.randn(365) * 10) + 500  # Simulated cumulative sales with noise
df = pd.DataFrame({'Date': dates, 'Sales': sales})
df.set_index('Date', inplace=True)

# Build and fit automated forecasting model (using ARIMA with regression-like differencing)
model = ARIMA(df['Sales'], order=(1, 1, 1))  # p=1 (autoregression), d=1 (differencing), q=1 (moving average)
model_fit = model.fit()

# Generate forecast (e.g., next 90 days)
forecast_steps = 90
forecast = model_fit.forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=forecast_steps, freq='D')

# Print forecast for reporting
print("Forecasted Sales for Next 90 Days:")
forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecasted_Sales': forecast})
print(forecast_df.head(10))  # Show first 10 for brevity

# Visualization (status-at-a-glance for business performance)
plt.figure(figsize=(12, 6))
plt.plot(df['Sales'], label='Historical Sales')
plt.plot(forecast_index, forecast, label='Forecasted Sales', color='red')
plt.title('Sales Forecasting Model (Historical and Future Predictions)')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('output/forecast.png')
plt.close()

# Additional insights: Business recommendations
avg_historical = df['Sales'].mean()
avg_forecast = forecast.mean()
print(f"\nAverage Historical Sales: {avg_historical:.2f}")
print(f"Average Forecasted Sales: {avg_forecast:.2f}")
if avg_forecast > avg_historical:
    print("Recommendation: Prepare for increased demand; optimize inventory and logistics.")
else:
    print("Recommendation: Monitor for potential downturn; review cost efficiencies.")

print("\nForecasting complete. Visualization saved in 'output/' folder.")
