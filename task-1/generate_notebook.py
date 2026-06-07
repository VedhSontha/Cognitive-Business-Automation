import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

text_intro = """# Sales Forecasting Analysis
This notebook covers the end-to-end process of sales forecasting:
1. Data Loading & Cleaning
2. Exploratory Data Analysis (EDA)
3. Feature Engineering
4. Model Training & Comparison (ARIMA, Prophet, XGBoost)
5. Forecasting (Global & by StoreType)
6. Exporting Results for Dashboard
"""

code_imports = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
import os
warnings.filterwarnings('ignore')

# Plot settings
plt.style.use('ggplot')
pd.set_option('display.max_columns', None)
"""

text_load = """## 1. Data Loading and Cleaning"""

code_load = """# Load datasets
train = pd.read_csv('../train.csv', parse_dates=['Date'], low_memory=False)
store = pd.read_csv('../store.csv')

# Merge store info
df = pd.merge(train, store, on='Store', how='left')

print(f"Train shape: {df.shape}")
df.head()
"""

code_clean = """# Filter data: Open stores only, Sales > 0
df = df[(df['Open'] == 1) & (df['Sales'] > 0)]

# Handle missing values
df['CompetitionDistance'].fillna(df['CompetitionDistance'].median(), inplace=True)
df['StoreType'] = df['StoreType'].astype(str) # Ensure string for grouping

print("Data cleaned. Missing values handled.")
"""

text_eda = """## 2. Exploratory Data Analysis (EDA) & Feature Extraction for Dashboard"""

code_eda = """# 1. Monthly & Yearly Stats
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

monthly_stats = df.groupby(['Year', 'Month'])['Sales'].sum().reset_index()
monthly_stats['Date'] = pd.to_datetime(monthly_stats[['Year', 'Month']].assign(DAY=1))

yearly_stats = df.groupby('Year')['Sales'].sum().reset_index()

# 2. Top Stores
top_stores = df.groupby('Store')['Sales'].sum().reset_index().sort_values('Sales', ascending=False).head(10)

# 3. Seasonality (Avg Sales per Month across all years)
seasonality = df.groupby('Month')['Sales'].mean().reset_index()

# 4. Heatmap Data (DayOfWeek vs Month)
heatmap_data = df.groupby(['DayOfWeek', 'Month'])['Sales'].mean().reset_index()

# 5. Day of Week Stats
dayofweek_stats = df.groupby('DayOfWeek')['Sales'].sum().reset_index()

print("EDA metrics calculated.")
"""

text_model_prep = """## 3. Model Training & Comparison (Global)"""

code_model_prep = """# Aggregate daily sales for global modeling
daily_sales = df.groupby('Date')['Sales'].sum().reset_index()
daily_sales.columns = ['ds', 'y']

# Split Train/Test
train_size = int(len(daily_sales) * 0.9)
train_data = daily_sales.iloc[:train_size]
test_data = daily_sales.iloc[train_size:]
"""

code_compare = """# Compare Models on Global Data
# 1. ARIMA
model_arima = ARIMA(train_data.set_index('ds')['y'], order=(5,1,0))
model_arima_fit = model_arima.fit()
arima_pred = model_arima_fit.forecast(steps=len(test_data))
arima_mae = mean_absolute_error(test_data['y'], arima_pred)

# 2. Prophet
m_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
m_prophet.add_country_holidays(country_name='DE')
m_prophet.fit(train_data)
future_p = m_prophet.make_future_dataframe(periods=len(test_data))
forecast_p = m_prophet.predict(future_p)
prophet_pred = forecast_p.iloc[-len(test_data):]['yhat'].values
prophet_mae = mean_absolute_error(test_data['y'], prophet_pred)

# 3. XGBoost
def create_features(df):
    df = df.copy()
    df['dayofweek'] = df['ds'].dt.dayofweek
    df['quarter'] = df['ds'].dt.quarter
    df['month'] = df['ds'].dt.month
    df['year'] = df['ds'].dt.year
    return df

xgb_train = create_features(train_data)
xgb_test = create_features(test_data)
features = ['dayofweek', 'quarter', 'month', 'year']

reg = xgb.XGBRegressor(n_estimators=1000, early_stopping_rounds=50, learning_rate=0.01)
reg.fit(xgb_train[features], xgb_train['y'],
        eval_set=[(xgb_train[features], xgb_train['y']), (xgb_test[features], xgb_test['y'])],
        verbose=False)
xgb_pred = reg.predict(xgb_test[features])
xgb_mae = mean_absolute_error(test_data['y'], xgb_pred)

print(f"ARIMA MAE: {arima_mae:.0f}")
print(f"Prophet MAE: {prophet_mae:.0f}")
print(f"XGBoost MAE: {xgb_mae:.0f}")

best_model = min([('ARIMA', arima_mae), ('Prophet', prophet_mae), ('XGBoost', xgb_mae)], key=lambda x: x[1])
print(f"Best Model: {best_model[0]}")
"""

text_forecast = """## 4. Generating Forecasts by StoreType"""

code_forecast = """# We will use Prophet for the final forecasts as it handles seasonality and missing data gracefully, 
# and provides easy confidence intervals which are great for dashboards.

store_types = df['StoreType'].unique()
all_forecasts = []

print(f"Forecasting for StoreTypes: {store_types}")

for st_type in store_types:
    # Filter data
    st_data = df[df['StoreType'] == st_type].groupby('Date')['Sales'].sum().reset_index()
    st_data.columns = ['ds', 'y']
    
    # Train Prophet
    m = Prophet(yearly_seasonality=True, daily_seasonality=False)
    m.add_country_holidays(country_name='DE')
    m.fit(st_data)
    
    # Forecast 90 days
    future = m.make_future_dataframe(periods=90)
    forecast = m.predict(future)
    
    # Add StoreType column
    forecast['StoreType'] = st_type
    all_forecasts.append(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'StoreType']])

final_forecast = pd.concat(all_forecasts)
"""

text_export = """## 5. Exporting Data for Dashboard"""

code_export = """os.makedirs('../outputs', exist_ok=True)

# 1. Forecasts
final_forecast.to_csv('../outputs/forecast_results.csv', index=False)

# 2. Actuals (Store-level for filtering)
# We need Store ID, StoreType, Date, and Sales
actuals_by_store = df[['Date', 'Store', 'StoreType', 'Sales']].copy()
actuals_by_store.to_csv('../outputs/actual_sales.csv', index=False)

# 3. Monthly Stats
monthly_stats.to_csv('../outputs/monthly_stats.csv', index=False)

# 4. Yearly Stats
yearly_stats.to_csv('../outputs/yearly_stats.csv', index=False)

# 5. Top Stores
top_stores.to_csv('../outputs/top_stores.csv', index=False)

# 6. Seasonality
seasonality.to_csv('../outputs/seasonality.csv', index=False)

# 7. Heatmap Data
heatmap_data.to_csv('../outputs/heatmap_data.csv', index=False)

# 8. Day of Week Stats
dayofweek_stats.to_csv('../outputs/dayofweek_stats.csv', index=False)

print("All data exported successfully to ../outputs/")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_imports),
    nbf.v4.new_markdown_cell(text_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_clean),
    nbf.v4.new_markdown_cell(text_eda),
    nbf.v4.new_code_cell(code_eda),
    nbf.v4.new_markdown_cell(text_model_prep),
    nbf.v4.new_code_cell(code_model_prep),
    nbf.v4.new_code_cell(code_compare),
    nbf.v4.new_markdown_cell(text_forecast),
    nbf.v4.new_code_cell(code_forecast),
    nbf.v4.new_markdown_cell(text_export),
    nbf.v4.new_code_cell(code_export)
]

with open('notebooks/forecasting_analysis.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook created successfully.")
