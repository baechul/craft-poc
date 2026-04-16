"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 15, 2026
Description: Predict sales tool 
License: MIT
"""

from langchain_core.tools import tool
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import joblib

# Tool: called by the agent when it decides a sales prediction is needed.
# Returns a plain string result that the agent incorporates into its final response.
@tool("predict_sales", return_direct=False)
def predict_sales(timeframe='month', top_n=3, history_df=None) -> str:
  """Predict the top sales by category for a given time frame (week, month, year)."""
  df = pd.read_csv('app/models/sales.csv')

  # Convert date column to datetime
  if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

  df_features = df.copy()

  df_features['year'] = df_features['date'].dt.year
  df_features['month'] = df_features['date'].dt.month
  df_features['week'] = df_features['date'].dt.isocalendar().week
  df_features['day_of_week'] = df_features['date'].dt.dayofweek
  df_features['day_of_month'] = df_features['date'].dt.day
  df_features['quarter'] = df_features['date'].dt.quarter

  # Aggregate to category-date level for forecasting
  # category_sales has only date,	product_category, total_revenue.
  category_sales = (
    df_features.groupby(['date', 'product_category'])['total_revenue']
    .sum()
    .reset_index()
    .sort_values(['product_category', 'date'])
  )
  category_sales['year'] = category_sales['date'].dt.year
  category_sales['month'] = category_sales['date'].dt.month
  category_sales['week'] = category_sales['date'].dt.isocalendar().week
  category_sales['day_of_week'] = category_sales['date'].dt.dayofweek

  le = LabelEncoder()
  category_sales['category_encoded'] = le.fit_transform(category_sales['product_category'])

  # Create lag features for time series forecasting
  category_sales['revenue_lag_1'] = category_sales.groupby('product_category')['total_revenue'].shift(1)
  category_sales['revenue_lag_7'] = category_sales.groupby('product_category')['total_revenue'].shift(7)
  category_sales['revenue_lag_14'] = category_sales.groupby('product_category')['total_revenue'].shift(14)
  category_sales['revenue_lag_30'] = category_sales.groupby('product_category')['total_revenue'].shift(30)

  # since shift() will create na.
  category_sales = category_sales.dropna(subset=['revenue_lag_1', 'revenue_lag_7', 'revenue_lag_14', 'revenue_lag_30'])
  category_sales = category_sales.sort_values('date')

  horizons = {
    'week': 7,
    'month': 30,
    'year': 365,
  }

  if timeframe not in horizons:
    raise ValueError("timeframe must be one of: 'week', 'month', 'year'")

  if history_df is None:
    history_df = category_sales.copy()

  model = joblib.load('./app/models/top_sales_prediction.joblib')
  history_df = history_df.sort_values(['product_category', 'date']).copy()
  horizon = horizons[timeframe]

  features = [
    'category_encoded', 'year', 'month', 'week', 'day_of_week',
    'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_14', 'revenue_lag_30'
  ]

  all_forecasts = []

  for category, group in history_df.groupby('product_category'):
    group = group.sort_values('date').copy()
    revenue_history = group['total_revenue'].tolist()
    category_code = int(group['category_encoded'].iloc[-1])
    current_date = group['date'].max()

    for _ in range(horizon):
      future_date = current_date + pd.Timedelta(days=1)

      def lag_value(lag):
        if len(revenue_history) >= lag:
          return revenue_history[-lag]
        return revenue_history[-1]

      row = pd.DataFrame([{
        'category_encoded': category_code,
        'year': future_date.year,
        'month': future_date.month,
        'week': int(future_date.isocalendar().week),
        'day_of_week': future_date.dayofweek,
        'revenue_lag_1': lag_value(1),
        'revenue_lag_7': lag_value(7),
        'revenue_lag_14': lag_value(14),
        'revenue_lag_30': lag_value(30),
      }], columns=features)

      predicted_revenue = max(float(model.predict(row)[0]), 0.0)

      all_forecasts.append({
        'date': future_date,
        'product_category': category,
        'predicted_revenue': predicted_revenue,
        'timeframe': timeframe,
      })

      revenue_history.append(predicted_revenue)
      current_date = future_date

  forecast_df = pd.DataFrame(all_forecasts)

  top_sales = (
    forecast_df.groupby('product_category', as_index=False)['predicted_revenue']
    .sum()
    .sort_values('predicted_revenue', ascending=False)  # type: ignore
    .head(top_n)
    .reset_index(drop=True)
  )

  return forecast_df, top_sales
