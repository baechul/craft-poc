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

_model = None
_le = None

# Feature cols trained. We have to use the same for prediction.
# 'category_encoded', 'year', 'month', 'week', 'day_of_week',
# 'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_14', 'revenue_lag_30'
_feature_columns = None

# Called when the app startup and shutdown (from main.py)
def set_model(model_artifact) -> None:
  global _model
  global _le
  global _feature_columns
  _model = model_artifact['model']
  _le = model_artifact['le']
  _feature_columns = model_artifact['feature_columns']

# Core prediction logic: shared by the LangChain tool and the REST endpoint.
# Returns a DataFrame with columns [product_category, predicted_revenue].
def run_prediction(top_n: int = 3, timeframe: str = 'month', frame_k: int = 1, history_df=None):
  """Return a DataFrame of top N predicted sales by category for the next frame_k timeframe units."""

  if _model is None:
    raise RuntimeError("Model has not been loaded. Call set_model() at startup.")

  # Read a partial columns required for prediction instead of full read
  df = pd.read_csv('app/data/sales.csv', usecols=['date', 'product_category', 'total_revenue'])

  # Convert date column to datetime
  if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

  # Aggregate to category-date level for forecasting
  category_sales = (
    df.groupby(['date', 'product_category'])['total_revenue']
    .sum()
    .reset_index()
    .sort_values(['product_category', 'date'])
  )

  # My model learned seasonal patterns hence adding the same features for prediction
  category_sales['year'] = category_sales['date'].dt.year
  category_sales['month'] = category_sales['date'].dt.month
  category_sales['week'] = category_sales['date'].dt.isocalendar().week

  # _le.transform - NOT _le.fit_transform (which should be used only during the learning)
  category_sales['category_encoded'] = _le.transform(category_sales['product_category'])

  # Create lag features for time series forecasting (same lags as trained)
  for lag in [1, 7, 14, 30]:
    category_sales[f'revenue_lag_{lag}'] = category_sales.groupby('product_category')[
      'total_revenue'].shift(lag)

  # Since shift() will create na to fill in when shifted.
  category_sales = category_sales.dropna(subset=[
    'revenue_lag_1', 'revenue_lag_7', 'revenue_lag_14', 'revenue_lag_30'])

  horizons = {
    'week': 7*frame_k,
    'month': 30*frame_k,
    'year': 365*frame_k,
  }

  if timeframe not in horizons:
    raise ValueError("timeframe must be one of: 'week', 'month', 'year'")

  if history_df is None:
    history_df = category_sales.copy()

  history_df = history_df.sort_values(['product_category', 'date']).copy()
  horizon = horizons[timeframe]

  all_forecasts = []
  for category, group in history_df.groupby('product_category'):
    group = group.sort_values('date').copy()
    revenue_history = group['total_revenue'].tolist()
    category_code = int(group['category_encoded'].iloc[-1]) # I just picked the last one as all have the same category.
    current_date = group['date'].max()

    for _ in range(horizon):
      future_date = current_date + pd.Timedelta(days=1)

      def lag_value(lag):
        if len(revenue_history) >= lag:
          return revenue_history[-lag]
        return revenue_history[-1]

      # TODO: For now, I am using the hard-coded feature cols. Should this be dynamic out of _feature_columns
      # The loaded model can only predict a single day based on lags (upto 30 days)
      # Hence using for-loop, we have to keep adding a day prediction upto the requested horizon.
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
      }], columns=_feature_columns)

      predicted_revenue = max(float(_model.predict(row)[0]), 0.0)

      all_forecasts.append({
        'date': future_date,
        'product_category': category,
        'predicted_revenue': predicted_revenue,
        'timeframe': timeframe,
      })

      revenue_history.append(predicted_revenue)
      current_date = future_date

  forecast_df = pd.DataFrame(all_forecasts)
  return (
    forecast_df.groupby('product_category', as_index=False)['predicted_revenue']
    .sum()
    .sort_values('predicted_revenue', ascending=False)  # type: ignore
    .head(top_n)
    .reset_index(drop=True)
  )


# Tool: called by the agent when it decides a sales prediction is needed.
@tool("predict_sales", return_direct=False)
def predict_sales(top_n=3, timeframe='month', frame_k=1, history_df=None) -> str:
  """Predict the N top sales by category for the next K units of time frame (week, month, year)."""
  top_sales = run_prediction(
      top_n=top_n, timeframe=timeframe, frame_k=frame_k, history_df=history_df)
  
  return top_sales
  # lines = [
  #     f"Top {top_n} predicted sales by category for the next {frame_k} {timeframe}(s):"]
  # for i, row in top_sales.iterrows():
  #     lines.append(
  #         f"  {i + 1}. {row['product_category']}: ${row['predicted_revenue']:,.2f}")
  # return "\n".join(lines)
