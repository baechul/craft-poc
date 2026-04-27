"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 19, 2026
Description: Predict top products tool
License: MIT
"""

from langchain_core.tools import tool
import pandas as pd

_model = None
_le_cat = None
_le_prod = None

# Feature cols trained:
# 'category_encoded', 'product_encoded', 'year', 'month', 'week', 'day_of_week',
# 'units_lag_1', 'units_lag_7', 'units_lag_14', 'units_lag_30'
_feature_columns = None


# Called at app startup and shutdown (main.py)
def set_model(model_artifact) -> None:
    global _model, _le_cat, _le_prod, _feature_columns
    if model_artifact is None:
        _model = _le_cat = _le_prod = _feature_columns = None
        return
    _model = model_artifact['model']
    _le_cat = model_artifact['le_cat']
    _le_prod = model_artifact['le_prod']
    _feature_columns = model_artifact['feature_columns']


# Core prediction logic: shared by the LangChain tool and the REST endpoint.
# Returns a DataFrame with columns [product_name, product_category, predicted_units].
def run_prediction(top_p: int = 3, timeframe: str = 'month', frame_k: int = 1, history_df=None):
    """Return a DataFrame of top P predicted products by units sold for the next frame_k timeframe units."""

    if _model is None:
        raise RuntimeError(
            "Product model has not been loaded. Call set_model() at startup.")

    horizons = {
        'week': 7 * frame_k,
        'month': 30 * frame_k,
        'year': 365 * frame_k,
    }

    if timeframe not in horizons:
        raise ValueError("timeframe must be one of: 'week', 'month', 'year'")

    # Read only the columns needed for prediction
    df = pd.read_csv(
        'app/data/sales.csv',
        usecols=['date', 'product_category', 'product_name', 'units_sold'],
    )

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Aggregate to product-date level
    product_sales = (
        df.groupby(['date', 'product_category', 'product_name'])['units_sold']
        .sum()
        .reset_index()
        .sort_values(['product_name', 'date'])
    )

    # Encode categories and products (transform only — encoders were fit during training)
    product_sales['category_encoded'] = _le_cat.transform(
        product_sales['product_category'])
    product_sales['product_encoded'] = _le_prod.transform(
        product_sales['product_name'])

    # Calendar features
    product_sales['year'] = product_sales['date'].dt.year
    product_sales['month'] = product_sales['date'].dt.month
    product_sales['week'] = product_sales['date'].dt.isocalendar().week

    # Lag features (same lags as trained)
    for lag in [1, 7, 14, 30]:
        product_sales[f'units_lag_{lag}'] = product_sales.groupby('product_name')[
            'units_sold'].shift(lag)

    product_sales = product_sales.dropna(subset=[
        'units_lag_1', 'units_lag_7', 'units_lag_14', 'units_lag_30'])

    if history_df is None:
        history_df = product_sales.copy()

    history_df = history_df.sort_values(['product_name', 'date']).copy()
    horizon = horizons[timeframe]

    all_forecasts = []
    for product_name, group in history_df.groupby('product_name'):
        group = group.sort_values('date').copy()
        units_history = group['units_sold'].tolist()
        category_code = int(group['category_encoded'].iloc[-1])
        product_code = int(group['product_encoded'].iloc[-1])
        product_category = group['product_category'].iloc[-1]
        current_date = group['date'].max()

        for _ in range(horizon):
            future_date = current_date + pd.Timedelta(days=1)

            def lag_value(lag):
                if len(units_history) >= lag:
                    return units_history[-lag]
                return units_history[-1]

            row = pd.DataFrame([{
                'category_encoded': category_code,
                'product_encoded': product_code,
                'year': future_date.year,
                'month': future_date.month,
                'week': int(future_date.isocalendar().week),
                'day_of_week': future_date.dayofweek,
                'units_lag_1': lag_value(1),
                'units_lag_7': lag_value(7),
                'units_lag_14': lag_value(14),
                'units_lag_30': lag_value(30),
            }], columns=_feature_columns)

            predicted_units = max(float(_model.predict(row)[0]), 0.0)

            all_forecasts.append({
                'date': future_date,
                'product_name': product_name,
                'product_category': product_category,
                'predicted_units': predicted_units,
            })

            units_history.append(predicted_units)
            current_date = future_date

    forecast_df = pd.DataFrame(all_forecasts)

    return (
        forecast_df.groupby(['product_name', 'product_category'], as_index=False)[
            'predicted_units']
        .sum()
        .sort_values('predicted_units', ascending=False)  # type: ignore
        .head(top_p)
        .reset_index(drop=True)
    )


# Tool: called by the agent when it decides a product prediction is needed.
@tool("predict_top_products", return_direct=False)
def predict_top_products(top_p=3, timeframe='month', frame_k=1, history_df=None) -> str:
    """Predict the P top selling products by units for the next K units of time frame (week, month, year)."""
    top_products = run_prediction(
        top_p=top_p, timeframe=timeframe, frame_k=frame_k, history_df=history_df)
    lines = [
        f"Top {top_p} predicted products by units sold for the next {frame_k} {timeframe}(s):"]
    for i, row in top_products.iterrows():
        lines.append(
            f"  {i + 1}. {row['product_name']} ({row['product_category']}): {row['predicted_units']:,.0f} units")
    return "\n".join(lines)
