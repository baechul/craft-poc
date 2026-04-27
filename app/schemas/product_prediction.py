"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 19, 2026
Description: Pydantic schemas for the top products prediction REST endpoint.
License: MIT
"""

from typing import Literal
from pydantic import BaseModel, Field


class ProductPredictionRequest(BaseModel):
    top_p: int = Field(default=3, ge=1, le=20,
                       description="Number of top products to return.")
    timeframe: Literal["week", "month", "year"] = Field(
        default="month", description="Forecast horizon unit.")
    frame_k: int = Field(default=1, ge=1, le=12,
                         description="Number of timeframe units to forecast ahead.")


class ProductPredictionResult(BaseModel):
    product_name: str = Field(description="Product name.")
    product_category: str = Field(description="Product category.")
    predicted_units: float = Field(
        description="Total predicted units sold for the forecast horizon.")


class ProductPredictionResponse(BaseModel):
    top_p: int
    timeframe: str
    frame_k: int
    predictions: list[ProductPredictionResult]
