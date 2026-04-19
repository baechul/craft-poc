"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 18, 2026
Description: Pydantic schemas for the sales prediction REST endpoint.
License: MIT
"""

from typing import Literal
from pydantic import BaseModel, Field

# Chat Interface Support
class ChatRequest(BaseModel):
  message: str

# REST Interface Support
class PredictionRequest(BaseModel):
  top_n: int = Field(default=3, description="Number of top categories to return.")
  timeframe: Literal["week", "month", "year"] = Field(default="month", description="Forecast horizon unit.")
  frame_k: int = Field(default=1, description="Number of timeframe units to forecast ahead.")

class PredictionResult(BaseModel):
  category: str = Field(description="Product category name.")
  predicted_revenue: float = Field(description="Total predicted revenue for the forecast horizon.")

class PredictionResponse(BaseModel):
  top_n: int
  timeframe: str
  frame_k: int
  predictions: list[PredictionResult]
