"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 15, 2026
Description: Main entry point. 
License: MIT
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.api.endpoints import router
from app.tools.predict_sales import set_model as set_sales_model
from app.tools.predict_products import set_model as set_products_model
import joblib

# Loading the model as a lifespan context in app booting.
@asynccontextmanager
async def lifespan(app: FastAPI):
  set_sales_model(joblib.load('./app/models/top_sales_prediction_lgb.joblib'))
  set_products_model(joblib.load('./app/models/top_products_prediction_lgb.joblib'))
  yield
  set_sales_model(None)
  set_products_model(None)

app = FastAPI(lifespan=lifespan)
app.include_router(router)
