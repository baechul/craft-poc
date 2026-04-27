"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 15, 2026
Description: Routers for various endpoints 
License: MIT
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from app.services.agents import stream_generator
from app.schemas.prediction import ChatRequest, PredictionRequest, PredictionResponse, PredictionResult
from app.schemas.product_prediction import ProductPredictionRequest, ProductPredictionResponse, ProductPredictionResult
from app.tools.predict_sales import run_prediction
from app.tools.predict_products import run_prediction as run_product_prediction

router = APIRouter()

# The followings demonstrate backend endpoints for both Chat and Rest clients.
# SSE Stream for Chat Client Support


@router.post(
    "/chat",
    summary="Chat with the sales prediction agent",
    description="Accepts a user message and returns a Server-Sent Events (SSE) stream. "
    "The agent uses the `predict_sales` tool to answer sales prediction queries "
    "by category and time frame (week, month, or year).",
    response_description="An SSE stream of JSON chunks `{content: str}`, terminated by `[DONE]`.",
)
async def chat_endpoint(request: ChatRequest):
    return StreamingResponse(stream_generator(request.message), media_type="text/event-stream")

# REST for API Client Support


@router.post(
    "/predict/sales",
    response_model=PredictionResponse,
    summary="Predict top sales by category",
    description="Returns the top N predicted revenue categories for the next K units of the given timeframe. "
    "Uses the pre-loaded LightGBM model loaded at startup.",
    response_description="Ranked list of categories with their total predicted revenue.",
)
async def predict_endpoint(request: PredictionRequest):
    try:
        top_sales = run_prediction(
            top_n=request.top_n,
            timeframe=request.timeframe,
            frame_k=request.frame_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    predictions = [
        PredictionResult(
            category=row["product_category"], predicted_revenue=row["predicted_revenue"])
        for _, row in top_sales.iterrows()
    ]
    return PredictionResponse(
        top_n=request.top_n,
        timeframe=request.timeframe,
        frame_k=request.frame_k,
        predictions=predictions,
    )

# For demo purpose I am using the same backend server for the frontend as well.
# In real prod application, I would move this to a frontend app.


@router.post(
    "/predict/products",
    response_model=ProductPredictionResponse,
    summary="Predict top selling products",
    description="Returns the top P predicted products by units sold for the next K units of the given timeframe. "
    "Uses the pre-loaded LightGBM product model loaded at startup.",
    response_description="Ranked list of products with their total predicted units sold.",
)
async def predict_products_endpoint(request: ProductPredictionRequest):
    try:
        top_products = run_product_prediction(
            top_p=request.top_p,
            timeframe=request.timeframe,
            frame_k=request.frame_k,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    predictions = [
        ProductPredictionResult(
            product_name=row["product_name"],
            product_category=row["product_category"],
            predicted_units=row["predicted_units"],
        )
        for _, row in top_products.iterrows()
    ]
    return ProductPredictionResponse(
        top_p=request.top_p,
        timeframe=request.timeframe,
        frame_k=request.frame_k,
        predictions=predictions,
    )


@router.get(
    "/",
    summary="Demo chat UI",
    description="Serves the static `index.html` demo chat window. "
    "Open http://127.0.0.1:8000 in a browser to interact with the agent.",
    response_description="The static HTML page for the demo chat interface.",
)
async def get_index():
    return FileResponse("static/index.html")
