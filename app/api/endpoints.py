"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 15, 2026
Description: Routers for various endpoints 
License: MIT
"""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from app.services.agents import stream_generator

router = APIRouter()

# Simple Pydantic model for the incoming request
class ChatRequest(BaseModel):
    message: str

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

@router.get(
    "/",
    summary="Demo chat UI",
    description="Serves the static `index.html` demo chat window. "
    "Open http://127.0.0.1:8000 in a browser to interact with the agent.",
    response_description="The static HTML page for the demo chat interface.",
)
async def get_index():
    return FileResponse("static/index.html")
