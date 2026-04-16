"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 15, 2026
Description: AI Agents and Tools 
License: MIT
"""

import json
from typing import AsyncIterable
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from app.core.config import settings

# Tool: called by the agent when it decides a sales prediction is needed.
# Returns a plain string result that the agent incorporates into its final response.
@tool("predict_sales", return_direct=False)
def predict_sales(time_frame: str) -> str:
  """Predict the top sales by category for a given time frame (week, month, year)."""
  return "Not implemented yet"

# Agent: a compiled LangGraph state graph that orchestrates LLM reasoning and tool calls.
# create_agent wires the model, tools, and system prompt into a ReAct-style loop.
agent = create_agent(
  model=ChatOpenAI(api_key=settings.openai_api_key, model="gpt-4.1-mini"),
  tools=[predict_sales],
  system_prompt="You are a helpful sales prediction assistant. Predict the top sales by category over different selectable time frames, such as week, month, or year.",
)

# Async generator that streams LLM token chunks as SSE-formatted strings.
# Filters astream_events to only forward on_chat_model_stream events (token-level output).
# Each yielded string follows the SSE protocol: "data: <json>\n\n".
# Yields "data: [DONE]\n\n" at the end to signal stream completion to the client.
async def stream_generator(user_input: str) -> AsyncIterable[str]:
    async for event in agent.astream_events({"messages": [("human", user_input)]}, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                yield f"data: {json.dumps({'content': chunk})}\n\n"
    yield "data: [DONE]\n\n"
