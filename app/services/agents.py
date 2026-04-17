"""
Copyright (c) 2026 Baechul Kim
All rights reserved.

Author: Baechul Kim <baechul@gmail.com>
Date: April 15, 2026
Description: AI agents 
License: MIT
"""

from typing import AsyncIterable
from pathlib import Path
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from app.core.config import settings
from app.tools.predict_sales import predict_sales
import json
import re

# Agent: a compiled LangGraph state graph that orchestrates LLM reasoning and tool calls.
# create_agent wires the model, tools, and system prompt into a ReAct-style loop.
_SYSTEM_PROMPT = (Path(__file__).parent.parent/"prompts"/"sales_agent.txt").read_text()
agent = create_agent(
  model=ChatOpenAI(api_key=settings.openai_api_key, model="gpt-4.1-mini"),
  tools=[predict_sales],
  system_prompt=_SYSTEM_PROMPT,
)

# Async generator that streams LLM token chunks as SSE-formatted strings.
# Filters astream_events to only forward on_chat_model_stream events (token-level output).
# Each yielded string follows the SSE protocol: "data: <json>\n\n".
# Yields "data: [DONE]\n\n" at the end to signal stream completion to the client.


async def stream_generator(user_input: str) -> AsyncIterable[str]:
  try:
    full_response = []
    async for event in agent.astream_events({"messages": [("human", user_input)]}, version="v2"):
      if event["event"] == "on_chat_model_stream":
        chunk = event["data"]["chunk"].content
        if chunk:
          full_response.append(chunk)
    response = re.sub(r'(?i)(top|next|last)(\d)', r'\1 \2', "".join(full_response))
    yield f"data: {json.dumps({'content': response})}\n\n"
    yield "data: [DONE]\n\n"
  except Exception as e:
    yield f"data: {json.dumps({'error': str(e)})}\n\n"
    yield "data: [DONE]\n\n"
