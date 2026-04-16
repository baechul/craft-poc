---
agent: agent
description: Scaffold a new FastAPI route with schema, service method, and tests
---

Create a new API endpoint for the `craft-poc` FastAPI application.

## Steps

1. **Schema** (`app/schemas/<resource>.py`)
   - Define Pydantic v2 request and response models using built-in generics.
   - All fields must be typed; use `Field(...)` with descriptions where helpful.

2. **Service** (`app/services/<resource>_service.py`)
   - Add an `async def` method containing the business logic.
   - If an LLM is involved, use an LCEL pipeline (`|` operator) with `ChatPromptTemplate`.
   - Never hard-code API keys; inject settings via `app/core/`.

3. **Router** (`app/api/<resource>.py`)
   - Create an `APIRouter` with an appropriate prefix and tags.
   - Set `response_model` on every route decorator.
   - Use `Depends` for service and settings injection.
   - Raise `HTTPException` for error cases — no raw dict returns.

4. **Registration** (`app/main.py`)
   - Import the new router and call `app.include_router(...)`.

5. **Tests** (`tests/api/test_<resource>.py` and `tests/services/test_<resource>_service.py`)
   - Use `pytest` + `pytest-asyncio`.
   - Mock all LLM / external calls; never hit real APIs.

## Input

Describe the endpoint you need:

- HTTP method and path
- Purpose / business logic
- Request payload fields
- Expected response fields
