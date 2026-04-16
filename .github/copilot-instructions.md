# GitHub Copilot Instructions

## Project Overview

`craft-poc` is a FastAPI + LangChain (OpenAI) proof-of-concept application written in Python 3.14+. It follows a modular layered architecture under the `app/` directory.

## Project Structure

```
app/
  api/          # SSE Routes and regular endpoints (FastAPI routers)
  core/         # Configuration, settings, startup logic
  schemas/      # Pydantic request/response models
  services/     # Business logic and LangChain integrations (Chains, Prompts, & Streaming)
data/           # Local vector store (e.g., FAISS index) if needed.
tests/          # Pytest scripts
pyproject.toml  # Dependencies
.env            # Secrets (OPENAI_API_KEY)
README.md
```

## Code Style & Conventions

- **Language**: Python 3.14+. Use modern syntax: `match`, `type X = ...`, structural pattern matching where appropriate.
- **Typing**: Use built-in generics (`list[str]`, `dict[str, int]`) — no `from typing import List, Dict`.
- **Async**: Prefer `async def` for FastAPI route handlers and I/O-bound service methods.
- **Pydantic**: Use Pydantic v2 models for all request/response schemas in `app/schemas/`.
- **Settings**: Load configuration via `pydantic-settings` or `python-dotenv` in `app/core/`.
- **Naming**: `snake_case` for variables/functions/modules, `PascalCase` for classes.
- **Imports**: Absolute imports from `app.*`. No wildcard imports.

## FastAPI Guidelines

- Register routers in `app/api/` and include them in `app/main.py`.
- Use dependency injection (`Depends`) for services and settings.
- Return typed response models — always set `response_model` on route decorators.
- Raise `HTTPException` with appropriate status codes; do not return raw dicts for errors.

## LangChain Guidelines

- Place all chain/agent construction in `app/services/`.
- Use `langchain_core` interfaces (`BaseChatModel`, `BaseChain`, `RunnableSequence`) for type hints.
- Prefer LCEL (LangChain Expression Language) pipelines (`|` operator) over legacy `LLMChain`.
- Store prompt templates as `ChatPromptTemplate` instances, not raw strings.
- Never hard-code API keys; read them from environment variables.

## Testing

- Tests live in `tests/`. Mirror the `app/` structure (e.g., `tests/services/`, `tests/api/`).
- Use `pytest` with `pytest-asyncio` for async tests.
- Mock external LLM calls with `unittest.mock` or `pytest-mock`; never hit real APIs in tests.

## Security

- Never commit secrets or API keys.
- Validate all external inputs through Pydantic schemas.
- Keep dependencies up to date; check with `uv pip list --outdated`.

## Environment

- Package manager: `uv` (see `pyproject.toml`).
- Virtual environment: `.venv/`.
- Run locally: `fastapi dev app/main.py`.
