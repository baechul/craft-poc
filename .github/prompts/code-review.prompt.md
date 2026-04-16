---
agent: agent
description: Review code for quality, security, and project conventions
---

Review the following code from the `craft-poc` project.

## Review Criteria

### Correctness
- Logic errors, off-by-one issues, unhandled edge cases.

### Security (OWASP Top 10)
- No secrets or API keys in code.
- All external inputs validated through Pydantic schemas.
- No SQL/command injection vectors.

### Code Style
- Python 3.14+ syntax; built-in generics (`list[str]` not `List[str]`).
- `snake_case` identifiers, `PascalCase` classes.
- Absolute imports from `app.*`; no wildcard imports.

### FastAPI Conventions
- `response_model` set on all route decorators.
- `Depends` used for DI; no manual instantiation in route bodies.
- `HTTPException` raised for errors — no raw dict returns.

### LangChain Conventions
- LCEL pipelines (`|`) used — no legacy `LLMChain`.
- `ChatPromptTemplate` for all prompts — no raw strings.
- API keys sourced from environment / settings only.

### Tests
- External calls mocked; no real API usage in tests.
- Both happy path and error cases covered.

## Input

Paste the code to review or reference the file path.
