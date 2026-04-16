---
agent: agent
description: Write pytest tests for an existing module in app/
---

Write tests for the specified `craft-poc` module.

## Rules

- Mirror the source path under `tests/` (e.g. `app/services/foo.py` → `tests/services/test_foo.py`).
- Use `pytest` and `pytest-asyncio`; mark async tests with `@pytest.mark.asyncio`.
- **Never** call real APIs or LLMs. Mock with `pytest-mock` (`mocker.patch`) or `unittest.mock`.
- Use `httpx.AsyncClient` + `app` fixture for FastAPI route tests.
- Cover: happy path, validation errors, and at least one edge case per function.
- Assert on response status codes *and* response body fields for API tests.

## Input

Paste the source file(s) you want tested, or reference them by path.
Describe any notable edge cases you want covered.
