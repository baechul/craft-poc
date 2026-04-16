---
agent: agent
description: Create a new LangChain LCEL service in app/services/
---

Create a new LangChain service for the `craft-poc` application following LCEL conventions.

## Checklist

- [ ] File lives in `app/services/<name>_service.py`.
- [ ] Import from `langchain_core` (`BaseChatModel`, `RunnableSequence`, `ChatPromptTemplate`, etc.).
- [ ] Build the pipeline with the `|` operator — no legacy `LLMChain`.
- [ ] Wrap the prompt as a `ChatPromptTemplate.from_messages([...])` instance.
- [ ] Expose an `async def` method; accept typed parameters, return a typed result.
- [ ] Read the OpenAI API key and model name from environment / `app/core/` settings — never hard-code them.
- [ ] Write a unit test in `tests/services/test_<name>_service.py` that mocks the LLM with `pytest-mock`.

## Input

Describe the service:

- What task should the LLM perform?
- What inputs does the chain accept?
- What should the output look like?
- Any special prompt engineering requirements?
