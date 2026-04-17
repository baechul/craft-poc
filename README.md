# Craft Demo Application
For demo purpose, I am using the FastAPI/uvicorn server for both backend and frontend. 
The key techstacks include FastAPI, LangChain, OpenAI LLM, and SSE.

## Getting Started
1. Create [project-root]/.env file
1. Add your own OPENAI_API_KEY='your-key'
1. Run the POC app
```bash
uv sync
uv run uvicorn app.main:app --reload
```
4. Open http://127.0.0.1:8000 for a simple Chat UI.
5. Enter like "How's my top 3 sales by category for the next 3 months?"

## Screenshots
<img src="./images/screen2
.png" width="600" height="auto">

<img src="./images/screen3
.png" width="600" height="auto">