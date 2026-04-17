# Craft — AI-Powered Sales Assistant

A proof-of-concept application that combines a **LightGBM sales forecasting model** with a **GPT-4.1-mini agent** to answer natural-language queries about sales predictions. Built with FastAPI, LangChain, and Server-Sent Events (SSE) for real-time streaming responses.

## Tech Stack

| Layer           | Technology                              |
| --------------- | --------------------------------------- |
| Backend / API   | FastAPI, Uvicorn                        |
| AI Agent        | LangChain (LCEL), LangGraph ReAct agent |
| LLM             | OpenAI GPT-4.1-mini                     |
| ML Model        | LightGBM (sales forecasting)            |
| Streaming       | Server-Sent Events (SSE)                |
| Config          | Pydantic Settings, python-dotenv        |
| Package Manager | `uv`                                    |

## Folder Structure

```
app/
  api/          # SSE and REST endpoints (FastAPI routers)
  core/         # Settings and configuration (pydantic-settings)
  models/       # LGBM model artifact and training data
  prompts/      # System prompt for the sales agent
  services/     # LangChain agent construction and SSE streaming logic
  tools/        # LangChain @tool — Sales prediction with feature engineering
```

The agent follows a **ReAct loop**: it receives a natural-language question, decides whether to invoke the `predict_sales` tool, interprets the model output, and streams a formatted answer back to the client via SSE.

## Prerequisites

- Python 3.14+
- [`uv`](https://github.com/astral-sh/uv) package manager
- An OpenAI API key

## Getting Started

**1. Clone the repository and install dependencies**

```bash
git clone https://github.com/baechul/craft-poc.git
cd craft-poc
uv sync
```

**2. Configure environment variables**

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key
```

**3. Start the development server**

```bash
uv run uvicorn app.main:app --reload
```

**4. Open the chat UI**

Navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) and ask a sales question, for example:

> _"What are my top 3 sales categories for the next 3 months?"_

## ML Model

The LightGBM model was trained separately on historical sales data with lag features and seasonal encodings. It is loaded at application startup via FastAPI's lifespan context.

For details on model training and generation, see the companion repository: [baechul/craft-poc-model](https://github.com/baechul/craft-poc-model).

## Screenshots

<img src="./images/screen2.png" width="600" height="auto">

<img src="./images/screen3.png" width="600" height="auto">

## License

MIT © 2026 Baechul Kim
