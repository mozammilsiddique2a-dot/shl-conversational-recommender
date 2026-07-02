# SHL Assessment Recommender

Conversational FastAPI service for the SHL AI Intern take-home assignment. It recommends SHL Individual Test Solutions from a local catalog without requiring a paid LLM API.

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Interactive docs are available at `http://127.0.0.1:8000/docs`.

## API

### `GET /health`

Returns exactly:

```json
{"status": "ok"}
```

### `POST /chat`

Request:

```json
{
  "messages": [
    {"role": "user", "content": "Recommend a Python developer assessment under 45 minutes"}
  ]
}
```

Response:

```json
{
  "reply": "I found 1 catalog matches. The strongest options are Python Programming Assessment.",
  "recommendations": [
    {
      "name": "Python Programming Assessment",
      "url": "https://www.shl.com/solutions/products/product-catalog/",
      "test_type": "Knowledge and Skills, Technical"
    }
  ],
  "end_of_conversation": true
}
```

The API is stateless. Send the complete conversation history in each `/chat` request.

## Behavior

- Vague requests receive one clarification question and no recommendations.
- Specific requests return 1 to 10 recommendations from `app/data/catalog.json`.
- URLs are copied only from the catalog.
- Constraint changes are handled by rereading the full message history.
- Comparison requests use catalog fields only.
- Off-topic, legal, general hiring-advice, and prompt-injection requests are refused.

## Tests

```bash
pytest
```

If local Python path issues interfere with pytest, run it through the virtual environment:

```bash
.\venv\Scripts\python.exe -c "import site, sys; sys.path.insert(0, site.getsitepackages()[1]); import pytest; raise SystemExit(pytest.main(['-v']))"
```

## Docker

```bash
docker build -t shl-assessment-recommender .
docker run -p 8000:8000 shl-assessment-recommender
```

## Deployment Checklist

### Render

1. Push the repository to GitHub.
2. Create a new Render Web Service.
3. Select the repository.
4. Use Docker deployment with the included `Dockerfile`, or use:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Confirm the service starts without errors.
6. Test public endpoints:
   - `GET /health`
   - `POST /chat`

### Railway

1. Create a new Railway project from the GitHub repository.
2. Let Railway detect the Dockerfile, or configure:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
3. Generate the public domain.
4. Test public endpoints:
   - `GET /health`
   - `POST /chat`

## Submission

Public API URL: https://your-app-name.onrender.com

Health URL: https://your-app-name.onrender.com/health

Chat URL: https://your-app-name.onrender.com/chat
