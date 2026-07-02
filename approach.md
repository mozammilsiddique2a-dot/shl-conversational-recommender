# Approach

## Problem Understanding

The assignment asks for a conversational SHL assessment recommender that runs without a paid LLM API. The system must accept stateless chat history, understand the latest user need in context, recommend only catalog-backed SHL Individual Test Solutions, refuse unsafe or off-topic requests, and return a strict API response shape.

## API Design

The FastAPI app exposes two public endpoints:

- `GET /health` returns exactly `{"status":"ok"}`.
- `POST /chat` accepts `messages` with `role` and `content`, then returns only `reply`, `recommendations`, and `end_of_conversation`.

The API is stateless. Each `/chat` call uses the full message history supplied by the client and stores no server-side conversation state.

## Catalog Scraping

The scraper in `scripts/scrape_catalog.py` targets SHL's public product catalog and writes normalized records to `app/data/catalog.json`. It collects `name`, `url`, `test_type`, `description`, and `category` when available. It uses polite headers, request timeouts, duplicate removal, absolute URL handling, detail-page enrichment when possible, and fallback logging. If scraping fails or times out, it does not overwrite the existing catalog unless explicitly requested.

## Retrieval Approach

Recommendations come only from `app/data/catalog.json`; names, URLs, and test types are never invented. Retrieval uses scikit-learn TF-IDF over catalog metadata such as name, description, test type, job family, and skills. The user query is built from the full conversation with extra weight on the latest user message. Results are ranked by cosine similarity and capped at 10 items.

## Conversation Logic

Rule-based logic handles the main interaction types:

- Vague requests receive one clarification question and no recommendations.
- Specific role or skill requests retrieve matching assessments.
- Refinement turns, such as adding personality or cognitive tests, are interpreted using the full conversation history.
- Comparison requests use catalog data only and avoid unsupported claims.
- If no catalog match is found, the assistant asks for more specific role, skill, or assessment-type details.

## Guardrails

The guardrail layer blocks prompt injection, hidden-prompt requests, legal advice, medical advice, financial or salary advice, political topics, general hiring strategy, and requests for products outside the SHL catalog. It uses simple case-insensitive keyword and pattern checks and returns a concise refusal string for the API layer to format.

## Evaluation and Testing

Tests use pytest and FastAPI TestClient. They cover `/health`, strict `/chat` schema, vague and empty queries, Java developer recommendations, catalog URL provenance, off-topic refusals, prompt-injection refusals, comparison behavior, additive refinement behavior, guardrails, and TF-IDF retrieval filters. The final local run passed 20 tests.

## Deployment

The project includes a `Dockerfile` and `render.yaml` for Render deployment. Render can deploy the app as a Docker web service with `/health` as the health check endpoint. The public API exposes only `GET /health` and `POST /chat`.

## What Did Not Work

The original SHL catalog URL redirected to SHL's current product pages, so the scraper was adjusted to collect product-specific assessment URLs from the current site structure. Also, the local machine's global Python installation contains a broken `py.py` file that interferes with plain `pytest`; running tests through the virtual environment with site-packages prioritized worked.

## AI Tools Used

Codex was used to scaffold and refine the FastAPI app, scraper, retrieval logic, guardrails, tests, README, Dockerfile, Render config, and this approach document. AI-generated code was reviewed, understood, tested, and refined before deployment. The implementation remains deterministic and does not depend on any paid LLM API at runtime.
