from fastapi import FastAPI

from app.agent import generate_response
from app.schemas import ChatRequest, ChatResponse


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    description="Conversational recommender for SHL Individual Test Solutions.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest) -> ChatResponse:
    return generate_response(payload.messages)
