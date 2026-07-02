import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=4000)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(default_factory=list, max_length=30)


class ChatRecommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reply: str
    recommendations: list[ChatRecommendation]
    end_of_conversation: bool


class Assessment(BaseModel):
    id: str
    name: str
    url: str
    description: str
    assessment_types: list[str]
    duration_minutes: int = 30
    remote_testing: bool = True
    adaptive: bool = False
    job_families: list[str] = Field(default_factory=list)
    skills: list[str] = Field(default_factory=list)
    category: str | None = None

    @model_validator(mode="before")
    @classmethod
    def accept_scraped_catalog_shape(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        copied = dict(data)
        if "id" not in copied and "name" in copied:
            copied["id"] = re.sub(r"[^a-z0-9]+", "-", copied["name"].lower()).strip("-")
        if "assessment_types" not in copied:
            test_type = copied.get("test_type") or copied.get("category") or "Unknown"
            copied["assessment_types"] = [part.strip() for part in str(test_type).split(",") if part.strip()]
        return copied

    @property
    def test_type(self) -> str:
        return ", ".join(self.assessment_types)
