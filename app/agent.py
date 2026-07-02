import re
from typing import Any

from app.catalog_loader import load_catalog
from app.guardrails import check_guardrails as external_check_guardrails
from app import retriever


ChatMessage = dict[str, str]
RecommendationDict = dict[str, str]
AgentResponse = dict[str, Any]

REFUSAL_REPLY = "I can only help with SHL assessment recommendations and catalog-based comparisons."
CLARIFY_REPLY = "What role, skills, seniority, or assessment type should I use to recommend SHL assessments?"

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(system|developer)\s+prompt", re.IGNORECASE),
    re.compile(r"show\s+me\s+(your\s+)?prompt", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:dan|an?\s+unrestricted)", re.IGNORECASE),
    re.compile(r"<\s*script", re.IGNORECASE),
]
OFF_TOPIC_PATTERNS = [
    re.compile(r"\b(contract|lawsuit|legal|attorney|lawyer|visa|immigration)\b", re.IGNORECASE),
    re.compile(r"\b(salary|compensation|offer letter|background check)\b", re.IGNORECASE),
    re.compile(r"\b(general hiring advice|hiring strategy|recruiting strategy|performance review)\b", re.IGNORECASE),
]
VAGUE_PATTERNS = [
    re.compile(r"^\s*i need an? assessment\s*$", re.IGNORECASE),
    re.compile(r"^\s*help me hire\s*$", re.IGNORECASE),
    re.compile(r"^\s*suggest an? test\s*$", re.IGNORECASE),
    re.compile(r"^\s*assessment for candidate\s*$", re.IGNORECASE),
]
COMPARISON_PATTERNS = [
    re.compile(r"\bcompare\b", re.IGNORECASE),
    re.compile(r"\bdifference between\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
]
ROLE_OR_SKILL_WORDS = {
    "analyst",
    "automation",
    "candidate",
    "cognitive",
    "communication",
    "customer",
    "data",
    "developer",
    "engineer",
    "graduate",
    "java",
    "leader",
    "leadership",
    "manager",
    "personality",
    "python",
    "sales",
    "service",
    "software",
    "sql",
    "technical",
}
TEST_TYPE_ALIASES = {
    "ability": "cognitive",
    "behavioral": "behavioral",
    "cognitive": "cognitive",
    "coding": "technical",
    "personality": "personality",
    "simulation": "simulation",
    "situational": "situational",
    "technical": "technical",
}


def _message_value(message: Any, key: str) -> str:
    if isinstance(message, dict):
        return str(message.get(key, ""))
    return str(getattr(message, key, ""))


def latest_user_message(messages: list[dict]) -> str:
    for message in reversed(messages):
        if _message_value(message, "role") == "user":
            return _clean_text(_message_value(message, "content"))
    return ""


def build_conversation_text(messages: list[dict]) -> str:
    parts = []
    for message in messages:
        role = _message_value(message, "role")
        content = _clean_text(_message_value(message, "content"))
        if role in {"user", "assistant"} and content:
            parts.append(f"{role}: {content}")
    return " ".join(parts)


def is_vague_query(text: str) -> bool:
    cleaned = _clean_text(text)
    if not cleaned:
        return True
    if any(pattern.search(cleaned) for pattern in VAGUE_PATTERNS):
        return True
    words = set(re.findall(r"[a-z0-9+#.]+", cleaned.lower()))
    if len(words) <= 4 and words.intersection({"assessment", "assessments", "test", "tests", "hire", "candidate"}):
        return not words.intersection(ROLE_OR_SKILL_WORDS)
    return not words.intersection(ROLE_OR_SKILL_WORDS) and len(words) < 6


def is_comparison_query(text: str) -> bool:
    return any(pattern.search(text) for pattern in COMPARISON_PATTERNS)


def extract_query_context(conversation_text: str) -> dict[str, Any]:
    lowered = conversation_text.lower()
    context: dict[str, Any] = {
        "job_role": _first_match(lowered, [r"\b(data analyst|software engineer|java developer|python developer|sales|manager|leader|customer service)\b"]),
        "seniority": _first_match(lowered, [r"\b(intern|junior|entry level|mid level|senior|lead|manager)\b"]),
        "years_experience": _first_match(lowered, [r"\b(\d+\+?\s*(?:years|yrs))\b"]),
        "technical_skills": _find_terms(lowered, ["python", "java", "sql", "algorithms", "debugging", "automation", "data structures"]),
        "soft_skills": _find_terms(lowered, ["communication", "leadership", "coaching", "negotiation", "resilience", "collaboration"]),
        "include_types": _extract_type_changes(lowered, verbs=("add", "include", "need", "want", "with")),
        "exclude_types": _extract_type_changes(lowered, verbs=("remove", "exclude", "without", "no")),
        "max_results": _extract_top_k(lowered),
    }
    if "personality" in lowered:
        context["personality_required"] = True
    if "cognitive" in lowered or "ability" in lowered:
        context["cognitive_required"] = True
    return context


def format_recommendations(items: list[Any]) -> list[RecommendationDict]:
    formatted: list[RecommendationDict] = []
    for item in items[:10]:
        name = _item_value(item, "name")
        url = _item_value(item, "url")
        test_type = _item_test_type(item)
        if name and url and test_type:
            formatted.append({"name": name, "url": url, "test_type": test_type})
    return formatted


def generate_response(messages: list[dict]) -> dict:
    latest = latest_user_message(messages)
    conversation_text = build_conversation_text(messages)

    refusal = _guardrail_refusal(conversation_text, latest)
    if refusal:
        return _response(refusal, [], True)

    if is_comparison_query(latest):
        catalog = load_catalog()
        compared_items = _find_catalog_items_in_text(conversation_text, catalog)
        if len(compared_items) < 2:
            return _response("Please name at least two SHL catalog assessments to compare.", [], False)
        reply = _format_comparison(compared_items[:4])
        return _response(reply, format_recommendations(compared_items[:4]), True)

    if is_vague_query(latest) and is_vague_query(conversation_text):
        return _response(CLARIFY_REPLY, [], False)

    context = extract_query_context(conversation_text)
    query = _build_search_query(conversation_text, latest, context)
    top_k = int(context.get("max_results") or 10)
    recommendations = _search_catalog(query, top_k=top_k)
    recommendations = _apply_context_filters(recommendations, context)

    if not recommendations:
        return _response("I could not find a catalog match. Which role, skill, or assessment type should I prioritize?", [], False)

    formatted = format_recommendations(recommendations)
    if not formatted:
        return _response("I found possible matches, but they were missing required catalog fields. Please try a more specific SHL assessment request.", [], False)

    names = ", ".join(item["name"] for item in formatted[:3])
    if len(formatted) == 1:
        reply = f"The best SHL catalog match is {names}."
    else:
        reply = f"I found {len(formatted)} SHL catalog matches. The strongest options are {names}."
    return _response(reply, formatted, True)


def _response(reply: str, recommendations: list[RecommendationDict], end_of_conversation: bool) -> AgentResponse:
    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": end_of_conversation,
    }


def _guardrail_refusal(conversation_text: str, latest: str) -> str | None:
    if any(pattern.search(latest) for pattern in PROMPT_INJECTION_PATTERNS):
        return "I can only help recommend SHL assessments from the catalog, so I cannot follow that instruction."
    if any(pattern.search(latest) for pattern in OFF_TOPIC_PATTERNS):
        return REFUSAL_REPLY

    try:
        decision = external_check_guardrails(conversation_text)  # type: ignore[arg-type]
    except Exception:
        return None

    if isinstance(decision, str):
        return decision
    if decision is None:
        return None
    if getattr(decision, "allowed", True) is False:
        return getattr(decision, "reply", None) or REFUSAL_REPLY
    return None


def _search_catalog(query: str, top_k: int) -> list[Any]:
    top_k = min(max(top_k, 1), 10)
    search = getattr(retriever, "search", None)
    if callable(search):
        try:
            return list(search(query, top_k=top_k))
        except Exception:
            pass

    catalog = load_catalog()
    retrieve_assessments = getattr(retriever, "retrieve_assessments", None)
    constraints_cls = getattr(retriever, "SearchConstraints", None)
    if callable(retrieve_assessments) and constraints_cls is not None:
        try:
            return list(retrieve_assessments(query, catalog, constraints_cls(max_results=top_k)))
        except Exception:
            pass
    return list(catalog)[:top_k]


def _apply_context_filters(items: list[Any], context: dict[str, Any]) -> list[Any]:
    include_types = {term.lower() for term in context.get("include_types", set())}
    exclude_types = {term.lower() for term in context.get("exclude_types", set())}
    filtered = []
    for item in items:
        test_type = _item_test_type(item).lower()
        if include_types and not any(term in test_type for term in include_types):
            continue
        if exclude_types and any(term in test_type for term in exclude_types):
            continue
        filtered.append(item)
    return filtered[:10]


def _find_catalog_items_in_text(text: str, catalog: list[Any]) -> list[Any]:
    lowered = text.lower()
    found = []
    for item in catalog:
        name = _item_value(item, "name")
        name_words = [word for word in re.findall(r"[a-z0-9+#.]+", name.lower()) if len(word) > 2]
        if name.lower() in lowered or sum(word in lowered for word in name_words) >= 2:
            found.append(item)
    return found


def _format_comparison(items: list[Any]) -> str:
    lines = []
    for item in items:
        details = [
            _item_test_type(item),
            _optional_detail(item, "description"),
            _optional_detail(item, "category", prefix="category: "),
        ]
        lines.append(f"{_item_value(item, 'name')}: {'; '.join(detail for detail in details if detail)}")
    return " ".join(lines)


def _build_search_query(conversation_text: str, latest: str, context: dict[str, Any]) -> str:
    context_parts = []
    for key in ("job_role", "seniority", "years_experience"):
        if context.get(key):
            context_parts.append(str(context[key]))
    context_parts.extend(context.get("technical_skills", []))
    context_parts.extend(context.get("soft_skills", []))
    context_parts.extend(context.get("include_types", []))
    return " ".join([conversation_text, latest, latest, " ".join(context_parts)]).strip()


def _item_value(item: Any, key: str) -> str:
    if isinstance(item, dict):
        return _clean_text(str(item.get(key, "")))
    return _clean_text(str(getattr(item, key, "")))


def _item_test_type(item: Any) -> str:
    direct = _item_value(item, "test_type")
    if direct:
        return direct
    if isinstance(item, dict):
        assessment_types = item.get("assessment_types", [])
    else:
        assessment_types = getattr(item, "assessment_types", [])
    if isinstance(assessment_types, list):
        return ", ".join(str(value) for value in assessment_types if value)
    return _clean_text(str(assessment_types))


def _optional_detail(item: Any, key: str, prefix: str = "") -> str:
    value = _item_value(item, key)
    return f"{prefix}{value}" if value else ""


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def _first_match(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _find_terms(text: str, terms: list[str]) -> list[str]:
    return [term for term in terms if re.search(rf"\b{re.escape(term)}\b", text)]


def _extract_top_k(text: str) -> int:
    match = re.search(r"\b(?:top|recommend|show|give me)\s+(\d{1,2})\b", text)
    if not match:
        return 10
    return min(max(int(match.group(1)), 1), 10)


def _extract_type_changes(text: str, verbs: tuple[str, ...]) -> set[str]:
    found: set[str] = set()
    for alias, canonical in TEST_TYPE_ALIASES.items():
        for verb in verbs:
            if re.search(rf"\b{verb}\b[^.?!;]*\b{alias}\b", text):
                found.add(canonical)
    return found
