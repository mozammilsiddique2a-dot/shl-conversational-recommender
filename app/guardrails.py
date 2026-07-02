import re
from collections.abc import Iterable
from re import Pattern


REFUSAL_MESSAGE = "I can only help with SHL assessment recommendations."

OFF_TOPIC_KEYWORDS: tuple[str, ...] = (
    "weather",
    "sports",
    "movie",
    "music",
    "travel",
    "recipe",
    "homework",
    "dating",
    "politics",
    "election",
    "government policy",
    "political",
)

LEGAL_KEYWORDS: tuple[str, ...] = (
    "legal advice",
    "lawyer",
    "attorney",
    "lawsuit",
    "contract",
    "immigration",
    "visa",
    "compliance advice",
)

MEDICAL_KEYWORDS: tuple[str, ...] = (
    "medical advice",
    "doctor",
    "diagnosis",
    "medication",
    "treatment",
    "symptoms",
    "therapy advice",
)

FINANCIAL_KEYWORDS: tuple[str, ...] = (
    "financial advice",
    "investment",
    "stock",
    "crypto",
    "salary advice",
    "compensation advice",
    "negotiate salary",
    "pay range",
)

HIRING_STRATEGY_KEYWORDS: tuple[str, ...] = (
    "hiring strategy",
    "recruiting strategy",
    "how should i hire",
    "interview questions",
    "performance review",
    "background check",
    "offer letter",
)

NON_SHL_PRODUCT_KEYWORDS: tuple[str, ...] = (
    "non-shl",
    "outside shl",
    "other vendors",
    "other assessment providers",
    "recommend books",
    "recommend software",
    "recommend courses",
)

PROMPT_INJECTION_PATTERNS: tuple[Pattern[str], ...] = (
    re.compile(r"ignore\s+(?:all\s+)?(?:previous\s+)?instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(?:the\s+)?(?:system|hidden|developer)\s+prompt", re.IGNORECASE),
    re.compile(r"show\s+(?:the\s+)?hidden\s+prompt", re.IGNORECASE),
    re.compile(r"print\s+(?:your|the)\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"\bdan\s+mode\b", re.IGNORECASE),
    re.compile(r"act\s+as\s+another\s+assistant", re.IGNORECASE),
    re.compile(r"act\s+as\s+(?:dan|an?\s+unrestricted)", re.IGNORECASE),
    re.compile(r"reveal\s+(?:your\s+)?chain\s+of\s+thought", re.IGNORECASE),
    re.compile(r"show\s+(?:your\s+)?chain\s+of\s+thought", re.IGNORECASE),
)


def normalize_text(text: str) -> str:
    """Return lower-cased text with collapsed whitespace for keyword checks."""
    return " ".join(text.lower().strip().split())


def contains_keyword(text: str, keywords: Iterable[str]) -> bool:
    """Return True when any keyword appears in normalized text."""
    normalized = normalize_text(text)
    return any(keyword.lower() in normalized for keyword in keywords)


def matches_pattern(text: str, patterns: Iterable[Pattern[str]]) -> bool:
    """Return True when any compiled regular expression matches text."""
    return any(pattern.search(text) for pattern in patterns)


def is_prompt_injection(text: str) -> bool:
    """Return True for common prompt-injection or hidden-prompt requests."""
    return matches_pattern(text, PROMPT_INJECTION_PATTERNS)


def is_blocked_topic(text: str) -> bool:
    """Return True for topics outside SHL assessment recommendation support."""
    keyword_groups = (
        OFF_TOPIC_KEYWORDS,
        LEGAL_KEYWORDS,
        MEDICAL_KEYWORDS,
        FINANCIAL_KEYWORDS,
        HIRING_STRATEGY_KEYWORDS,
        NON_SHL_PRODUCT_KEYWORDS,
    )
    return any(contains_keyword(text, keywords) for keywords in keyword_groups)


def check_guardrails(text: str) -> str | None:
    """Return None for allowed requests, otherwise a concise refusal string.

    The API layer converts this refusal into its response schema. This function
    never returns JSON and never raises for normal user input.
    """
    if not text or not text.strip():
        return None
    if is_prompt_injection(text) or is_blocked_topic(text):
        return REFUSAL_MESSAGE
    return None
