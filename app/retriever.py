import re
from dataclasses import dataclass, field

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas import Assessment


ROLE_KEYWORDS = {
    "analyst",
    "data",
    "developer",
    "engineer",
    "graduate",
    "leader",
    "leadership",
    "manager",
    "sales",
    "service",
    "software",
}
SKILL_KEYWORDS = {
    "algorithms",
    "automation",
    "coaching",
    "communication",
    "debugging",
    "java",
    "leadership",
    "negotiation",
    "python",
    "reasoning",
    "resilience",
    "sql",
}
TYPE_ALIASES = {
    "ability": "cognitive ability",
    "behavioral": "behavioral",
    "cognitive": "cognitive ability",
    "coding": "technical",
    "personality": "personality",
    "simulation": "simulation",
    "sjt": "situational judgement",
    "situational": "situational judgement",
    "technical": "technical",
}
TECHNICAL_ROLE_TERMS = {
    "backend",
    "coding",
    "developer",
    "devops",
    "engineer",
    "engineering",
    "frontend",
    "java",
    "javascript",
    "programmer",
    "programming",
    "python",
    "software",
    "sql",
    "technical",
}
TECHNICAL_ASSESSMENT_TERMS = {
    "coding",
    "coding simulations",
    "developer",
    "engineering",
    "java",
    "knowledge and skills",
    "programming",
    "python",
    "software",
    "technical",
    "technical skills",
}
PREFERRED_TECHNICAL_PRODUCT_TERMS = {
    "coding simulations",
    "technical skills",
}
BUSINESS_COMMUNICATION_QUERY_TERMS = {
    "business communication",
    "business skills",
    "communication",
    "presentation",
    "stakeholder",
    "stakeholder communication",
}
LANGUAGE_QUERY_TERMS = {
    "english",
    "language",
    "language evaluation",
    "verbal",
    "writing",
}
SECONDARY_TECHNICAL_PRODUCTS = {
    "business skills",
    "language evaluation",
}
IRRELEVANT_FOR_TECHNICAL_TERMS = {
    "call center",
    "commercial",
    "contact center",
    "customer service",
    "customer success",
    "hospitality",
    "negotiation",
    "retail",
    "sales",
    "service orientation",
}


@dataclass
class SearchConstraints:
    assessment_types: set[str] = field(default_factory=set)
    max_duration_minutes: int | None = None
    remote_testing: bool | None = None
    adaptive: bool | None = None
    max_results: int = 5


def assessment_document(assessment: Assessment) -> str:
    return " ".join(
        [
            assessment.name,
            assessment.description,
            " ".join(assessment.assessment_types),
            " ".join(assessment.job_families),
            " ".join(assessment.skills),
        ]
    )


def is_technical_software_query(query: str) -> bool:
    query_terms = set(re.findall(r"[a-z0-9+#.]+", query.lower()))
    return bool(query_terms.intersection(TECHNICAL_ROLE_TERMS))


def _contains_any(text: str, terms: set[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _asks_for_business_communication(query: str) -> bool:
    return _contains_any(query, BUSINESS_COMMUNICATION_QUERY_TERMS)


def _asks_for_language(query: str) -> bool:
    return _contains_any(query, LANGUAGE_QUERY_TERMS)


def _domain_adjustment(query: str, assessment: Assessment) -> float:
    if not is_technical_software_query(query):
        return 0.0

    document = assessment_document(assessment).lower()
    score = 0.0

    if _contains_any(document, PREFERRED_TECHNICAL_PRODUCT_TERMS):
        score += 1.0
    if _contains_any(document, TECHNICAL_ASSESSMENT_TERMS):
        score += 0.45
    if "technical" in {item.lower() for item in assessment.assessment_types}:
        score += 0.35
    if "business skills" in document and not _asks_for_business_communication(query):
        score -= 0.8
    if "language evaluation" in document and not _asks_for_language(query):
        score -= 0.8
    if _contains_any(document, SECONDARY_TECHNICAL_PRODUCTS):
        score -= 0.15
    if _contains_any(document, IRRELEVANT_FOR_TECHNICAL_TERMS):
        score -= 0.55
    if not _contains_any(document, TECHNICAL_ASSESSMENT_TERMS) and not any(
        item.lower() in {"personality", "cognitive ability"} for item in assessment.assessment_types
    ):
        score -= 0.35

    return score


def is_vague(text: str) -> bool:
    words = set(re.findall(r"[a-z0-9+#.]+", text.lower()))
    if len(words) < 3:
        return True
    known_signals = ROLE_KEYWORDS | SKILL_KEYWORDS | set(TYPE_ALIASES)
    return not words.intersection(known_signals)


def extract_constraints(text: str) -> SearchConstraints:
    lowered = text.lower()
    constraints = SearchConstraints()

    for word, test_type in TYPE_ALIASES.items():
        if re.search(rf"\b{re.escape(word)}\b", lowered):
            constraints.assessment_types.add(test_type)

    duration_match = re.search(r"(?:under|within|max(?:imum)?|less than)\s+(\d{1,3})\s*(?:min|mins|minutes)?", lowered)
    if duration_match:
        constraints.max_duration_minutes = int(duration_match.group(1))

    count_match = re.search(r"\b(?:top|recommend|show|give me)\s+(\d{1,2})\b", lowered)
    if count_match:
        constraints.max_results = min(max(int(count_match.group(1)), 1), 10)

    if re.search(r"\bremote\b", lowered):
        constraints.remote_testing = not bool(re.search(r"\b(no|not|without)\s+remote\b", lowered))
    if re.search(r"\badaptive\b", lowered):
        constraints.adaptive = not bool(re.search(r"\b(no|not|non)\s*-?\s*adaptive\b", lowered))

    return constraints


def merge_constraints(history_text: str, latest_text: str) -> SearchConstraints:
    merged = extract_constraints(history_text)
    latest = extract_constraints(latest_text)
    if latest.assessment_types:
        merged.assessment_types = latest.assessment_types
    if latest.max_duration_minutes is not None:
        merged.max_duration_minutes = latest.max_duration_minutes
    if latest.remote_testing is not None:
        merged.remote_testing = latest.remote_testing
    if latest.adaptive is not None:
        merged.adaptive = latest.adaptive
    if latest.max_results != SearchConstraints().max_results:
        merged.max_results = latest.max_results
    return merged


def filter_catalog(catalog: list[Assessment], constraints: SearchConstraints) -> list[Assessment]:
    filtered: list[Assessment] = []
    for assessment in catalog:
        if constraints.assessment_types and not any(
            _assessment_matches_type(assessment, requested_type)
            for requested_type in constraints.assessment_types
        ):
            continue
        if constraints.max_duration_minutes is not None and assessment.duration_minutes > constraints.max_duration_minutes:
            continue
        if constraints.remote_testing is not None and assessment.remote_testing != constraints.remote_testing:
            continue
        if constraints.adaptive is not None and assessment.adaptive != constraints.adaptive:
            continue
        filtered.append(assessment)
    return filtered


def _assessment_matches_type(assessment: Assessment, requested_type: str) -> bool:
    requested = requested_type.lower()
    document = assessment_document(assessment).lower()
    catalog_types = {item.lower() for item in assessment.assessment_types}
    if requested in catalog_types:
        return True
    if requested == "technical":
        return "knowledge and skills" in catalog_types or _contains_any(document, PREFERRED_TECHNICAL_PRODUCT_TERMS)
    if requested == "cognitive ability":
        return "cognitive ability" in catalog_types or "verify" in document
    return False


def retrieve_assessments(
    query: str,
    catalog: list[Assessment],
    constraints: SearchConstraints,
) -> list[Assessment]:
    candidates = filter_catalog(catalog, constraints)
    if not candidates:
        return []

    documents = [assessment_document(item) for item in candidates]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(documents + [query])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    adjusted_scores = [
        (float(score) + _domain_adjustment(query, assessment), assessment)
        for score, assessment in zip(scores, candidates)
    ]
    ranked = sorted(adjusted_scores, key=lambda item: item[0], reverse=True)
    if is_technical_software_query(query):
        limit = min(constraints.max_results, 3)
        return [assessment for score, assessment in ranked[:limit] if score > 0.4]
    return [assessment for score, assessment in ranked[: constraints.max_results] if score > 0.05]


def search(query: str, top_k: int = 10) -> list[Assessment]:
    from app.catalog_loader import load_catalog

    return retrieve_assessments(query, load_catalog(), SearchConstraints(max_results=top_k))


def find_assessments_in_text(text: str, catalog: list[Assessment]) -> list[Assessment]:
    lowered = text.lower()
    found = []
    for assessment in catalog:
        name_words = [word for word in re.findall(r"[a-z0-9+#.]+", assessment.name.lower()) if len(word) > 2]
        if assessment.name.lower() in lowered or sum(word in lowered for word in name_words) >= 2:
            found.append(assessment)
    return found
