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
        catalog_types = {item.lower() for item in assessment.assessment_types}
        if constraints.assessment_types and not catalog_types.intersection(constraints.assessment_types):
            continue
        if constraints.max_duration_minutes is not None and assessment.duration_minutes > constraints.max_duration_minutes:
            continue
        if constraints.remote_testing is not None and assessment.remote_testing != constraints.remote_testing:
            continue
        if constraints.adaptive is not None and assessment.adaptive != constraints.adaptive:
            continue
        filtered.append(assessment)
    return filtered


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
    ranked = sorted(zip(scores, candidates), key=lambda item: item[0], reverse=True)
    return [assessment for score, assessment in ranked[: constraints.max_results] if score > 0]


def find_assessments_in_text(text: str, catalog: list[Assessment]) -> list[Assessment]:
    lowered = text.lower()
    found = []
    for assessment in catalog:
        name_words = [word for word in re.findall(r"[a-z0-9+#.]+", assessment.name.lower()) if len(word) > 2]
        if assessment.name.lower() in lowered or sum(word in lowered for word in name_words) >= 2:
            found.append(assessment)
    return found
