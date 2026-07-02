from app.catalog_loader import load_catalog
from app.retriever import SearchConstraints, extract_constraints, retrieve_assessments


def test_tfidf_retrieval_returns_python_for_python_query():
    results = retrieve_assessments(
        "python developer algorithms data structures",
        load_catalog(),
        SearchConstraints(max_results=3),
    )

    assert results
    assert any(
        "coding" in item.name.lower() or "technical" in item.name.lower() or "skills" in item.name.lower()
        for item in results
    )


def test_retrieval_applies_duration_filter():
    results = retrieve_assessments(
        "data analyst sql",
        load_catalog(),
        SearchConstraints(max_duration_minutes=35, max_results=10),
    )

    assert all(item.duration_minutes <= 35 for item in results)


def test_extract_constraints_detects_type_and_count():
    constraints = extract_constraints("Show 3 technical tests under 45 minutes")

    assert constraints.assessment_types == {"technical"}
    assert constraints.max_duration_minutes == 45
    assert constraints.max_results == 3
