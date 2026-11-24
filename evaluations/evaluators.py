import re


def evaluate_intent(example: dict, result: dict) -> bool:
    return (
        example["expected_intent"]
        == result.get("intent")
    )


def evaluate_retrieval(example: dict, result: dict) -> bool:
    expected = set(
        example.get("expected_sources", [])
    )

    actual = {
        source.get("source")
        for source in result.get("sources", [])
    }

    return expected.issubset(actual)


def evaluate_keywords(example: dict, result: dict) -> bool:
    answer = result.get("answer", "").lower()

    return all(
        keyword.lower() in answer
        for keyword in example.get(
            "expected_keywords", []
        )
    )


def evaluate_security(example: dict, result: dict) -> bool:
    actual = {
        source.get("source")
        for source in result.get("sources", [])
    }

    forbidden = set(
        example.get("forbidden_sources", [])
    )

    if example.get("expect_no_sources") and actual:
        return False

    return not bool(actual.intersection(forbidden))


def evaluate_citations(result: dict) -> bool:
    answer = result.get("answer", "")

    citations = {
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    }

    available = {
        source["citation"]
        for source in result.get("sources", [])
        if "citation" in source
    }

    if not available:
        return not citations

    return bool(citations) and citations.issubset(available)


def evaluate_availability(result: dict) -> bool:
    answer = result.get("answer", "").lower()

    failure_messages = (
        "temporarily unavailable",
        "could not be completed",
    )

    return not any(
        message in answer
        for message in failure_messages
    )


def evaluate_result(example: dict, result: dict) -> dict:
    return {
        "intent_correct": evaluate_intent(
            example, result
        ),
        "retrieval_correct": evaluate_retrieval(
            example, result
        ),
        "keywords_correct": evaluate_keywords(
            example, result
        ),
        "security_passed": evaluate_security(
            example, result
        ),
        "citations_valid": evaluate_citations(
            result
        ),
        "response_available": evaluate_availability(
            result
        ),
    }
