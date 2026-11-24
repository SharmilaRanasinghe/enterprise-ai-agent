
import asyncio
import json
import time
import uuid
from pathlib import Path

import httpx

from backend.config import settings
from evaluations.evaluators import evaluate_result
from evaluations.llm_judge import judge_answer

BASE_DIR = Path(__file__).resolve().parent

API_URL = "http://127.0.0.1:8000/api/chat"

API_KEYS = {
    "viewer": settings.demo_viewer_api_key,
    "engineer": settings.demo_engineer_api_key,
    "admin": settings.demo_admin_api_key,
}


async def evaluate_case(
    client: httpx.AsyncClient,
    example: dict,
) -> dict:

    start = time.perf_counter()

    try:
        response = await client.post(
            API_URL,
            headers={
                "X-API-Key": API_KEYS[example["role"]]
            },
            json={
                "message": example["question"],
                "session_id": str(uuid.uuid4()),
            },
        )

        response.raise_for_status()

        result = response.json()

        latency_ms = round(
            (time.perf_counter() - start) * 1000,
            2,
        )

        scores = evaluate_result(example, result)

        judge_result = None

        if example.get("reference_answer"):
            if scores["response_available"]:
                try:
                    judge_result = await judge_answer(
                        question=example["question"],
                        reference_answer=example["reference_answer"],
                        actual_answer=result["answer"],
                    )

                    scores["semantic_correctness"] = (
                        judge_result["passed"]
                    )

                except Exception as exc:
                    judge_result = {
                        "status": "evaluation_error",
                        "error_type": type(exc).__name__,
                    }

                    scores["semantic_correctness"] = False

            else:
                judge_result = {
                    "status": "skipped",
                    "reason": "Application response unavailable",
                }

                scores["semantic_correctness"] = False

        return {
            "id": example["id"],
            "category": example["category"],
            "question": example["question"],
            "role": example["role"],
            "passed": all(scores.values()),
            "latency_ms": latency_ms,
            "scores": scores,
            "judge_result": judge_result,
            "answer": result.get("answer"),
            "sources": result.get("sources", []),
            "trace_id": result.get("trace_id"),
        }

    except (httpx.HTTPError, ValueError) as exc:
        return {
            "id": example["id"],
            "category": example["category"],
            "passed": False,
            "latency_ms": round(
                (time.perf_counter() - start) * 1000,
                2,
            ),
            "error": str(exc),
        }


async def main():
    with (BASE_DIR / "dataset.json").open(
        encoding="utf-8"
    ) as file:
        dataset = json.load(file)

    results = []

    async with httpx.AsyncClient(
        timeout=180.0
    ) as client:

        for example in dataset:
            print(
                f"Running {example['id']}: "
                f"{example['question']}",
                flush=True,
            )

            result = await evaluate_case(
                client, example
            )

            results.append(result)

            status = (
                "PASS" if result["passed"] else "FAIL"
            )

            print(
                f"  {status} | "
                f"{result['latency_ms']} ms"
            )

    total = len(results)

    passed = sum(
        result["passed"]
        for result in results
    )

    summary = {
        "total_cases": total,
        "passed_cases": passed,
        "failed_cases": total - passed,
        "pass_rate_pct": round(
            passed / total * 100, 2
        ) if total else 0,
        "average_latency_ms": round(
            sum(
                result["latency_ms"]
                for result in results
            ) / total,
            2,
        ) if total else 0,
    }

    judged_results = [
        result["judge_result"]
        for result in results
        if result.get("judge_result")
        and "average_score" in result["judge_result"]
    ]

    semantic_passed = sum(
        result["passed"]
        for result in judged_results
    )

    summary["semantic_evaluated_cases"] = len(
        judged_results
    )

    summary["semantic_passed_cases"] = semantic_passed

    summary["semantic_pass_rate_pct"] = (
        round(
            semantic_passed / len(judged_results) * 100,
            2,
        )
        if judged_results else None
    )

    summary["average_semantic_score"] = (
        round(
            sum(
                result["average_score"]
                for result in judged_results
            ) / len(judged_results),
            2,
        )
        if judged_results else None
    )

    summary["semantic_evaluation_errors"] = sum(
        1 for result in results
        if (result.get("judge_result") or {}).get("status")
        == "evaluation_error"
    )
    report = {
        "summary": summary,
        "results": results,
    }

    output_file = BASE_DIR / "results.json"

    with output_file.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, indent=2)

    print("\nEvaluation Summary")
    print(json.dumps(summary, indent=2))

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())