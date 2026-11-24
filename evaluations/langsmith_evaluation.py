import json
import asyncio
import time
import uuid

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
    Path(__file__).resolve().parents[1] / ".env"
)

import httpx

from langsmith import Client
from langsmith.evaluation import aevaluate

from backend.config import settings

from evaluations.evaluators import evaluate_result
from evaluations.llm_judge import judge_answer


BASE_DIR = Path(__file__).resolve().parent

DATASET_NAME = "enterprise-ai-agent-evaluation-v1"

API_URL = "http://127.0.0.1:8000/api/chat"

API_KEYS = {
    "viewer": settings.demo_viewer_api_key,
    "engineer": settings.demo_engineer_api_key,
    "admin": settings.demo_admin_api_key,
}

client = Client()


def create_dataset():

    with (BASE_DIR / "dataset.json").open(
        encoding="utf-8"
    ) as file:
        examples = json.load(file)

    if client.has_dataset(
        dataset_name=DATASET_NAME
    ):
        print("Using existing LangSmith dataset")
        return DATASET_NAME

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Evaluation dataset for the enterprise "
            "banking AI assistant covering routing, "
            "RAG, analytics and security."
        ),
    )

    client.create_examples(
        dataset_id=dataset.id,
        inputs=[
            {
                "question": example["question"],
                "role": example["role"],
            }
            for example in examples
        ],
        outputs=examples,
    )

    print("LangSmith dataset created successfully")

    return DATASET_NAME


async def target(inputs: dict) -> dict:

    start = time.perf_counter()

    async with httpx.AsyncClient(
        timeout=180.0
    ) as http_client:

        response = await http_client.post(
            API_URL,
            headers={
                "X-API-Key": API_KEYS[inputs["role"]]
            },
            json={
                "message": inputs["question"],
                "session_id": str(uuid.uuid4()),
            },
        )

        response.raise_for_status()

        result = response.json()

    result["latency_ms"] = round(
        (time.perf_counter() - start) * 1000,
        2,
    )

    return result


def deterministic_evaluator(
    outputs: dict,
    reference_outputs: dict,
) -> list[dict]:

    scores = evaluate_result(
        reference_outputs,
        outputs,
    )

    return [
        {
            "key": name,
            "score": int(passed),
        }
        for name, passed in scores.items()
    ]


async def semantic_evaluator(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict,
) -> list[dict]:

    reference = reference_outputs.get(
        "reference_answer"
    )

    if not reference:
        return []

    try:
        result = await judge_answer(
            question=inputs["question"],
            reference_answer=reference,
            actual_answer=outputs["answer"],
        )

        return [
            {
                "key": "correctness",
                "score": result["correctness"],
            },
            {
                "key": "completeness",
                "score": result["completeness"],
            },
            {
                "key": "relevance",
                "score": result["relevance"],
            },
            {
                "key": "consistency",
                "score": result["consistency"],
            },
            {
                "key": "semantic_average",
                "score": result["average_score"],
                "comment": result["explanation"],
            },
            {
                "key": "semantic_passed",
                "score": int(result["passed"]),
            },
        ]

    except Exception as exc:
        return [
            {
                "key": "judge_error",
                "score": 1,
                "comment": type(exc).__name__,
            }
        ]


def latency_evaluator(outputs: dict) -> dict:

    return {
        "key": "application_latency_ms",
        "score": outputs["latency_ms"],
    }



async def main():

    dataset_name = create_dataset()

    await aevaluate(
        target,
        data=dataset_name,
        evaluators=[
            deterministic_evaluator,
            semantic_evaluator,
            latency_evaluator,
        ],
        experiment_prefix=(
            "enterprise-ai-agent-baseline"
        ),
        max_concurrency=1,
        metadata={
            "architecture": "LangGraph",
            "retrieval": "Pinecone + BM25 + RRF",
            "model": settings.gemini_model,
            "evaluation_version": "v1",
            "milestone": 7,
        },
    )


if __name__ == "__main__":
    asyncio.run(main())
