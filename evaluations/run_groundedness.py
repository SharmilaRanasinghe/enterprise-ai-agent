
import asyncio
import json
import uuid

from pathlib import Path
from dotenv import load_dotenv

load_dotenv(
    Path(__file__).resolve().parents[1] / ".env"
)

from langchain_core.messages import HumanMessage

from backend.agents.graph import graph

from evaluations.groundedness import (
    evaluate_groundedness,
)


BASE_DIR = Path(__file__).resolve().parent


async def main():

    with (BASE_DIR / "dataset.json").open(
        encoding="utf-8"
    ) as file:

        dataset = json.load(file)

    examples = [
        example
        for example in dataset
        if example["expected_intent"] in {
            "knowledge_search",
            "research",
        }
        and example.get("expected_sources")
    ]

    results = []

    for example in examples:

        print(
            f"\nEvaluating {example['id']}: "
            f"{example['question']}"
        )

        state = await graph.ainvoke(
            {
                "messages": [
                    HumanMessage(
                        content=example["question"]
                    )
                ],
                "intent": "",
                "user_id": (
                    f"evaluation-{example['role']}"
                ),
                "user_role": example["role"],
            },
            config={
                "configurable": {
                    "thread_id": str(uuid.uuid4())
                },
                "run_name": "groundedness-evaluation",
            },
        )

        answer = state["messages"][-1].content

        if not isinstance(answer, str):
            answer = state["messages"][-1].text

            if callable(answer):
                answer = answer()

        if any(
            phrase in answer.lower()
            for phrase in (
                "temporarily unavailable",
                "could not be completed",
            )
        ):
            results.append({
                "id": example["id"],
                "passed": False,
                "reason": "Application unavailable",
            })

            continue

        # Match the actual context used by response_agent.
        sources = state.get("sources", [])[:4]

        context = [
            {
                "citation": index,
                "source": source.get("source"),
                "title": source.get("title"),
                "text": source.get("text", ""),
            }
            for index, source in enumerate(
                sources, start=1
            )
        ]

        if not context or not all(
            item["text"].strip()
            for item in context
        ):

            results.append({
                "id": example["id"],
                "passed": False,
                "reason": "Retrieved context missing",
            })

            continue

        try:
            evaluation = await evaluate_groundedness(
                question=example["question"],
                answer=answer,
                context=context,
            )

        except Exception as exc:

            results.append({
                "id": example["id"],
                "passed": False,
                "reason": type(exc).__name__,
            })

            continue

        result = {
            "id": example["id"],
            **evaluation,
        }

        results.append(result)

        print(json.dumps(result, indent=2))

    scored_results = [
        result for result in results
        if "groundedness_score" in result
    ]

    passed = sum(
        result["passed"]
        for result in results
    )

    summary = {
        "total_cases": len(results),
        "passed_cases": passed,
        "failed_cases": len(results) - passed,
        "pass_rate_pct": round(
            passed / len(results) * 100, 2
        ) if results else None,
        "average_groundedness_score": round(
            sum(
                result["groundedness_score"]
                for result in scored_results
            ) / len(scored_results),
            2,
        ) if scored_results else None,
    }

    print("\nGroundedness Summary")

    print(json.dumps(summary, indent=2))

    with (BASE_DIR / "groundedness_results.json").open(
        "w", encoding="utf-8"
    ) as file:

        json.dump({
            "summary": summary,
            "results": results,
        }, file, indent=2)


if __name__ == "__main__":
    asyncio.run(main())

