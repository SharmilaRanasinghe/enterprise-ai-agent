
import asyncio
import json

from evaluations.llm_judge import judge_answer


async def main():

    question = (
        "What caused the payment gateway incident?"
    )

    reference = (
        "High latency in the downstream payment "
        "processor caused gateway requests to time out."
    )

    answers = {
        "correct": (
            "The downstream payment processor "
            "experienced high latency, causing timeouts."
        ),
        "incorrect": (
            "The incident was caused by a database "
            "storage failure."
        ),
        "contradictory": (
            "The incident was not caused by "
            "downstream processor latency."
        ),
    }

    for name, answer in answers.items():

        result = await judge_answer(
            question=question,
            reference_answer=reference,
            actual_answer=answer,
        )

        print(f"\n{name.upper()}")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
