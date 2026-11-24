
import asyncio
import json

from evaluations.groundedness import (
    evaluate_groundedness,
)


async def main():

    question = (
        "What caused the payment gateway incident?"
    )

    context = [
        {
            "source": "synthetic_test_document.txt",
            "text": (
                "The payment gateway incident was "
                "caused by downstream payment "
                "processor latency. Retry "
                "concurrency was reduced."
            ),
        }
    ]

    answers = {
        "grounded": (
            "The incident was caused by downstream "
            "payment processor latency."
        ),
        "partially_grounded": (
            "The incident was caused by downstream "
            "processor latency and affected "
            "5000 customers."
        ),
        "ungrounded": (
            "The incident was caused by a database "
            "storage failure."
        ),
    }

    for name, answer in answers.items():

        result = await evaluate_groundedness(
            question=question,
            answer=answer,
            context=context,
        )

        print(f"\n{name.upper()}")

        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
