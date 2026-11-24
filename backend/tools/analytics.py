
import csv
from pathlib import Path

from langchain_core.tools import tool


DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "transactions.csv"
)


@tool
def analyze_transactions() -> dict:
    """Calculate payment transaction statistics from
    the authorized demo transaction dataset."""

    with DATA_FILE.open(
        newline="",
        encoding="utf-8",
    ) as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return {
            "total_transactions": 0,
            "successful_transactions": 0,
            "failed_transactions": 0,
            "success_rate_pct": 0.0,
            "total_successful_amount": 0.0,
            "average_latency_ms": 0.0,
        }

    total = len(rows)

    successful = [
        row for row in rows
        if row["status"] == "SUCCESS"
    ]

    failed = [
        row for row in rows
        if row["status"] == "FAILED"
    ]

    success_rate = (
        len(successful) / total
    ) * 100

    total_amount = sum(
        float(row["amount"])
        for row in successful
    )

    avg_latency = sum(
        float(row["latency_ms"])
        for row in rows
    ) / total

    return {
        "total_transactions": total,
        "successful_transactions": len(successful),
        "failed_transactions": len(failed),
        "success_rate_pct": round(success_rate, 2),
        "total_successful_amount": round(
            total_amount, 2
        ),
        "average_latency_ms": round(
            avg_latency, 2
        ),
    }
