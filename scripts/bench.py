"""Latency benchmark against a running service.

    uvicorn remark.api:app --port 8000 &
    python scripts/bench.py --n 1000 [--url http://127.0.0.1:8000]

Sends N sequential single-item requests (keep-alive session) built from
test.csv narrations, prints p50/p95/p99 and writes reports/bench_<model>.json
so remark.compare can pick up the p95.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import requests


def percentile(sorted_values: list[float], q: float) -> float:
    index = min(len(sorted_values) - 1, max(0, round(q / 100 * len(sorted_values)) - 1))
    return sorted_values[index]


def main() -> None:
    parser = argparse.ArgumentParser(description="p50/p95 latency of POST /v1/score")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--data", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    with args.data.open(encoding="utf-8") as f:
        rows = [{"narration": r["narration"], "channel": r["channel"], "amount": float(r["amount"])}
                for r in csv.DictReader(f)]
    session = requests.Session()
    health = session.get(f"{args.url}/health", timeout=5).json()
    for i in range(args.warmup):
        session.post(f"{args.url}/v1/score", json=rows[i % len(rows)], timeout=5).raise_for_status()

    latencies_ms = []
    for i in range(args.n):
        start = time.perf_counter()
        response = session.post(f"{args.url}/v1/score", json=rows[i % len(rows)], timeout=5)
        latencies_ms.append((time.perf_counter() - start) * 1000)
        response.raise_for_status()

    latencies_ms.sort()
    result = {
        "model": health["model"], "model_version": health["model_version"], "n": args.n,
        "p50_ms": round(percentile(latencies_ms, 50), 2),
        "p95_ms": round(percentile(latencies_ms, 95), 2),
        "p99_ms": round(percentile(latencies_ms, 99), 2),
        "mean_ms": round(statistics.fmean(latencies_ms), 2),
    }
    print(f"model={result['model']} ({result['model_version']}) n={args.n} "
          f"p50={result['p50_ms']}ms p95={result['p95_ms']}ms p99={result['p99_ms']}ms")
    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / f"bench_{health['model']}.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
