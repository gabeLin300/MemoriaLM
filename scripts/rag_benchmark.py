import argparse
import json
import time
from statistics import mean
from typing import Any

import requests


def _request(method: str, url: str, *, headers=None, params=None, json_body=None, timeout=180) -> dict[str, Any]:
    resp = requests.request(
        method=method,
        url=url,
        headers=headers,
        params=params,
        json=json_body,
        timeout=timeout,
    )
    resp.raise_for_status()
    if resp.content:
        return resp.json()
    return {}


def run_benchmark(
    *,
    base_url: str,
    user_id: str,
    notebook_id: str,
    query: str,
    top_k: int,
    runs: int,
) -> dict[str, Any]:
    headers = {"X-User-Id": user_id}
    modes = ["topk", "rerank"]
    results: dict[str, Any] = {}

    for mode in modes:
        latencies_ms: list[float] = []
        citations_count: list[int] = []
        used_chunks: list[int] = []

        for _ in range(runs):
            start = time.perf_counter()
            payload = _request(
                "POST",
                f"{base_url}/api/notebooks/{notebook_id}/chat",
                headers=headers,
                json_body={
                    "user_id": user_id,
                    "message": query,
                    "top_k": top_k,
                    "retrieval_mode": mode,
                },
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies_ms.append(elapsed_ms)
            citations_count.append(len(payload.get("citations", [])))
            used_chunks.append(int(payload.get("used_chunks", 0)))

        results[mode] = {
            "avg_latency_ms": round(mean(latencies_ms), 2),
            "min_latency_ms": round(min(latencies_ms), 2),
            "max_latency_ms": round(max(latencies_ms), 2),
            "avg_citations": round(mean(citations_count), 2),
            "avg_used_chunks": round(mean(used_chunks), 2),
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark topk vs rerank retrieval modes.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--notebook-id", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    results = run_benchmark(
        base_url=args.base_url.rstrip("/"),
        user_id=args.user_id,
        notebook_id=args.notebook_id,
        query=args.query,
        top_k=args.top_k,
        runs=max(1, args.runs),
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
