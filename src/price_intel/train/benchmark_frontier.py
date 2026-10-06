"""
benchmark_frontier.py

Compare the FrontierAgent's LLM providers on the same held-out test items.

- Similar products are retrieved from Chroma once per item, so every provider gets identical context.
- Each provider is run several times (--runs), since LLM outputs vary between runs.
- Reports, per provider (mean ± std across runs): average error, % of estimates within 20% of
  the true price, failure rate, cost per 1,000 estimates, and average LLM latency.
- Writes per-call predictions (CSV) and a summary (JSON) to artifacts/benchmarks/<timestamp>/.

Usage:
    python src/price_intel/train/benchmark_frontier.py --runs 3
    python src/price_intel/train/benchmark_frontier.py --providers claude --limit 10 --runs 1
"""

import argparse
import csv
import json
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
import chromadb
import openai
from tqdm import tqdm

from price_intel.agents.frontier_agent import FrontierAgent
from price_intel.agents.frontier_providers import PriceResult
from price_intel.data.eval_data import (
    ENSEMBLE_FIT_END,
    ENSEMBLE_FIT_START,
    EvalItem,
    load_eval_items,
)
from price_intel.train.testing import summarize


DB_PATH = "products_vectorstore"
COLLECTION_NAME = "products"

# USD per 1M tokens (input, output), standard tier - checked 2026-10-06.
# Models missing here (e.g. deepseek-chat) get no cost figure.
PRICES_PER_MTOK = {
    "gpt-4o-mini": (0.15, 0.60),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Errors that won't go away on the next item (bad key, bad model name, invalid request):
# stop the benchmark rather than count every item as a failure.
FATAL_API_ERRORS = (
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.BadRequestError,
    anthropic.AuthenticationError,
    anthropic.PermissionDeniedError,
    anthropic.NotFoundError,
    anthropic.BadRequestError,
)
# Anything else from the APIs (timeouts, rate limits after the SDK's retries, 5xx) is recorded
# as a failure for that item.
API_ERRORS = (openai.APIError, anthropic.APIError)

METRICS = ["average_error", "within_20_rate", "failure_rate", "cost_per_1000", "avg_latency_s"]


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark frontier LLM providers.")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["openai", "claude"],
        choices=["openai", "deepseek", "claude"],
    )
    parser.add_argument("--runs", type=int, default=3, help="Runs per provider")
    parser.add_argument("--start", type=int, default=0, help="First test item")
    parser.add_argument("--limit", type=int, default=250, help="Number of test items")
    parser.add_argument("--k", type=int, default=5, help="Similar products in the prompt")
    parser.add_argument("--workers", type=int, default=1, help="Parallel LLM calls per provider")
    parser.add_argument("--out", default="artifacts/benchmarks", help="Output directory")
    return parser.parse_args()


def call_cost(model: str, result: PriceResult) -> Optional[float]:
    if model not in PRICES_PER_MTOK:
        return None
    input_price, output_price = PRICES_PER_MTOK[model]
    return (result.input_tokens * input_price + result.output_tokens * output_price) / 1_000_000


def estimate_one(agent: FrontierAgent, item: EvalItem, context) -> PriceResult:
    documents, prices = context
    start = time.perf_counter()
    try:
        return agent.estimate(item.description, documents, prices)
    except FATAL_API_ERRORS:
        raise
    except API_ERRORS as e:
        return PriceResult(
            price=None,
            latency_s=time.perf_counter() - start,
            failure=f"api error: {type(e).__name__}",
        )


def run_provider(agent: FrontierAgent, items: List[EvalItem], contexts, workers: int, desc: str):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(
            tqdm(
                pool.map(lambda pair: estimate_one(agent, *pair), zip(items, contexts)),
                total=len(items),
                desc=desc,
            )
        )


def summarize_run(results: List[PriceResult], truths: List[float], model: str) -> Dict:
    summary = summarize([r.price for r in results], truths)
    costs = [call_cost(model, r) for r in results]
    summary["cost_per_1000"] = None if None in costs else 1000 * sum(costs) / len(costs)
    summary["avg_latency_s"] = statistics.mean(r.latency_s for r in results)
    summary["avg_input_tokens"] = statistics.mean(r.input_tokens for r in results)
    summary["avg_output_tokens"] = statistics.mean(r.output_tokens for r in results)
    summary["failure_reasons"] = dict(Counter(r.failure for r in results if r.failure))
    return summary


def aggregate(runs: List[Dict]) -> Dict:
    """
    Mean and standard deviation of each metric across runs.
    """
    result = {}
    for metric in METRICS:
        values = [run[metric] for run in runs if run[metric] is not None]
        if not values:
            result[metric] = {"mean": None, "std": None}
            continue
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        result[metric] = {"mean": statistics.mean(values), "std": std}
    return result


def fmt(stat: Dict, kind: str) -> str:
    mean, std = stat["mean"], stat["std"]
    if mean is None:
        return "n/a"
    if kind == "dollars":
        return f"${mean:,.2f} ± {std:,.2f}"
    if kind == "percent":
        return f"{mean * 100:.1f}% ± {std * 100:.1f}"
    if kind == "cost":
        return f"${mean:,.3f}"
    return f"{mean:.2f}s ± {std:.2f}"


def print_table(summary: Dict, runs: int, n_items: int):
    print(f"\nFrontier benchmark: {n_items} held-out items, {runs} run(s) per provider (mean ± std)\n")
    header = f"{'Provider (model)':<30}{'Avg error':>18}{'Within 20%':>18}{'Failures':>18}{'$ / 1k est.':>14}{'Latency':>16}"
    print(header)
    print("-" * len(header))
    for name, data in summary.items():
        agg = data["aggregate"]
        print(
            f"{name:<30}"
            f"{fmt(agg['average_error'], 'dollars'):>18}"
            f"{fmt(agg['within_20_rate'], 'percent'):>18}"
            f"{fmt(agg['failure_rate'], 'percent'):>18}"
            f"{fmt(agg['cost_per_1000'], 'cost'):>14}"
            f"{fmt(agg['avg_latency_s'], 'seconds'):>16}"
        )
    print("\nError metrics exclude failed estimates - read them together with the failure rate.")


def main():
    args = parse_args()

    end = args.start + args.limit
    if args.start < ENSEMBLE_FIT_END and end > ENSEMBLE_FIT_START:
        print(
            f"Warning: items [{args.start}:{end}] overlap the ensemble's fitting slice "
            f"[{ENSEMBLE_FIT_START}:{ENSEMBLE_FIT_END}]"
        )

    items = load_eval_items(args.start, end)
    truths = [item.price for item in items]

    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection(COLLECTION_NAME)

    # Create every agent first, so a missing API key fails before any money is spent
    agents = {provider: FrontierAgent(collection, provider=provider) for provider in args.providers}

    # Retrieve similar products once; every provider and run uses the same context
    retriever = next(iter(agents.values()))
    contexts = [
        retriever.find_similars(item.description, k=args.k)
        for item in tqdm(items, desc="Retrieving similar products")
    ]

    out_dir = Path(args.out) / datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = {}

    for provider, agent in agents.items():
        name = f"{provider} ({agent.MODEL})"
        run_summaries = []
        for run in range(1, args.runs + 1):
            results = run_provider(agent, items, contexts, args.workers, f"{name} run {run}/{args.runs}")
            run_summaries.append(summarize_run(results, truths, agent.MODEL))
            for i, (item, result) in enumerate(zip(items, results)):
                rows.append({
                    "run": run,
                    "provider": provider,
                    "model": agent.MODEL,
                    "item_index": args.start + i,
                    "title": item.title[:80],
                    "truth": item.price,
                    "price": result.price,
                    "failure": result.failure or "",
                    "latency_s": round(result.latency_s, 4),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "raw": result.raw,
                })
        summary[name] = {"runs": run_summaries, "aggregate": aggregate(run_summaries)}

    with open(out_dir / "predictions.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {"config": vars(args), "prices_per_mtok": PRICES_PER_MTOK, "results": summary},
            f,
            indent=2,
        )

    print_table(summary, args.runs, len(items))
    print(f"\nSaved predictions.csv and summary.json to {out_dir}")


if __name__ == "__main__":
    main()
