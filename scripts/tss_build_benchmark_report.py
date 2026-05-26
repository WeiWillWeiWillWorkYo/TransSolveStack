"""Build per-system benchmark evaluation report from profile artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.benchmarks.evaluation import (
    build_candidate_evaluations,
    write_benchmark_report,
    write_candidate_evaluations,
)
from transsolvestack.benchmarks.oracle import build_oracles_for_contexts
from transsolvestack.profiling.artifacts import read_candidate_performance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", default="runs/phase1_smoke_taichi")
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    records = read_candidate_performance(profile_dir / "candidate_performance.jsonl")
    oracles = build_oracles_for_contexts(records)
    evaluations = build_candidate_evaluations(records, oracles)
    table_path = write_candidate_evaluations(
        evaluations,
        profile_dir / "candidate_evaluation.jsonl",
    )
    report_path = write_benchmark_report(
        evaluations,
        profile_dir / "benchmark_report.md",
    )
    print(f"evaluations: {len(evaluations)}")
    print(f"table: {table_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()

