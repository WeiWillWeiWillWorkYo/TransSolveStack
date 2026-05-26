"""Summarize profile artifacts and build oracle records."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.benchmarks.oracle import (
    build_oracles_for_contexts,
    write_oracle_plans,
)
from transsolvestack.benchmarks.summary import (
    summarize_candidate_performance,
    write_performance_summary_report,
)
from transsolvestack.profiling.artifacts import read_candidate_performance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile-dir",
        default="runs/phase1_smoke_dry_run",
    )
    args = parser.parse_args()

    profile_dir = Path(args.profile_dir)
    performance_path = profile_dir / "candidate_performance.jsonl"
    records = read_candidate_performance(performance_path)
    oracles = build_oracles_for_contexts(records)
    summary = summarize_candidate_performance(records)
    oracle_path = write_oracle_plans(oracles, profile_dir / "oracle_plan.jsonl")
    report_path = write_performance_summary_report(
        summary,
        oracles,
        profile_dir / "performance_summary.md",
    )
    print(f"records: {summary.num_records}")
    print(f"oracle_plans: {len(oracles)}")
    print(f"oracle_path: {oracle_path}")
    print(f"report_path: {report_path}")


if __name__ == "__main__":
    main()

