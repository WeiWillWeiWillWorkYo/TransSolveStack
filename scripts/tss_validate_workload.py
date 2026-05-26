"""Dry-run validate a workload without running GPU kernels."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.benchmarks.candidates import load_candidate_set
from transsolvestack.benchmarks.config import load_workload_config
from transsolvestack.benchmarks.report import write_dry_run_report
from transsolvestack.benchmarks.validation import validate_workload_for_budget
from transsolvestack.runtime.config import load_resource_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "workload",
        nargs="?",
        default="configs/workloads/phase1_smoke.yaml",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    workload = load_workload_config(args.workload)
    candidate_set = load_candidate_set(workload.candidate_set_path)
    if workload.resource_limits_path is None:
        raise SystemExit("workload does not define resource_limits")
    budget = load_resource_budget(workload.resource_limits_path)
    report = validate_workload_for_budget(workload, candidate_set, budget)
    if args.report:
        write_dry_run_report(report, args.report)
    for key, value in asdict(report).items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
