"""Build dry-run profile artifacts without running GPU kernels."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.benchmarks.candidates import load_candidate_set
from transsolvestack.benchmarks.config import load_workload_config
from transsolvestack.profiling.runner import ProfileRunner
from transsolvestack.runtime.config import load_resource_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        default="configs/workloads/phase1_smoke.yaml",
    )
    parser.add_argument(
        "--out",
        default="runs/phase1_smoke_dry_run",
    )
    args = parser.parse_args()

    workload = load_workload_config(args.workload)
    candidate_set = load_candidate_set(workload.candidate_set_path)
    if workload.resource_limits_path is None:
        raise SystemExit("workload does not define resource_limits")
    budget = load_resource_budget(workload.resource_limits_path)
    runner = ProfileRunner()
    result = runner.dry_run(workload, candidate_set, budget)
    paths = runner.write_dry_run_artifacts(result, args.out)
    print(f"run_plans: {paths['run_plans']}")
    print(f"run_traces: {paths['run_traces']}")
    print(f"candidate_performance: {paths['candidate_performance']}")
    print(f"report: {paths['report']}")


if __name__ == "__main__":
    main()

