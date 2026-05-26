"""Run real Taichi numerical regression checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.benchmarks.candidates import load_candidate_set
from transsolvestack.benchmarks.config import load_workload_config
from transsolvestack.profiling.numerical_regression import (
    TaichiNumericalRegressionRunner,
)
from transsolvestack.runtime.config import load_resource_budget


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="configs/workloads/phase1_smoke.yaml")
    parser.add_argument("--out", default="runs/phase1_numerical_regression")
    parser.add_argument("--device-memory-gb", type=float, default=0.5)
    args = parser.parse_args()

    workload = load_workload_config(args.workload)
    candidate_set = load_candidate_set(workload.candidate_set_path)
    if workload.resource_limits_path is None:
        raise SystemExit("workload does not define resource_limits")
    budget = load_resource_budget(workload.resource_limits_path)
    paths = TaichiNumericalRegressionRunner(
        device_memory_gb=args.device_memory_gb
    ).run(workload, candidate_set, budget, args.out)
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

