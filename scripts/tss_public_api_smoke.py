"""Exercise the public Python API on a real Taichi GPU smoke solve."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import transsolvestack as tss
from transsolvestack.core.types import SolveContext
from transsolvestack.operators.synthetic import build_synthetic_system
from transsolvestack.profiling.provenance import (
    CORE_PUBLIC_API_SMOKE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)
from transsolvestack.profiling.trace import trace_to_record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/phase1_public_api_smoke")
    parser.add_argument("--device-memory-gb", type=float, default=0.5)
    args = parser.parse_args()

    system = build_synthetic_system(
        "poisson_2d_stencil",
        (32, 32),
        dtype="float32",
        expected_operator_kind="structured_stencil",
    ).system
    context = SolveContext(context_id="default_f32", max_iter=300)
    plan = tss.plan(
        system,
        context,
        policy_mode="benchmark_artifact",
        candidate_set="configs/candidates/phase1_taichi_gpu.yaml",
        evaluation_path="runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    )
    result = tss.solve(
        system=system,
        rhs=None,
        context=context,
        policy_plan=plan,
        device_memory_gb=args.device_memory_gb,
    )
    training_rows = tss.export_policy_training_rows(
        "configs/workloads/phase1_smoke.yaml",
        "runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    )
    download_plan = tss.plan_dataset_downloads("configs/datasets/phase1_external_small.yaml")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "result": output / "public_api_smoke.json",
        "manifest": output / "artifact_manifest.json",
    }
    payload = {
        "plan": asdict(plan),
        "trace": trace_to_record(result.trace),
        "status": result.status,
        "num_training_rows": len(training_rows),
        "num_dataset_plan_entries": len(download_plan.entries),
    }
    paths["result"].write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    write_manifest(
        build_artifact_manifest(
            artifact_kind="public_api_smoke",
            command="scripts/tss_public_api_smoke.py",
            tracked_files=CORE_PUBLIC_API_SMOKE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": result.status,
                "num_training_rows": len(training_rows),
                "num_dataset_plan_entries": len(download_plan.entries),
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
