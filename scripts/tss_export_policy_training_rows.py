"""Export policy training rows and Transformer feature schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.benchmarks.config import load_workload_config
from transsolvestack.policies.training_data import (
    build_policy_training_rows,
    write_policy_training_report,
    write_policy_training_rows,
    write_transformer_feature_schema,
)
from transsolvestack.profiling.provenance import (
    CORE_TRANSFORMER_READINESS_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", default="configs/workloads/phase1_smoke.yaml")
    parser.add_argument(
        "--evaluation",
        default="runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_transformer_readiness")
    args = parser.parse_args()

    workload = load_workload_config(args.workload)
    rows = build_policy_training_rows(workload, args.evaluation)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "rows": output / "policy_training_rows.jsonl",
        "schema": output / "transformer_feature_schema.json",
        "report": output / "transformer_readiness_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_policy_training_rows(rows, paths["rows"])
    write_transformer_feature_schema(paths["schema"])
    write_policy_training_report(rows, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="transformer_readiness_export",
            command="scripts/tss_export_policy_training_rows.py",
            tracked_files=CORE_TRANSFORMER_READINESS_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "num_rows": len(rows),
                "num_oracle_rows": sum(1 for row in rows if row.label_is_oracle),
                "workload": args.workload,
                "evaluation": args.evaluation,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
