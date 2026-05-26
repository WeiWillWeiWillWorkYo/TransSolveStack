"""Select PolicyPlan records from benchmark evaluation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.benchmarks.candidates import load_candidate_set
from transsolvestack.policies.artifact_selector import BenchmarkArtifactPolicySelector
from transsolvestack.policies.selection_artifacts import (
    build_selected_policy_records,
    write_policy_selection_report,
    write_selected_policy_records,
)
from transsolvestack.profiling.provenance import (
    CORE_POLICY_SELECTION_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-set",
        default="configs/candidates/phase1_taichi_gpu.yaml",
    )
    parser.add_argument(
        "--evaluation",
        default="runs/phase1_smoke_taichi/candidate_evaluation.jsonl",
    )
    parser.add_argument("--out", default="runs/phase1_policy_selection")
    parser.add_argument("--fallback-candidate-id", default=None)
    args = parser.parse_args()

    candidate_set = load_candidate_set(args.candidate_set)
    selector = BenchmarkArtifactPolicySelector.from_evaluation_artifact(
        candidate_set=candidate_set,
        evaluation_path=args.evaluation,
        fallback_candidate_id=args.fallback_candidate_id,
    )
    records = build_selected_policy_records(selector, args.evaluation)
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected_plans": output / "selected_policy_plans.jsonl",
        "report": output / "policy_selection_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_selected_policy_records(records, paths["selected_plans"])
    write_policy_selection_report(records, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="policy_selection",
            command="scripts/tss_select_policies_from_artifact.py",
            tracked_files=CORE_POLICY_SELECTION_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "num_selected_plans": len(records),
                "candidate_set": args.candidate_set,
                "evaluation": args.evaluation,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
