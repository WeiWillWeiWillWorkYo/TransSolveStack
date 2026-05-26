"""Plan guarded learned-policy promotion coverage expansion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transsolvestack.profiling.csr_guarded_promotion_coverage import (
    build_csr_guarded_promotion_coverage_plan_from_files,
    write_csr_guarded_promotion_coverage_report,
    write_csr_guarded_promotion_coverage_rows,
    write_csr_guarded_promotion_coverage_schema,
    write_csr_guarded_promotion_coverage_summary,
)
from transsolvestack.profiling.provenance import (
    CORE_CSR_GUARDED_PROMOTION_COVERAGE_PROVENANCE_FILES,
    build_artifact_manifest,
    git_commit_or_unknown,
    write_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selector-rows",
        default="runs/phase1_csr_transformer_ready/combined_csr_selector_rows.jsonl",
    )
    parser.add_argument(
        "--learned-predictions",
        default="runs/phase1_csr_transformer_ranker/csr_transformer_ranker_predictions.jsonl",
    )
    parser.add_argument(
        "--quality-gate-summary",
        default="runs/phase1_csr_transformer_quality_gate/csr_transformer_quality_gate_summary.json",
    )
    parser.add_argument(
        "--csr",
        action="append",
        default=[
            "runs/phase1_suitesparse_csr_import/csr_matrices.jsonl",
            "runs/phase1_csr_micro_campaign/csr_matrices.jsonl",
        ],
    )
    parser.add_argument("--max-current-blocked", type=int, default=4)
    parser.add_argument("--max-fixture-promotions", type=int, default=4)
    parser.add_argument("--max-non-success-blocks", type=int, default=4)
    parser.add_argument("--max-runtime-fallbacks", type=int, default=3)
    parser.add_argument("--out", default="runs/phase1_csr_guarded_promotion_coverage_plan")
    args = parser.parse_args()

    plan = build_csr_guarded_promotion_coverage_plan_from_files(
        selector_rows_path=args.selector_rows,
        predictions_path=args.learned_predictions,
        quality_gate_summary_path=args.quality_gate_summary,
        csr_paths=tuple(args.csr),
        max_current_blocked=args.max_current_blocked,
        max_fixture_promotions=args.max_fixture_promotions,
        max_non_success_blocks=args.max_non_success_blocks,
        max_runtime_fallbacks=args.max_runtime_fallbacks,
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "scenario_queue": output / "csr_guarded_promotion_coverage_scenarios.jsonl",
        "summary": output / "csr_guarded_promotion_coverage_summary.json",
        "schema": output / "csr_guarded_promotion_coverage_schema.json",
        "report": output / "csr_guarded_promotion_coverage_report.md",
        "manifest": output / "artifact_manifest.json",
    }
    write_csr_guarded_promotion_coverage_rows(plan.rows, paths["scenario_queue"])
    write_csr_guarded_promotion_coverage_summary(plan.summary, paths["summary"])
    write_csr_guarded_promotion_coverage_schema(plan.schema, paths["schema"])
    write_csr_guarded_promotion_coverage_report(plan, paths["report"])
    write_manifest(
        build_artifact_manifest(
            artifact_kind="csr_guarded_promotion_coverage_plan",
            command="scripts/tss_csr_guarded_promotion_coverage_plan.py",
            tracked_files=CORE_CSR_GUARDED_PROMOTION_COVERAGE_PROVENANCE_FILES,
            metadata={
                "git_commit": git_commit_or_unknown(),
                "status": plan.summary.status,
                "schema_version": plan.summary.schema_version,
                "plan_id": plan.summary.plan_id,
                "planned_scenarios": plan.summary.planned_scenarios,
                "planned_gpu_final_solves": plan.summary.planned_gpu_final_solves,
                "runtime_selector_changed": plan.summary.runtime_selector_changed,
                "executes_gpu": plan.summary.executes_gpu,
            },
        ),
        paths["manifest"],
    )
    for key, value in paths.items():
        print(f"{key}: {value}")
    print(f"status: {plan.summary.status}")
    if plan.summary.status != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
