"""Artifact verifier for D20 blocked-gap augmented ranker."""

from __future__ import annotations

import json
import math
from pathlib import Path

from transsolvestack.profiling.artifacts import read_jsonl
from transsolvestack.profiling.provenance import read_manifest, verify_manifest_hashes


def verify_csr_blocked_gap_augmented_ranker(path: Path) -> None:
    manifest = read_manifest(path / "artifact_manifest.json")
    stale = verify_manifest_hashes(manifest)
    if stale:
        raise SystemExit(f"stale CSR blocked gap augmented ranker manifest: {stale}")
    model = json.loads(
        (path / "csr_blocked_gap_augmented_ranker_model.json").read_text(
            encoding="utf-8"
        )
    )
    predictions = read_jsonl(path / "csr_blocked_gap_augmented_ranker_predictions.jsonl")
    positive_predictions = read_jsonl(
        path / "csr_blocked_gap_augmented_ranker_positive_predictions.jsonl"
    )
    summary = json.loads(
        (path / "csr_blocked_gap_augmented_ranker_summary.json").read_text(
            encoding="utf-8"
        )
    )
    schema = json.loads(
        (path / "csr_blocked_gap_augmented_ranker_schema.json").read_text(
            encoding="utf-8"
        )
    )
    if manifest.artifact_kind != "csr_blocked_gap_augmented_ranker":
        raise SystemExit("unexpected CSR blocked gap augmented ranker artifact kind")
    if summary["status"] != "passed":
        raise SystemExit("CSR blocked gap augmented ranker did not pass")
    if summary["schema_version"] != "phase1_csr_blocked_gap_augmented_ranker_v1":
        raise SystemExit("CSR blocked gap augmented ranker schema mismatch")
    if schema["schema_version"] != summary["schema_version"]:
        raise SystemExit("CSR blocked gap augmented ranker schema/summary mismatch")
    boundary = schema["runtime_boundary"]
    if (
        boundary["runtime_selector_changed"] is not False
        or boundary["shadow_only"] is not True
    ):
        raise SystemExit("CSR blocked gap augmented ranker runtime boundary mismatch")
    if (
        summary["runtime_selector_changed"] is not False
        or summary["shadow_only"] is not True
        or summary["model_trained"] is not True
    ):
        raise SystemExit("CSR blocked gap augmented ranker changed runtime boundary")
    if model["runtime_integration"]["runtime_selector_changed"] is not False:
        raise SystemExit("CSR blocked gap augmented ranker model changed runtime selector")
    if len(predictions) != summary["ranker_num_predictions"]:
        raise SystemExit("CSR blocked gap augmented ranker prediction count mismatch")
    if summary["ranker_num_predictions"] != 105 or summary["ranker_num_requests"] != 105:
        raise SystemExit("CSR blocked gap augmented ranker request count mismatch")
    if summary["ranker_num_global_candidates"] != 12:
        raise SystemExit("CSR blocked gap augmented ranker candidate count mismatch")
    if summary["ranker_token_feature_dim"] != 40:
        raise SystemExit("CSR blocked gap augmented ranker token feature mismatch")
    if summary["ranker_scorer_feature_dim"] != len(model["scorer_head"]):
        raise SystemExit("CSR blocked gap augmented ranker scorer shape mismatch")
    if not math.isfinite(float(summary["ranker_final_train_pairwise_loss"])):
        raise SystemExit("CSR blocked gap augmented ranker loss is not finite")
    if int(summary["ranker_num_pairwise_constraints"]) <= 0:
        raise SystemExit("CSR blocked gap augmented ranker has no pairwise constraints")
    if int(summary["ranker_num_pairwise_updates"]) <= 0:
        raise SystemExit("CSR blocked gap augmented ranker has no pairwise updates")
    if len(positive_predictions) != summary["blocked_gap_prediction_rows"]:
        raise SystemExit("CSR blocked gap augmented ranker positive row count mismatch")
    if summary["blocked_gap_prediction_rows"] != 5:
        raise SystemExit("CSR blocked gap augmented ranker positive request mismatch")
    if summary["blocked_gap_train_predictions"] != 4:
        raise SystemExit("CSR blocked gap augmented ranker train split mismatch")
    if summary["blocked_gap_eval_predictions"] != 1:
        raise SystemExit("CSR blocked gap augmented ranker eval split mismatch")
    if summary["blocked_gap_success_predictions"] != 5:
        raise SystemExit("CSR blocked gap augmented ranker missed positive successes")
    if summary["blocked_gap_non_success_predictions"] != 0:
        raise SystemExit("CSR blocked gap augmented ranker selected non-success positive")
    if summary["blocked_gap_oracle_matches"] < 4:
        raise SystemExit("CSR blocked gap augmented ranker oracle match floor missed")
    if summary["blocked_gap_eval_success_selection_rate"] != 1.0:
        raise SystemExit("CSR blocked gap augmented ranker eval positive success mismatch")
    if summary["blocked_gap_positive_candidate_coverage_complete"] is not True:
        raise SystemExit("CSR blocked gap augmented ranker candidate coverage incomplete")
    if {row["selected_target_status"] for row in positive_predictions} != {"success"}:
        raise SystemExit("CSR blocked gap augmented ranker positive rows not success")
    if any(not row["selected_is_profiled_success"] for row in positive_predictions):
        raise SystemExit("CSR blocked gap augmented ranker positive success flag mismatch")
    if any(not row["selected_is_positive_candidate"] for row in positive_predictions):
        raise SystemExit("CSR blocked gap augmented ranker selected outside positives")
    if set(summary["blocked_gap_ranked_new_candidate_ids"]) != {
        "taichi_csr_bicgstab_ilu0_float64",
        "taichi_csr_bicgstab_row_column_equilibration_float64",
        "taichi_csr_pcg_symmetric_equilibration_float64",
    }:
        raise SystemExit("CSR blocked gap augmented ranker new candidate coverage mismatch")
    if (
        manifest.metadata["runtime_selector_changed"] is not False
        or manifest.metadata["shadow_only"] is not True
    ):
        raise SystemExit("CSR blocked gap augmented ranker manifest boundary mismatch")
