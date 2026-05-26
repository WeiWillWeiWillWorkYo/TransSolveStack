import json

import pytest

import transsolvestack as tss
from transsolvestack.datasets.csr import CsrMatrix
from transsolvestack.profiling.artifacts import read_jsonl, write_jsonl


def test_csr_learned_guard_shadows_current_blocked_ranker():
    rows = {
        str(row["matrix_id"]): row
        for row in read_jsonl("runs/phase1_suitesparse_csr_import/csr_matrices.jsonl")
    }

    decision = tss.plan_csr_with_learned_guard(
        rows["suitesparse:HB/curtis54"],
        mode="promote_if_safe",
    )

    assert decision.guard_status == "blocked_quality_gate"
    assert decision.runtime_selection_source == "artifact"
    assert decision.runtime_selector_changed is False
    assert decision.runtime_candidate_id == decision.artifact_candidate_id
    assert decision.fallback_chain_enforced is True
    assert "quality_gate:non_success_eval_selections" in decision.guard_reasons
    assert decision.learned_prediction is not None


def test_csr_learned_guard_promotes_only_profiled_success_candidate(tmp_path):
    selector_path = tmp_path / "selector.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    gate_path = tmp_path / "gate.json"
    write_jsonl(_selector_rows(), selector_path)
    write_jsonl((_prediction("learned_safe"),), predictions_path)
    gate_path.write_text(json.dumps(_eligible_gate()) + "\n", encoding="utf-8")

    decision = tss.plan_csr_with_learned_guard(
        _fixture_csr(),
        selector_path=selector_path,
        learned_predictions_path=predictions_path,
        quality_gate_summary_path=gate_path,
        context=tss.SolveContext(context_id="fixture_guard"),
        mode="promote_if_safe",
        min_confidence=0.75,
    )

    assert decision.guard_status == "promoted"
    assert decision.runtime_selection_source == "learned"
    assert decision.runtime_selector_changed is True
    assert decision.runtime_candidate_id == "learned_safe"
    assert decision.artifact_candidate_id == "artifact_fast"
    assert decision.fallback_candidate_ids == ("artifact_fast",)
    assert decision.fallback_chain_enforced is True


def test_csr_learned_guard_rejects_screened_out_candidate(tmp_path):
    selector_path = tmp_path / "selector.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    gate_path = tmp_path / "gate.json"
    write_jsonl(_selector_rows(), selector_path)
    write_jsonl((_prediction("screened_bad"),), predictions_path)
    gate_path.write_text(json.dumps(_eligible_gate()) + "\n", encoding="utf-8")

    decision = tss.plan_csr_with_learned_guard(
        _fixture_csr(),
        selector_path=selector_path,
        learned_predictions_path=predictions_path,
        quality_gate_summary_path=gate_path,
        context=tss.SolveContext(context_id="fixture_guard"),
        mode="promote_if_safe",
        min_confidence=0.75,
    )

    assert decision.guard_status == "blocked_non_success_candidate"
    assert decision.runtime_selection_source == "artifact"
    assert decision.runtime_selector_changed is False
    assert decision.runtime_candidate_id == "artifact_fast"


def test_csr_artifact_selector_rejects_direct_non_success_candidate(tmp_path):
    from transsolvestack.policies.csr_artifact_selector import CsrArtifactPolicySelector

    selector_path = tmp_path / "selector.jsonl"
    write_jsonl(_selector_rows(), selector_path)
    selector = CsrArtifactPolicySelector.from_selector_rows(selector_path)

    with pytest.raises(ValueError):
        selector.select_candidate(
            _fixture_csr(),
            "screened_bad",
            context_id="fixture_guard",
        )


def _fixture_csr() -> CsrMatrix:
    return CsrMatrix(
        matrix_id="fixture:learned_guard",
        n_rows=2,
        n_cols=2,
        row_ptr=(0, 1, 2),
        col_ind=(0, 1),
        values=(1.0, 1.0),
        field="real",
        symmetry="symmetric",
        source_path="memory://learned_guard",
    )


def _selector_rows() -> tuple[dict, ...]:
    base = {
        "matrix_id": "fixture:learned_guard",
        "context_id": "fixture_guard",
        "precision": "float64",
        "target_success_rate": 1.0,
        "target_measurement_repeats": 1,
        "target_solve_time_iqr_ms": 0.0,
        "target_final_relative_residual": 1.0e-8,
        "target_cpu_recomputed_relative_residual": 1.0e-8,
        "target_solution_relative_error": 0.0,
        "features": {},
    }
    return (
        {
            **base,
            "candidate_id": "artifact_fast",
            "solver": "cg",
            "preconditioner": "none",
            "solver_parameters": {},
            "target_status": "success",
            "label_is_oracle": True,
            "target_solve_time_ms": 1.0,
            "target_median_solve_time_ms": 1.0,
        },
        {
            **base,
            "candidate_id": "learned_safe",
            "solver": "pcg",
            "preconditioner": "jacobi",
            "solver_parameters": {},
            "target_status": "success",
            "label_is_oracle": False,
            "target_solve_time_ms": 2.0,
            "target_median_solve_time_ms": 2.0,
        },
        {
            **base,
            "candidate_id": "screened_bad",
            "solver": "gmres",
            "preconditioner": "jacobi",
            "solver_parameters": {"restart": 8},
            "target_status": "screened_out",
            "target_success_rate": 0.0,
            "label_is_oracle": False,
            "target_solve_time_ms": None,
            "target_median_solve_time_ms": None,
        },
    )


def _prediction(candidate_id: str) -> dict:
    scores = {
        "artifact_fast": 4.0,
        "learned_safe": 8.0 if candidate_id == "learned_safe" else 2.0,
        "screened_bad": 8.0 if candidate_id == "screened_bad" else -20.0,
    }
    ranked = sorted(scores, key=lambda item: (-scores[item], item))
    return {
        "model_id": "fixture_ranker",
        "request_id": "fixture:learned_guard:fixture_guard",
        "matrix_id": "fixture:learned_guard",
        "context_id": "fixture_guard",
        "selected_candidate_id": candidate_id,
        "selected_target_status": "success" if candidate_id != "screened_bad" else "screened_out",
        "evaluation_status": "profiled_success_non_oracle",
        "ranked_candidate_ids": ranked,
        "scores": scores,
    }


def _eligible_gate() -> dict:
    return {
        "status": "passed",
        "challenger_model_id": "fixture_ranker",
        "runtime_selected_model_id": "fixture_ranker",
        "runtime_selector_changed": False,
        "challenger_runtime_eligible": True,
        "challenger_gate_failures": [],
    }
