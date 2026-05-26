import pytest

from transsolvestack.policies.csr_artifact_selector import CsrArtifactPolicySelector
from transsolvestack.profiling.artifacts import read_jsonl


def test_csr_artifact_selector_selects_oracle_and_fallbacks():
    rows = {
        str(row["matrix_id"]): row
        for row in read_jsonl("runs/phase1_suitesparse_csr_import/csr_matrices.jsonl")
    }
    selector = CsrArtifactPolicySelector.from_selector_rows(
        "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl"
    )
    selector_rows = read_jsonl("runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl")

    trefethen = selector.select(rows["suitesparse:JGD_Trefethen/Trefethen_20b"])
    curtis = selector.select(rows["suitesparse:HB/curtis54"])
    trefethen_oracle = next(
        row
        for row in selector_rows
        if row["matrix_id"] == "suitesparse:JGD_Trefethen/Trefethen_20b"
        and row["label_is_oracle"]
    )
    curtis_oracle = next(
        row
        for row in selector_rows
        if row["matrix_id"] == "suitesparse:HB/curtis54" and row["label_is_oracle"]
    )

    assert trefethen.candidate_id == trefethen_oracle["candidate_id"]
    assert trefethen.plan.solver["name"] == trefethen_oracle["solver"]
    assert trefethen.plan.audit["is_oracle"] is True
    if trefethen.plan.solver["name"] == "richardson":
        assert trefethen.plan.solver["omega"] == pytest.approx(2.0 / 3.0)
    assert "taichi_csr_chebyshev_jacobi_float64" in {
        trefethen.candidate_id,
        *trefethen.fallback_candidate_ids,
    }
    chebyshev_fallback = next(
        row
        for row in (
            [{"candidate_id": trefethen.candidate_id, "solver": trefethen.plan.solver}]
            + trefethen.plan.fallback_chain
        )
        if row["candidate_id"] == "taichi_csr_chebyshev_jacobi_float64"
    )
    assert chebyshev_fallback["solver"]["name"] == "chebyshev"
    assert chebyshev_fallback["solver"]["lambda_min"] > 0.0
    assert (
        chebyshev_fallback["solver"]["lambda_min"]
        < chebyshev_fallback["solver"]["lambda_max"]
    )
    assert curtis.candidate_id == curtis_oracle["candidate_id"]
    assert curtis.plan.solver["name"] == "bicgstab"
    assert curtis.plan.preconditioner["name"] == curtis_oracle["preconditioner"]
    assert curtis.plan.audit["is_oracle"] is True
    failed_curtis = next(
        row
        for row in selector_rows
        if row["matrix_id"] == "suitesparse:HB/curtis54"
        and row["target_status"] == "screened_out"
    )
    assert failed_curtis["candidate_id"] not in set(curtis.fallback_candidate_ids)
    assert set(curtis.fallback_candidate_ids) == {
        row["candidate_id"]
        for row in selector_rows
        if row["matrix_id"] == "suitesparse:HB/curtis54"
        and not row["label_is_oracle"]
        and row["target_status"] == "success"
    }


def test_csr_artifact_selector_rejects_unknown_objective():
    rows = {
        str(row["matrix_id"]): row
        for row in read_jsonl("runs/phase1_suitesparse_csr_import/csr_matrices.jsonl")
    }
    selector = CsrArtifactPolicySelector.from_selector_rows(
        "runs/phase1_csr_selector_readiness/csr_selector_rows.jsonl"
    )

    with pytest.raises(ValueError):
        selector.select(
            rows["suitesparse:JGD_Trefethen/Trefethen_20b"],
            objective="unknown",
        )
