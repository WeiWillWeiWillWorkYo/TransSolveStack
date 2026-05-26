from transsolvestack.datasets.csr import CsrMatrix
from transsolvestack.datasets.csr_diagnostics import (
    build_csr_matrix_diagnostics,
    classify_unresolved_matrix,
)


def test_identity_csr_diagnostics_detect_spd_structure():
    csr = CsrMatrix(
        matrix_id="identity3",
        n_rows=3,
        n_cols=3,
        row_ptr=(0, 1, 2, 3),
        col_ind=(0, 1, 2),
        values=(1.0, 1.0, 1.0),
        field="real",
        symmetry="symmetric",
        source_path="fixture",
    )

    diagnostics = build_csr_matrix_diagnostics(csr, lanczos_steps=3)

    assert diagnostics["diagonal"]["zero_count"] == 0
    assert diagnostics["diagonal_dominance"]["weakly_dominant_rows"] == 3
    assert diagnostics["symmetry_check"]["relative_frobenius_asymmetry"] == 0.0
    assert diagnostics["dense_cholesky_probe"]["status"] == "passed"
    assert diagnostics["symmetric_spectrum_probe"]["negative_ritz_count"] == 0
    assert (
        classify_unresolved_matrix(diagnostics, best_solution_error=0.01)
        == "ic0_or_equilibration_preconditioner"
    )


def test_zero_diagonal_symmetric_pattern_routes_to_formulation_diagnostic():
    csr = CsrMatrix(
        matrix_id="adjacency2",
        n_rows=2,
        n_cols=2,
        row_ptr=(0, 1, 2),
        col_ind=(1, 0),
        values=(1.0, 1.0),
        field="pattern",
        symmetry="symmetric",
        source_path="fixture",
    )

    diagnostics = build_csr_matrix_diagnostics(csr, lanczos_steps=2)

    assert diagnostics["diagonal"]["zero_count"] == 2
    assert diagnostics["symmetry_check"]["relative_frobenius_asymmetry"] == 0.0
    assert diagnostics["dense_cholesky_probe"]["status"] == "failed"
    assert diagnostics["symmetric_spectrum_probe"]["ritz_min"] < 0.0
    assert (
        classify_unresolved_matrix(diagnostics, best_solution_error=0.02)
        == "formulation_diagnostic_required"
    )
