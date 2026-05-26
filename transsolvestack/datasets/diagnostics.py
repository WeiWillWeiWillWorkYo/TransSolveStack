"""CSR matrix diagnostics used by solver-selection policy work."""

from __future__ import annotations

import math
from dataclasses import dataclass

from transsolvestack.datasets.csr import CsrMatrix


@dataclass(frozen=True)
class CsrMatrixDiagnostic:
    matrix_id: str
    n_rows: int
    n_cols: int
    csr_nnz: int
    field: str
    declared_symmetry: str
    is_square: bool
    actual_symmetric: bool
    symmetry_missing_pairs: int
    symmetry_max_abs_error: float
    symmetry_relative_error: float
    diagonal_present_count: int
    zero_diagonal_count: int
    min_abs_diagonal: float | None
    max_abs_diagonal: float | None
    has_nonpositive_diagonal: bool
    cg_candidate: bool
    recommended_precision: str
    skip_reason: str | None


def diagnose_csr_matrix(
    csr: CsrMatrix,
    *,
    symmetry_abs_tol: float = 1.0e-10,
    symmetry_rel_tol: float = 1.0e-8,
) -> CsrMatrixDiagnostic:
    """Build selector-oriented diagnostics for a CSR matrix."""

    csr.validate()
    is_square = csr.n_rows == csr.n_cols
    if is_square:
        missing_pairs, symmetry_max_abs, symmetry_scale = _symmetry_error(csr)
    else:
        missing_pairs, symmetry_max_abs, symmetry_scale = 0, math.inf, 1.0
    symmetry_relative = symmetry_max_abs / max(symmetry_scale, 1.0e-30)
    actual_symmetric = (
        is_square
        and missing_pairs == 0
        and symmetry_max_abs <= max(symmetry_abs_tol, symmetry_rel_tol * symmetry_scale)
    )
    diag = _diagonal(csr) if is_square else ()
    diagonal_present_count = sum(1 for value in diag if value is not None)
    zero_diagonal_count = sum(1 for value in diag if value is None or abs(value) <= 1.0e-30)
    present_diag = tuple(abs(value) for value in diag if value is not None)
    has_nonpositive_diagonal = any(value is not None and value <= 0.0 for value in diag)
    skip_reason = _cg_skip_reason(
        is_square=is_square,
        actual_symmetric=actual_symmetric,
        zero_diagonal_count=zero_diagonal_count,
    )
    return CsrMatrixDiagnostic(
        matrix_id=csr.matrix_id,
        n_rows=csr.n_rows,
        n_cols=csr.n_cols,
        csr_nnz=csr.nnz,
        field=csr.field,
        declared_symmetry=csr.symmetry,
        is_square=is_square,
        actual_symmetric=actual_symmetric,
        symmetry_missing_pairs=missing_pairs,
        symmetry_max_abs_error=symmetry_max_abs,
        symmetry_relative_error=symmetry_relative,
        diagonal_present_count=diagonal_present_count,
        zero_diagonal_count=zero_diagonal_count,
        min_abs_diagonal=min(present_diag) if present_diag else None,
        max_abs_diagonal=max(present_diag) if present_diag else None,
        has_nonpositive_diagonal=has_nonpositive_diagonal,
        cg_candidate=skip_reason is None,
        recommended_precision=_recommended_precision(csr),
        skip_reason=skip_reason,
    )


def _symmetry_error(csr: CsrMatrix) -> tuple[int, float, float]:
    entries: dict[tuple[int, int], float] = {}
    scale = 0.0
    for row in range(csr.n_rows):
        for offset in range(csr.row_ptr[row], csr.row_ptr[row + 1]):
            col = csr.col_ind[offset]
            value = csr.values[offset]
            entries[(row, col)] = value
            scale = max(scale, abs(value))
    missing_pairs = 0
    max_abs = 0.0
    for (row, col), value in entries.items():
        mirror = entries.get((col, row))
        if mirror is None:
            missing_pairs += 1
            max_abs = max(max_abs, abs(value))
        else:
            max_abs = max(max_abs, abs(value - mirror))
    return missing_pairs, max_abs, scale


def _diagonal(csr: CsrMatrix) -> tuple[float | None, ...]:
    diag: list[float | None] = [None for _ in range(csr.n_rows)]
    for row in range(csr.n_rows):
        total = 0.0
        present = False
        for offset in range(csr.row_ptr[row], csr.row_ptr[row + 1]):
            if csr.col_ind[offset] == row:
                total += csr.values[offset]
                present = True
        if present:
            diag[row] = total
    return tuple(diag)


def _cg_skip_reason(
    *,
    is_square: bool,
    actual_symmetric: bool,
    zero_diagonal_count: int,
) -> str | None:
    if not is_square:
        return "not_square"
    if not actual_symmetric:
        return "not_symmetric"
    if zero_diagonal_count:
        return "zero_diagonal"
    return None


def _recommended_precision(csr: CsrMatrix) -> str:
    max_abs = max((abs(value) for value in csr.values), default=0.0)
    min_abs = min((abs(value) for value in csr.values if value != 0.0), default=max_abs)
    dynamic_range = max_abs / max(min_abs, 1.0e-30)
    if dynamic_range > 1.0e8 or max_abs > 1.0e6:
        return "float64"
    return "float32"
