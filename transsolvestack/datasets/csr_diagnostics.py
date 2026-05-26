"""Deterministic CPU-side diagnostics for imported CSR matrices."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from transsolvestack.datasets.csr import CsrMatrix


def build_csr_matrix_diagnostics(
    csr: CsrMatrix,
    *,
    lanczos_steps: int = 48,
    dense_cholesky_limit: int = 900,
) -> dict[str, Any]:
    """Return structural and conditioning diagnostics for a CSR matrix.

    This is not a solver path. It is an offline diagnostic boundary used to
    decide whether failed fallback candidates deserve a GPU probe, a stronger
    preconditioner, or a matrix-formulation change.
    """

    arrays = _CsrArrays.from_csr(csr)
    diagonal = arrays.diagonal()
    row_abs_sum = arrays.row_abs_sum()
    col_abs_sum = arrays.col_abs_sum()
    offdiag_abs_sum = row_abs_sum - np.abs(diagonal)
    dominance_ratio = np.divide(
        np.abs(diagonal),
        offdiag_abs_sum,
        out=np.full(csr.n_rows, np.inf, dtype=np.float64),
        where=offdiag_abs_sum > 0.0,
    )
    symmetry = _symmetry_diagnostic(arrays)
    graph = _structural_graph_diagnostic(arrays)
    lanczos = _lanczos_diagnostic(
        arrays,
        use_symmetric_part=symmetry["relative_frobenius_asymmetry"] > 1.0e-12,
        max_steps=lanczos_steps,
    )
    cholesky = _cholesky_probe(arrays, symmetry, dense_cholesky_limit)
    return {
        "matrix_id": csr.matrix_id,
        "n_rows": csr.n_rows,
        "n_cols": csr.n_cols,
        "csr_nnz": csr.nnz,
        "density": csr.nnz / max(float(csr.n_rows * csr.n_cols), 1.0),
        "field": csr.field,
        "symmetry": csr.symmetry,
        "diagonal": {
            "zero_count": int(np.count_nonzero(np.abs(diagonal) <= 1.0e-30)),
            "near_zero_count": int(np.count_nonzero(np.abs(diagonal) <= 1.0e-12)),
            "negative_count": int(np.count_nonzero(diagonal < 0.0)),
            "positive_count": int(np.count_nonzero(diagonal > 0.0)),
            "min": _finite_or_none(np.min(diagonal) if diagonal.size else None),
            "max": _finite_or_none(np.max(diagonal) if diagonal.size else None),
            "min_abs": _finite_or_none(np.min(np.abs(diagonal)) if diagonal.size else None),
            "max_abs": _finite_or_none(np.max(np.abs(diagonal)) if diagonal.size else None),
        },
        "row_structure": {
            "zero_row_count": int(np.count_nonzero(np.diff(arrays.row_ptr) == 0)),
            "min_nnz_per_row": int(np.min(np.diff(arrays.row_ptr))),
            "max_nnz_per_row": int(np.max(np.diff(arrays.row_ptr))),
            "min_row_abs_sum": _finite_or_none(np.min(row_abs_sum)),
            "max_row_abs_sum": _finite_or_none(np.max(row_abs_sum)),
            "min_col_abs_sum": _finite_or_none(np.min(col_abs_sum)),
            "max_col_abs_sum": _finite_or_none(np.max(col_abs_sum)),
        },
        "diagonal_dominance": {
            "weakly_dominant_rows": int(
                np.count_nonzero(np.abs(diagonal) + 1.0e-30 >= offdiag_abs_sum)
            ),
            "strictly_dominant_rows": int(np.count_nonzero(np.abs(diagonal) > offdiag_abs_sum)),
            "min_ratio": _finite_or_none(np.min(dominance_ratio)),
            "median_ratio": _finite_or_none(np.median(dominance_ratio)),
            "max_ratio": _finite_or_none(np.max(dominance_ratio[np.isfinite(dominance_ratio)]))
            if np.any(np.isfinite(dominance_ratio))
            else None,
        },
        "value_signs": {
            "positive_values": int(np.count_nonzero(arrays.values > 0.0)),
            "negative_values": int(np.count_nonzero(arrays.values < 0.0)),
            "zero_values": int(np.count_nonzero(arrays.values == 0.0)),
        },
        "symmetry_check": symmetry,
        "structural_graph": graph,
        "symmetric_spectrum_probe": lanczos,
        "dense_cholesky_probe": cholesky,
    }


def classify_unresolved_matrix(
    diagnostics: dict[str, Any],
    *,
    best_solution_error: float | None,
) -> str:
    diag = diagnostics["diagonal"]
    spectrum = diagnostics["symmetric_spectrum_probe"]
    cholesky = diagnostics["dense_cholesky_probe"]
    asym = float(diagnostics["symmetry_check"]["relative_frobenius_asymmetry"])
    min_ritz = spectrum.get("ritz_min")
    if asym > 1.0e-8:
        return "ilu_or_nonsymmetric_scaling_preconditioner"
    if (
        int(diag["zero_count"]) > 0
        and min_ritz is not None
        and float(min_ritz) < -1.0e-8
    ):
        return "formulation_diagnostic_required"
    if cholesky["status"] == "passed" and (
        best_solution_error is None or best_solution_error > 5.0e-3
    ):
        return "ic0_or_equilibration_preconditioner"
    if cholesky["status"] == "failed":
        return "indefinite_or_shifted_formulation_diagnostic"
    return "preconditioner_followup_required"


class _CsrArrays:
    def __init__(
        self,
        *,
        n_rows: int,
        n_cols: int,
        row_ptr: np.ndarray,
        col_ind: np.ndarray,
        values: np.ndarray,
    ) -> None:
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.row_ptr = row_ptr
        self.col_ind = col_ind
        self.values = values

    @classmethod
    def from_csr(cls, csr: CsrMatrix) -> "_CsrArrays":
        return cls(
            n_rows=csr.n_rows,
            n_cols=csr.n_cols,
            row_ptr=np.asarray(csr.row_ptr, dtype=np.int64),
            col_ind=np.asarray(csr.col_ind, dtype=np.int64),
            values=np.asarray(csr.values, dtype=np.float64),
        )

    def matvec(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(self.n_rows, dtype=np.float64)
        for row in range(self.n_rows):
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            y[row] = float(np.dot(self.values[start:end], x[self.col_ind[start:end]]))
        return y

    def transpose_matvec(self, x: np.ndarray) -> np.ndarray:
        y = np.zeros(self.n_cols, dtype=np.float64)
        for row in range(self.n_rows):
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            y[self.col_ind[start:end]] += self.values[start:end] * x[row]
        return y

    def diagonal(self) -> np.ndarray:
        diagonal = np.zeros(self.n_rows, dtype=np.float64)
        for row in range(self.n_rows):
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            columns = self.col_ind[start:end]
            hits = np.where(columns == row)[0]
            if hits.size:
                diagonal[row] = float(np.sum(self.values[start:end][hits]))
        return diagonal

    def row_abs_sum(self) -> np.ndarray:
        out = np.zeros(self.n_rows, dtype=np.float64)
        for row in range(self.n_rows):
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            out[row] = float(np.sum(np.abs(self.values[start:end])))
        return out

    def col_abs_sum(self) -> np.ndarray:
        out = np.zeros(self.n_cols, dtype=np.float64)
        for row in range(self.n_rows):
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            out[self.col_ind[start:end]] += np.abs(self.values[start:end])
        return out

    def dense(self) -> np.ndarray:
        dense = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
        for row in range(self.n_rows):
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            dense[row, self.col_ind[start:end]] = self.values[start:end]
        return dense


def _symmetry_diagnostic(arrays: _CsrArrays) -> dict[str, Any]:
    if arrays.n_rows != arrays.n_cols:
        return {
            "square": False,
            "relative_frobenius_asymmetry": None,
            "missing_transpose_entries": None,
            "max_abs_asymmetry": None,
        }
    entries: dict[tuple[int, int], float] = {}
    for row in range(arrays.n_rows):
        start = int(arrays.row_ptr[row])
        end = int(arrays.row_ptr[row + 1])
        for offset in range(start, end):
            entries[(row, int(arrays.col_ind[offset]))] = float(arrays.values[offset])
    keys = set(entries) | {(col, row) for row, col in entries}
    diff_sq = 0.0
    norm_sq = 0.0
    max_abs = 0.0
    missing = 0
    for row, col in keys:
        value = entries.get((row, col), 0.0)
        transpose_value = entries.get((col, row), 0.0)
        if (col, row) not in entries:
            missing += 1
        diff = value - transpose_value
        diff_sq += diff * diff
        norm_sq += value * value
        max_abs = max(max_abs, abs(diff))
    return {
        "square": True,
        "relative_frobenius_asymmetry": float(np.sqrt(diff_sq) / max(np.sqrt(norm_sq), 1.0e-30)),
        "missing_transpose_entries": missing,
        "max_abs_asymmetry": max_abs,
    }


def _structural_graph_diagnostic(arrays: _CsrArrays) -> dict[str, Any]:
    if arrays.n_rows != arrays.n_cols:
        return {"square": False, "connected_components": None, "isolated_vertices": None}
    adjacency: list[list[int]] = [[] for _ in range(arrays.n_rows)]
    for row in range(arrays.n_rows):
        start = int(arrays.row_ptr[row])
        end = int(arrays.row_ptr[row + 1])
        for col in arrays.col_ind[start:end]:
            col_int = int(col)
            if col_int != row:
                adjacency[row].append(col_int)
                adjacency[col_int].append(row)
    seen = [False] * arrays.n_rows
    components = 0
    isolated = 0
    for start in range(arrays.n_rows):
        if seen[start]:
            continue
        components += 1
        if not adjacency[start]:
            isolated += 1
        seen[start] = True
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            for neighbor in adjacency[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    queue.append(neighbor)
    return {
        "square": True,
        "connected_components": components,
        "isolated_vertices": isolated,
    }


def _lanczos_diagnostic(
    arrays: _CsrArrays,
    *,
    use_symmetric_part: bool,
    max_steps: int,
) -> dict[str, Any]:
    if arrays.n_rows != arrays.n_cols:
        return {"status": "skipped_non_square"}
    n = arrays.n_rows
    steps = max(1, min(max_steps, n))
    q = _initial_vector(n)
    previous = np.zeros(n, dtype=np.float64)
    beta = 0.0
    alphas: list[float] = []
    betas: list[float] = []
    basis: list[np.ndarray] = []
    for _ in range(steps):
        basis.append(q.copy())
        z = _symmetric_matvec(arrays, q, use_symmetric_part=use_symmetric_part)
        if beta:
            z = z - beta * previous
        alpha = float(np.dot(q, z))
        z = z - alpha * q
        for vector in basis:
            z = z - float(np.dot(vector, z)) * vector
        beta = float(np.linalg.norm(z))
        alphas.append(alpha)
        if beta <= 1.0e-12:
            break
        betas.append(beta)
        previous = q
        q = z / beta
    tridiagonal = np.diag(np.asarray(alphas, dtype=np.float64))
    if len(alphas) > 1:
        off = np.asarray(betas[: len(alphas) - 1], dtype=np.float64)
        tridiagonal += np.diag(off, 1) + np.diag(off, -1)
    ritz = np.linalg.eigvalsh(tridiagonal)
    abs_ritz = np.abs(ritz)
    positive_abs = abs_ritz[abs_ritz > 1.0e-12]
    return {
        "status": "passed",
        "operator": "symmetric_part" if use_symmetric_part else "matrix",
        "steps": len(alphas),
        "ritz_min": float(ritz[0]),
        "ritz_max": float(ritz[-1]),
        "ritz_abs_min_nonzero": None
        if positive_abs.size == 0
        else float(np.min(positive_abs)),
        "ritz_abs_max": float(np.max(abs_ritz)) if abs_ritz.size else None,
        "ritz_condition_proxy": None
        if positive_abs.size == 0
        else float(np.max(abs_ritz) / np.min(positive_abs)),
        "negative_ritz_count": int(np.count_nonzero(ritz < -1.0e-10)),
        "near_zero_ritz_count": int(np.count_nonzero(abs_ritz <= 1.0e-10)),
    }


def _cholesky_probe(
    arrays: _CsrArrays,
    symmetry: dict[str, Any],
    dense_cholesky_limit: int,
) -> dict[str, Any]:
    if arrays.n_rows != arrays.n_cols:
        return {"status": "skipped_non_square"}
    if symmetry["relative_frobenius_asymmetry"] is None or float(
        symmetry["relative_frobenius_asymmetry"]
    ) > 1.0e-12:
        return {"status": "skipped_nonsymmetric"}
    if arrays.n_rows > dense_cholesky_limit:
        return {"status": "skipped_size_limit", "limit": dense_cholesky_limit}
    dense = arrays.dense()
    try:
        np.linalg.cholesky(dense)
    except np.linalg.LinAlgError as exc:
        return {"status": "failed", "reason": str(exc)}
    return {"status": "passed"}


def _symmetric_matvec(
    arrays: _CsrArrays,
    vector: np.ndarray,
    *,
    use_symmetric_part: bool,
) -> np.ndarray:
    if use_symmetric_part:
        return 0.5 * (arrays.matvec(vector) + arrays.transpose_matvec(vector))
    return arrays.matvec(vector)


def _initial_vector(n: int) -> np.ndarray:
    values = np.sin(np.arange(1, n + 1, dtype=np.float64) * 0.61803398875) + 1.0
    norm = float(np.linalg.norm(values))
    if norm <= 1.0e-30:
        values = np.ones(n, dtype=np.float64)
        norm = float(np.linalg.norm(values))
    return values / norm


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return value
