"""CPU screens for currently blocked CSR candidate-coverage gaps."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from transsolvestack.datasets.csr import CsrMatrix



def screen_blocked_gap_candidate(
    csr: CsrMatrix,
    candidate: dict[str, Any],
) -> dict[str, Any] | None:
    solver = str(candidate["solver"])
    preconditioner = str(candidate["preconditioner"])
    max_iter = int(candidate["max_iter"])
    tolerance_rel = float(candidate["tolerance_rel"])
    if solver == "bicgstab" and preconditioner == "ilu0":
        params = dict(candidate.get("preconditioner_parameters", {}))
        return _cpu_bicgstab_ilu0_screen(
            csr,
            tolerance_rel=tolerance_rel,
            max_iter=max_iter,
            pivot_tolerance=float(params.get("pivot_tolerance", 1.0e-12)),
        )
    if solver == "bicgstab" and preconditioner == "row_column_equilibration":
        params = dict(candidate.get("preconditioner_parameters", {}))
        return _cpu_bicgstab_row_column_equilibration_screen(
            csr,
            tolerance_rel=tolerance_rel,
            max_iter=max_iter,
            passes=int(params.get("passes", 4)),
        )
    if solver == "pcg" and preconditioner == "symmetric_equilibration":
        return _cpu_pcg_symmetric_equilibration_screen(
            csr,
            tolerance_rel=tolerance_rel,
            max_iter=max_iter,
        )
    if solver == "chebyshev" and preconditioner == "jacobi":
        if csr.symmetry != "symmetric" or csr.n_rows != csr.n_cols:
            return _failed_screen(breakdown="chebyshev_requires_symmetric_square")
        params = dict(candidate.get("solver_parameters", {}))
        bounds = _jacobi_spectral_bounds(
            csr,
            lambda_min=params.get("lambda_min"),
            lambda_max=params.get("lambda_max"),
        )
        if not bounds["success"]:
            return _failed_screen(
                breakdown=str(bounds["breakdown"]),
                extra={"spectral_bounds": bounds},
            )
        return _cpu_chebyshev_screen(
            csr,
            tolerance_rel=tolerance_rel,
            max_iter=max_iter,
            lambda_min=float(bounds["lambda_min"]),
            lambda_max=float(bounds["lambda_max"]),
            spectral_bounds_source=str(bounds["source"]),
        )
    return None


@dataclass(frozen=True)
class _CsrNp:
    n_rows: int
    n_cols: int
    row_ptr: np.ndarray
    col_ind: np.ndarray
    values: np.ndarray
    diagonal: np.ndarray

    @classmethod
    def from_csr(cls, csr: CsrMatrix) -> "_CsrNp":
        row_ptr = np.asarray(csr.row_ptr, dtype=np.int64)
        col_ind = np.asarray(csr.col_ind, dtype=np.int64)
        values = np.asarray(csr.values, dtype=np.float64)
        diagonal = np.zeros(csr.n_rows, dtype=np.float64)
        for row in range(csr.n_rows):
            start = int(row_ptr[row])
            end = int(row_ptr[row + 1])
            columns = col_ind[start:end]
            hits = np.where(columns == row)[0]
            if hits.size:
                diagonal[row] = float(np.sum(values[start:end][hits]))
        return cls(
            n_rows=csr.n_rows,
            n_cols=csr.n_cols,
            row_ptr=row_ptr,
            col_ind=col_ind,
            values=values,
            diagonal=diagonal,
        )

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        out = np.zeros(self.n_rows, dtype=np.float64)
        for row in range(self.n_rows):
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            out[row] = float(np.dot(self.values[start:end], vector[self.col_ind[start:end]]))
        return out

    def to_dense(self, *, row_scale: np.ndarray | None = None) -> np.ndarray:
        dense = np.zeros((self.n_rows, self.n_cols), dtype=np.float64)
        for row in range(self.n_rows):
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            for offset in range(start, end):
                col = int(self.col_ind[offset])
                value = float(self.values[offset])
                if row_scale is not None:
                    value *= float(row_scale[row]) * float(row_scale[col])
                dense[row, col] += value
        return dense


@dataclass(frozen=True)
class _LinearSolveScreen:
    x: np.ndarray
    iterations: int
    relative_residual: float
    solution_relative_error: float
    breakdown: str | None
    residual_history: tuple[float, ...]

    @property
    def success(self) -> bool:
        return self.breakdown is None


@dataclass(frozen=True)
class _Ilu0Factor:
    rows: tuple[dict[int, float], ...]
    min_abs_pivot: float
    pivot_tolerance: float

    def apply(self, values: np.ndarray) -> np.ndarray:
        n = len(self.rows)
        y = np.zeros(n, dtype=np.float64)
        for row in range(n):
            total = float(values[row])
            for col, value in self.rows[row].items():
                if col < row:
                    total -= value * y[col]
            y[row] = total
        x = np.zeros(n, dtype=np.float64)
        for row in range(n - 1, -1, -1):
            total = y[row]
            pivot = self.rows[row][row]
            for col, value in self.rows[row].items():
                if col > row:
                    total -= value * x[col]
            x[row] = total / pivot
        return x


def _cpu_bicgstab_ilu0_screen(
    csr: CsrMatrix,
    *,
    tolerance_rel: float,
    max_iter: int,
    pivot_tolerance: float,
) -> dict[str, Any]:
    data = _CsrNp.from_csr(csr)
    factor = _ilu0_factor(data, pivot_tolerance=pivot_tolerance)
    if isinstance(factor, dict):
        return _failed_screen(
            breakdown=str(factor["breakdown"]),
            extra={
                "pivot_tolerance": pivot_tolerance,
                "ilu0_valid": False,
            },
        )
    rhs = data.matvec(np.ones(csr.n_cols, dtype=np.float64))
    solve = _bicgstab_solve(
        matvec=data.matvec,
        rhs=rhs,
        apply_preconditioner=factor.apply,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )
    return _screen_from_original_solution(
        csr,
        rhs=rhs,
        x=solve.x,
        solve=solve,
        tolerance_rel=tolerance_rel,
        extra={
            "pivot_tolerance": pivot_tolerance,
            "ilu0_valid": True,
            "ilu0_min_abs_pivot": factor.min_abs_pivot,
        },
    )


def _cpu_bicgstab_row_column_equilibration_screen(
    csr: CsrMatrix,
    *,
    tolerance_rel: float,
    max_iter: int,
    passes: int,
) -> dict[str, Any]:
    data = _CsrNp.from_csr(csr)
    row_scale, col_scale = _row_column_l1_scales(data, passes=passes)
    rhs = data.matvec(np.ones(csr.n_cols, dtype=np.float64))
    rhs_scaled = row_scale * rhs

    def matvec_scaled(vector: np.ndarray) -> np.ndarray:
        return row_scale * data.matvec(col_scale * vector)

    solve = _bicgstab_solve(
        matvec=matvec_scaled,
        rhs=rhs_scaled,
        apply_preconditioner=lambda values: values.copy(),
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )
    x_original = col_scale * solve.x
    return _screen_from_original_solution(
        csr,
        rhs=rhs,
        x=x_original,
        solve=solve,
        tolerance_rel=tolerance_rel,
        extra={
            "passes": passes,
            "scaled_relative_residual": solve.relative_residual,
            "row_scale_min": float(np.min(row_scale)),
            "row_scale_max": float(np.max(row_scale)),
            "col_scale_min": float(np.min(col_scale)),
            "col_scale_max": float(np.max(col_scale)),
        },
    )


def _cpu_pcg_symmetric_equilibration_screen(
    csr: CsrMatrix,
    *,
    tolerance_rel: float,
    max_iter: int,
) -> dict[str, Any]:
    if csr.symmetry != "symmetric" or csr.n_rows != csr.n_cols:
        return _failed_screen(breakdown="symmetric_equilibration_requires_symmetric_square")
    data = _CsrNp.from_csr(csr)
    if np.any(data.diagonal <= 1.0e-30):
        return _failed_screen(breakdown="nonpositive_diagonal")
    scale = np.divide(
        1.0,
        np.sqrt(data.diagonal),
        out=np.zeros_like(data.diagonal),
        where=data.diagonal > 1.0e-30,
    )
    rhs = data.matvec(np.ones(csr.n_cols, dtype=np.float64))
    rhs_scaled = scale * rhs

    def matvec_scaled(vector: np.ndarray) -> np.ndarray:
        return scale * data.matvec(scale * vector)

    solve = _cg_solve(
        matvec=matvec_scaled,
        rhs=rhs_scaled,
        max_iter=max_iter,
        tolerance_rel=tolerance_rel,
    )
    x_original = scale * solve.x
    return _screen_from_original_solution(
        csr,
        rhs=rhs,
        x=x_original,
        solve=solve,
        tolerance_rel=tolerance_rel,
        extra={
            "scaled_relative_residual": solve.relative_residual,
            "min_diagonal": float(np.min(data.diagonal)),
            "max_diagonal": float(np.max(data.diagonal)),
        },
    )


def _cpu_chebyshev_screen(
    csr: CsrMatrix,
    *,
    tolerance_rel: float,
    max_iter: int,
    lambda_min: float,
    lambda_max: float,
    spectral_bounds_source: str,
) -> dict[str, Any]:
    if not 0.0 < lambda_min < lambda_max:
        return _failed_screen(breakdown="invalid_spectral_bounds")
    data = _CsrNp.from_csr(csr)
    if np.any(np.abs(data.diagonal) <= 1.0e-30):
        return _failed_screen(breakdown="zero_diagonal")
    x = np.zeros(csr.n_cols, dtype=np.float64)
    rhs = data.matvec(np.ones(csr.n_cols, dtype=np.float64))
    r = rhs.copy()
    p = np.zeros(csr.n_cols, dtype=np.float64)
    b_norm = max(float(np.linalg.norm(rhs)), 1.0e-30)
    rel = float(np.linalg.norm(r)) / b_norm
    history = [rel]
    d = 0.5 * (lambda_max + lambda_min)
    c = 0.5 * (lambda_max - lambda_min)
    alpha = 0.0
    breakdown = None
    iterations = 0
    for iteration in range(1, max_iter + 1):
        z = np.divide(
            r,
            data.diagonal,
            out=r.copy(),
            where=np.abs(data.diagonal) > 1.0e-30,
        )
        if iteration == 1:
            p = z.copy()
            alpha = 1.0 / d
        else:
            if abs(alpha) < 1.0e-30:
                breakdown = "zero_alpha"
                break
            beta = 0.5 * (c * alpha) ** 2 if iteration == 2 else (0.5 * c * alpha) ** 2
            denom = d - beta / alpha
            if not math.isfinite(denom) or abs(denom) < 1.0e-30:
                breakdown = "zero_or_nonfinite_alpha_denominator"
                break
            alpha = 1.0 / denom
            if not math.isfinite(alpha):
                breakdown = "nonfinite_alpha"
                break
            p = z + beta * p
        ap = data.matvec(p)
        x = x + alpha * p
        r = r - alpha * ap
        rel = float(np.linalg.norm(r)) / b_norm
        if not math.isfinite(rel):
            breakdown = "nonfinite_residual"
            break
        history.append(rel)
        iterations = iteration
        if rel <= tolerance_rel:
            break
    solve = _LinearSolveScreen(
        x=x,
        iterations=iterations,
        relative_residual=rel,
        solution_relative_error=_relative_error_to_ones(x),
        breakdown=breakdown,
        residual_history=tuple(history),
    )
    return _screen_from_original_solution(
        csr,
        rhs=rhs,
        x=x,
        solve=solve,
        tolerance_rel=tolerance_rel,
        extra={
            "lambda_min": lambda_min,
            "lambda_max": lambda_max,
            "spectral_bounds_source": spectral_bounds_source,
        },
    )


def _bicgstab_solve(
    *,
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    apply_preconditioner: Callable[[np.ndarray], np.ndarray],
    max_iter: int,
    tolerance_rel: float,
) -> _LinearSolveScreen:
    n = rhs.shape[0]
    x = np.zeros(n, dtype=np.float64)
    r = rhs.copy()
    r_hat = r.copy()
    p = np.zeros(n, dtype=np.float64)
    v = np.zeros(n, dtype=np.float64)
    b_norm = max(float(np.linalg.norm(rhs)), 1.0e-30)
    rel = float(np.linalg.norm(r)) / b_norm
    history = [rel]
    rho_old = 1.0
    alpha = 1.0
    omega = 1.0
    breakdown = None
    iterations = 0
    for iteration in range(1, max_iter + 1):
        rho_new = float(np.dot(r_hat, r))
        if not math.isfinite(rho_new) or abs(rho_new) < 1.0e-30:
            breakdown = "zero_or_nonfinite_rho"
            break
        if iteration == 1:
            p = r.copy()
        else:
            if abs(omega) < 1.0e-30:
                breakdown = "zero_omega"
                break
            beta = (rho_new / rho_old) * (alpha / omega)
            if not math.isfinite(beta):
                breakdown = "nonfinite_beta"
                break
            p = r + beta * (p - omega * v)
        phat = apply_preconditioner(p)
        v = matvec(phat)
        denom = float(np.dot(r_hat, v))
        if not math.isfinite(denom) or abs(denom) < 1.0e-30:
            breakdown = "zero_or_nonfinite_alpha_denominator"
            break
        alpha = rho_new / denom
        if not math.isfinite(alpha):
            breakdown = "nonfinite_alpha"
            break
        x_alpha = x + alpha * phat
        s = r - alpha * v
        rel_s = float(np.linalg.norm(s)) / b_norm
        if rel_s <= tolerance_rel:
            x = x_alpha
            r = s
            rel = rel_s
            history.append(rel)
            iterations = iteration
            break
        shat = apply_preconditioner(s)
        t = matvec(shat)
        tt = float(np.dot(t, t))
        if not math.isfinite(tt) or abs(tt) < 1.0e-30:
            breakdown = "zero_or_nonfinite_omega_denominator"
            break
        omega = float(np.dot(t, s)) / tt
        if not math.isfinite(omega) or abs(omega) < 1.0e-30:
            breakdown = "zero_or_nonfinite_omega"
            break
        x = x_alpha + omega * shat
        r = s - omega * t
        rel = float(np.linalg.norm(r)) / b_norm
        history.append(rel)
        iterations = iteration
        if rel <= tolerance_rel:
            break
        rho_old = rho_new
    return _LinearSolveScreen(
        x=x,
        iterations=iterations,
        relative_residual=rel,
        solution_relative_error=_relative_error_to_ones(x),
        breakdown=breakdown,
        residual_history=tuple(history),
    )


def _cg_solve(
    *,
    matvec: Callable[[np.ndarray], np.ndarray],
    rhs: np.ndarray,
    max_iter: int,
    tolerance_rel: float,
) -> _LinearSolveScreen:
    n = rhs.shape[0]
    x = np.zeros(n, dtype=np.float64)
    r = rhs.copy()
    p = r.copy()
    rr_old = float(np.dot(r, r))
    b_norm = max(float(np.linalg.norm(rhs)), 1.0e-30)
    rel = math.sqrt(max(rr_old, 0.0)) / b_norm
    history = [rel]
    breakdown = None
    iterations = 0
    for iteration in range(1, max_iter + 1):
        ap = matvec(p)
        denom = float(np.dot(p, ap))
        if not math.isfinite(denom) or abs(denom) < 1.0e-30:
            breakdown = "zero_or_nonfinite_denominator"
            break
        if denom <= 0.0:
            breakdown = "non_positive_curvature"
            break
        alpha = rr_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = float(np.dot(r, r))
        rel = math.sqrt(max(rr_new, 0.0)) / b_norm
        history.append(rel)
        iterations = iteration
        if rel <= tolerance_rel:
            break
        if not math.isfinite(rr_new) or abs(rr_old) < 1.0e-30:
            breakdown = "zero_or_nonfinite_rr"
            break
        beta = rr_new / rr_old
        if not math.isfinite(beta):
            breakdown = "nonfinite_beta"
            break
        p = r + beta * p
        rr_old = rr_new
    return _LinearSolveScreen(
        x=x,
        iterations=iterations,
        relative_residual=rel,
        solution_relative_error=_relative_error_to_ones(x),
        breakdown=breakdown,
        residual_history=tuple(history),
    )


def _screen_from_original_solution(
    csr: CsrMatrix,
    *,
    rhs: np.ndarray,
    x: np.ndarray,
    solve: _LinearSolveScreen,
    tolerance_rel: float,
    extra: dict[str, Any],
) -> dict[str, Any]:
    original_rel = _relative_residual(csr, x, rhs)
    solution_error = _relative_error_to_ones(x)
    success = (
        solve.breakdown is None
        and original_rel <= tolerance_rel
        and solution_error <= 5.0e-3
        and solve.residual_history[-1] <= solve.residual_history[0]
    )
    breakdown = solve.breakdown
    if breakdown is None and original_rel > tolerance_rel:
        breakdown = "original_residual_above_tolerance"
    if breakdown is None and solution_error > 5.0e-3:
        breakdown = "solution_error_above_tolerance"
    if breakdown is None and solve.residual_history[-1] > solve.residual_history[0]:
        breakdown = "residual_did_not_drop"
    result = {
        "success": success,
        "iterations": solve.iterations,
        "relative_residual": original_rel,
        "solution_relative_error": solution_error,
        "breakdown": breakdown,
        "residual_history_length": len(solve.residual_history),
        "residual_history_head": tuple(solve.residual_history[:8]),
        "residual_history_tail": tuple(solve.residual_history[-8:]),
    }
    result.update(extra)
    return result


def _failed_screen(
    *,
    breakdown: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "success": False,
        "iterations": 0,
        "relative_residual": math.inf,
        "solution_relative_error": math.inf,
        "breakdown": breakdown,
        "residual_history_length": 0,
        "residual_history_head": (),
        "residual_history_tail": (),
    }
    if extra:
        result.update(extra)
    return result


def _ilu0_factor(
    data: _CsrNp,
    *,
    pivot_tolerance: float,
) -> _Ilu0Factor | dict[str, Any]:
    if pivot_tolerance < 0.0 or not math.isfinite(pivot_tolerance):
        return {"breakdown": "invalid_pivot_tolerance"}
    rows: list[dict[int, float]] = []
    for row in range(data.n_rows):
        start = int(data.row_ptr[row])
        end = int(data.row_ptr[row + 1])
        rows.append(
            {
                int(data.col_ind[offset]): float(data.values[offset])
                for offset in range(start, end)
            }
        )
        if row not in rows[-1]:
            return {"breakdown": "missing_diagonal"}
    min_abs_pivot = math.inf
    for row in range(data.n_rows):
        for col in sorted(key for key in rows[row] if key < row):
            pivot = rows[col].get(col)
            if pivot is None or abs(pivot) <= pivot_tolerance:
                return {
                    "breakdown": "zero_or_tiny_pivot",
                    "row": row,
                    "pivot_row": col,
                }
            multiplier = rows[row][col] / pivot
            rows[row][col] = multiplier
            for upper_col, upper_value in tuple(rows[col].items()):
                if upper_col > col and upper_col in rows[row]:
                    rows[row][upper_col] -= multiplier * upper_value
        pivot = rows[row].get(row)
        if pivot is None or abs(pivot) <= pivot_tolerance:
            return {"breakdown": "zero_or_tiny_pivot", "row": row}
        min_abs_pivot = min(min_abs_pivot, abs(pivot))
    return _Ilu0Factor(
        rows=tuple(rows),
        min_abs_pivot=float(min_abs_pivot),
        pivot_tolerance=pivot_tolerance,
    )


def _row_column_l1_scales(
    data: _CsrNp,
    *,
    passes: int,
) -> tuple[np.ndarray, np.ndarray]:
    passes = min(max(int(passes), 1), 16)
    row_scale = np.ones(data.n_rows, dtype=np.float64)
    col_scale = np.ones(data.n_cols, dtype=np.float64)
    for _ in range(passes):
        row_norm = np.zeros(data.n_rows, dtype=np.float64)
        col_norm = np.zeros(data.n_cols, dtype=np.float64)
        for row in range(data.n_rows):
            start = int(data.row_ptr[row])
            end = int(data.row_ptr[row + 1])
            for offset in range(start, end):
                col = int(data.col_ind[offset])
                scaled_abs = abs(row_scale[row] * data.values[offset] * col_scale[col])
                row_norm[row] += scaled_abs
                col_norm[col] += scaled_abs
        row_scale = np.divide(
            row_scale,
            np.sqrt(row_norm),
            out=row_scale.copy(),
            where=row_norm > 1.0e-30,
        )
        col_scale = np.divide(
            col_scale,
            np.sqrt(col_norm),
            out=col_scale.copy(),
            where=col_norm > 1.0e-30,
        )
        row_scale = np.clip(row_scale, 1.0e-12, 1.0e12)
        col_scale = np.clip(col_scale, 1.0e-12, 1.0e12)
    return row_scale, col_scale


def _jacobi_spectral_bounds(
    csr: CsrMatrix,
    *,
    lambda_min: Any = None,
    lambda_max: Any = None,
    max_dense_rows: int = 2048,
) -> dict[str, Any]:
    if csr.n_rows != csr.n_cols:
        return {"success": False, "breakdown": "spectral_bounds_require_square_matrix"}
    if lambda_min is not None and lambda_max is not None:
        lower = float(lambda_min)
        upper = float(lambda_max)
        if math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper:
            return {
                "success": True,
                "lambda_min": lower,
                "lambda_max": upper,
                "source": "candidate_explicit",
            }
        return {"success": False, "breakdown": "invalid_explicit_spectral_bounds"}
    if csr.n_rows > max_dense_rows:
        return {
            "success": False,
            "breakdown": "spectral_bounds_dense_size_limit",
            "max_dense_rows": max_dense_rows,
        }
    data = _CsrNp.from_csr(csr)
    if np.any(data.diagonal <= 0.0):
        return {"success": False, "breakdown": "nonpositive_diagonal"}
    inv_sqrt_diag = np.divide(
        1.0,
        np.sqrt(data.diagonal),
        out=np.zeros_like(data.diagonal),
        where=data.diagonal > 0.0,
    )
    dense = data.to_dense(row_scale=inv_sqrt_diag)
    eig = np.linalg.eigvalsh(dense)
    eig_min = float(eig[0])
    eig_max = float(eig[-1])
    if not (math.isfinite(eig_min) and math.isfinite(eig_max) and eig_min > 0.0):
        return {
            "success": False,
            "breakdown": "non_spd_jacobi_preconditioned_operator",
            "eig_min": eig_min,
            "eig_max": eig_max,
        }
    return {
        "success": True,
        "lambda_min": max(eig_min * 0.95, 1.0e-12),
        "lambda_max": eig_max * 1.05,
        "eig_min": eig_min,
        "eig_max": eig_max,
        "source": "dense_eigvalsh_jacobi_preconditioned_padded",
    }


def _relative_residual(
    csr: CsrMatrix,
    solution: np.ndarray,
    rhs: np.ndarray,
) -> float:
    actual = np.asarray(csr.matvec(tuple(float(value) for value in solution)))
    return float(np.linalg.norm(actual - rhs)) / max(float(np.linalg.norm(rhs)), 1.0e-30)


def _relative_error_to_ones(solution: np.ndarray) -> float:
    return float(np.linalg.norm(solution - 1.0)) / max(
        math.sqrt(float(solution.shape[0])),
        1.0e-30,
    )
