"""Taichi GPU CSR matvec operator.

This is the first real sparse-operator device boundary. It intentionally exposes
CSR vector primitives and iterative solve boundaries for imported CSR fixtures.
"""

from dataclasses import dataclass
import math
import time
from typing import Iterable

import taichi as ti

from transsolvestack.core.result import PolicyPlan, RunTrace, SolveResult
from transsolvestack.core.types import LinearOperatorSpec
from transsolvestack.core.types import SolveContext
from transsolvestack.datasets.csr import CsrMatrix
from transsolvestack.operators.base import LinearOperator


@ti.data_oriented
@dataclass(eq=False)
class TaichiCsrMatrixOperator(LinearOperator):
    spec: LinearOperatorSpec
    n_rows: int
    n_cols: int
    nnz: int
    row_ptr_host: tuple[int, ...]
    col_ind_host: tuple[int, ...]
    values_host: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.row_ptr_host) != self.n_rows + 1:
            raise ValueError("row_ptr length must be n_rows + 1")
        if len(self.col_ind_host) != self.nnz or len(self.values_host) != self.nnz:
            raise ValueError("col_ind and values length must equal nnz")
        if self.row_ptr_host[-1] != self.nnz:
            raise ValueError("row_ptr final value must equal nnz")
        if self.nnz <= 0:
            raise ValueError("Taichi CSR operator requires at least one nonzero")
        float_dtype = _taichi_float_dtype(self.spec.dtype)
        self._numpy_float_dtype = _numpy_float_dtype(self.spec.dtype)
        self.row_ptr = ti.field(dtype=ti.i32, shape=self.n_rows + 1)
        self.col_ind = ti.field(dtype=ti.i32, shape=self.nnz)
        self.values = ti.field(dtype=float_dtype, shape=self.nnz)
        self.x = ti.field(dtype=float_dtype, shape=self.n_cols)
        self.y = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.b = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.r = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.rhs_scaled = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.r_hat = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.z = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.p = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.ap = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.v = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.diag = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.diag_index = ti.field(dtype=ti.i32, shape=self.n_rows)
        self.eq_scale = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.row_scale = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.col_scale = ti.field(dtype=float_dtype, shape=self.n_cols)
        self.row_norm = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.col_norm = ti.field(dtype=float_dtype, shape=self.n_cols)
        self.ilu0_values = ti.field(dtype=float_dtype, shape=self.nnz)
        self.ilu0_work = ti.field(dtype=float_dtype, shape=self.n_rows)
        self.ilu0_valid = ti.field(dtype=ti.i32, shape=())
        self.ilu0_min_abs_pivot = ti.field(dtype=float_dtype, shape=())
        self.scalar = ti.field(dtype=float_dtype, shape=())
        self._load_host_arrays()

    @classmethod
    def from_csr_matrix(
        cls,
        csr: CsrMatrix,
        *,
        dtype: str = "float32",
    ) -> "TaichiCsrMatrixOperator":
        csr.validate()
        spec = LinearOperatorSpec(
            operator_id=f"taichi_csr:{csr.matrix_id}",
            kind="assembled_sparse",
            shape=(csr.n_rows, csr.n_cols),
            dtype=dtype,
            symmetry=csr.symmetry,
            device_resident=True,
            metadata={
                "matrix_id": csr.matrix_id,
                "source_path": csr.source_path,
                "field": csr.field,
                "csr_nnz": csr.nnz,
                "operator_backend": "taichi_csr",
            },
        )
        return cls(
            spec=spec,
            n_rows=csr.n_rows,
            n_cols=csr.n_cols,
            nnz=csr.nnz,
            row_ptr_host=csr.row_ptr,
            col_ind_host=csr.col_ind,
            values_host=csr.values,
        )

    def apply(self, x: Iterable[float] | None = None, y=None):
        if x is not None:
            vector = tuple(float(value) for value in x)
            if len(vector) != self.n_cols:
                raise ValueError(f"expected x length {self.n_cols}, got {len(vector)}")
            self.x.from_numpy(_numpy_array(vector, dtype=self._numpy_float_dtype))
        self._matvec()
        ti.sync()
        result = tuple(float(value) for value in self.y.to_numpy())
        if y is not None:
            if hasattr(y, "__setitem__"):
                for index, value in enumerate(result):
                    y[index] = value
                return y
        return result

    def fill_x(self, value: float) -> None:
        self._fill_x(float(value))
        ti.sync()

    def load_b(self, values: Iterable[float]) -> None:
        vector = tuple(float(value) for value in values)
        if len(vector) != self.n_rows:
            raise ValueError(f"expected b length {self.n_rows}, got {len(vector)}")
        self.b.from_numpy(_numpy_array(vector, dtype=self._numpy_float_dtype))
        ti.sync()

    def residual(
        self,
        x: Iterable[float] | None = None,
        b: Iterable[float] | None = None,
    ) -> tuple[float, ...]:
        """Compute ``r = b - A @ x`` and return the residual vector."""

        if x is not None:
            vector = tuple(float(value) for value in x)
            if len(vector) != self.n_cols:
                raise ValueError(f"expected x length {self.n_cols}, got {len(vector)}")
            self.x.from_numpy(_numpy_array(vector, dtype=self._numpy_float_dtype))
        if b is not None:
            self.load_b(b)
        self._matvec()
        self._residual()
        ti.sync()
        return self.r_to_tuple()

    def dot(self, left: str, right: str) -> float:
        left_field, left_length = self._field_by_name(left)
        right_field, right_length = self._field_by_name(right)
        if left_length != right_length:
            raise ValueError(
                f"cannot dot vectors with different lengths: {left}={left_length}, "
                f"{right}={right_length}"
            )
        self._dot(left_field, right_field, left_length)
        ti.sync()
        return float(self.scalar.to_numpy()[()])

    def norm(self, name: str) -> float:
        return math.sqrt(max(self.dot(name, name), 0.0))

    def solve_cg(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        return self._solve_cg(rhs=rhs, context=context, plan=plan, use_jacobi=False)

    def solve_pcg(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        preconditioner = str(plan.preconditioner.get("name", "jacobi"))
        if preconditioner == "symmetric_equilibration":
            return self._solve_pcg_symmetric_equilibration(
                rhs=rhs,
                context=context,
                plan=plan,
            )
        if preconditioner != "jacobi":
            raise ValueError(f"unsupported Taichi CSR PCG preconditioner: {preconditioner}")
        return self._solve_cg(rhs=rhs, context=context, plan=plan, use_jacobi=True)

    def solve_bicgstab(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        preconditioner = str(plan.preconditioner.get("name", "none"))
        if preconditioner == "row_column_equilibration":
            return self._solve_bicgstab_row_column_equilibration(
                rhs=rhs,
                context=context,
                plan=plan,
            )
        if preconditioner not in {"none", "jacobi", "ilu0"}:
            raise ValueError(f"unsupported Taichi CSR BiCGSTAB preconditioner: {preconditioner}")
        return self._solve_bicgstab(
            rhs=rhs,
            context=context,
            plan=plan,
            preconditioner=preconditioner,
        )

    def solve_gmres(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        preconditioner = str(plan.preconditioner.get("name", "jacobi"))
        if preconditioner == "row_column_equilibration":
            return self._solve_gmres_row_column_equilibration(
                rhs=rhs,
                context=context,
                plan=plan,
            )
        if preconditioner not in {"none", "jacobi"}:
            raise ValueError(f"unsupported Taichi CSR GMRES preconditioner: {preconditioner}")
        return self._solve_gmres(
            rhs=rhs,
            context=context,
            plan=plan,
            use_jacobi=preconditioner == "jacobi",
        )

    def solve_richardson(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        preconditioner = str(plan.preconditioner.get("name", "jacobi"))
        if preconditioner not in {"none", "jacobi"}:
            raise ValueError(f"unsupported Taichi CSR Richardson preconditioner: {preconditioner}")
        return self._solve_richardson(
            rhs=rhs,
            context=context,
            plan=plan,
            use_jacobi=preconditioner == "jacobi",
        )

    def solve_chebyshev(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        preconditioner = str(plan.preconditioner.get("name", "jacobi"))
        if preconditioner != "jacobi":
            raise ValueError("Taichi CSR Chebyshev currently requires jacobi")
        return self._solve_chebyshev(rhs=rhs, context=context, plan=plan)

    def y_to_tuple(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.y.to_numpy())

    def b_to_tuple(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.b.to_numpy())

    def r_to_tuple(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.r.to_numpy())

    def solution_to_tuple(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.x.to_numpy())

    def _solve_cg(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
        use_jacobi: bool,
    ) -> SolveResult:
        if self.n_rows != self.n_cols:
            raise ValueError("Taichi CSR CG requires a square matrix")
        start = time.perf_counter()
        if rhs is not None:
            self.load_b(rhs)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        solve_start = time.perf_counter()
        self._zero_x()
        self._copy(self.b, self.r)
        if use_jacobi:
            self._jacobi(self.r, self.z)
        else:
            self._copy(self.r, self.z)
        self._copy(self.z, self.p)
        rz_old = self._dot_value(self.r, self.z)
        absolute_residual = self._norm_value(self.r)
        residual_history = [absolute_residual / b_norm]
        converged = (
            absolute_residual <= context.tolerance_abs
            or residual_history[-1] <= context.tolerance_rel
        )
        breakdown: str | None = None
        iterations = 0

        if abs(rz_old) < 1.0e-30 and not converged:
            breakdown = "zero_initial_rz"
        for iteration in range(1, context.max_iter + 1):
            if converged or breakdown is not None:
                break
            self._matvec_field(self.p, self.ap)
            denom = self._dot_value(self.p, self.ap)
            if not math.isfinite(denom) or abs(denom) < 1.0e-30:
                breakdown = "zero_or_nonfinite_denominator"
                break
            if denom <= 0.0:
                breakdown = "non_positive_curvature"
                break
            alpha = rz_old / denom
            self._cg_update_x_r(alpha)
            absolute_residual = self._norm_value(self.r)
            relative_residual = absolute_residual / b_norm
            residual_history.append(relative_residual)
            iterations = iteration
            if (
                absolute_residual <= context.tolerance_abs
                or relative_residual <= context.tolerance_rel
            ):
                converged = True
                break
            if use_jacobi:
                self._jacobi(self.r, self.z)
            else:
                self._copy(self.r, self.z)
            rz_new = self._dot_value(self.r, self.z)
            if not math.isfinite(rz_new) or abs(rz_old) < 1.0e-30:
                breakdown = "zero_or_nonfinite_rz"
                break
            beta = rz_new / rz_old
            self._cg_update_p(beta)
            rz_old = rz_new
        ti.sync()

        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        status = "success" if converged else "failed"
        failure_class = None if converged else breakdown or "not_converged"
        final_relative_residual = residual_history[-1]
        solution = self.solution_to_tuple()
        metadata = {
            "solver": plan.solver.get("name"),
            "preconditioner": plan.preconditioner.get("name", "jacobi" if use_jacobi else "none"),
            "matrix_id": self.spec.metadata.get("matrix_id"),
            "operator_backend": "taichi_csr",
            "precision": self.spec.dtype,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "csr_nnz": self.nnz,
            "field": self.spec.metadata.get("field"),
            "symmetry": self.spec.symmetry,
            "absolute_residual_norm": absolute_residual,
            "relative_residual_norm": final_relative_residual,
            "b_norm": b_norm,
        }
        trace = RunTrace(
            run_id=f"run:{plan.plan_id}",
            plan_id=plan.plan_id,
            backend=plan.backend,
            status=status,
            failure_class=failure_class,
            setup_time_ms=0.0,
            solve_time_ms=solve_ms,
            total_time_ms=total_ms,
            gpu_kernel_time_ms=solve_ms,
            transfer_time_ms=0.0,
            num_iterations=iterations,
            final_residual_norm=final_relative_residual,
            residual_history=residual_history,
            metadata=metadata,
        )
        return SolveResult(status=status, solution=solution, trace=trace, metadata=metadata)

    def _solve_pcg_symmetric_equilibration(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        if self.n_rows != self.n_cols:
            raise ValueError("Taichi CSR symmetric equilibration requires a square matrix")
        if self.spec.symmetry != "symmetric":
            raise ValueError("Taichi CSR symmetric equilibration requires symmetric metadata")
        diag_host = self.diag.to_numpy()
        min_diag = float(diag_host.min())
        if min_diag <= 1.0e-30:
            raise ValueError("Taichi CSR symmetric equilibration requires positive diagonal")

        start = time.perf_counter()
        if rhs is not None:
            self.load_b(rhs)
        original_b_norm = max(self._norm_value(self.b), 1.0e-30)
        self._build_symmetric_equilibration_scale()
        solve_start = time.perf_counter()
        self._zero_x()
        self._scale_b_to_r()
        scaled_b_norm = max(self._norm_value(self.r), 1.0e-30)
        self._copy(self.r, self.z)
        self._copy(self.z, self.p)
        rz_old = self._dot_value(self.r, self.z)
        scaled_abs_residual = self._norm_value(self.r)
        residual_history = [scaled_abs_residual / scaled_b_norm]
        converged = (
            scaled_abs_residual <= context.tolerance_abs
            or residual_history[-1] <= context.tolerance_rel
        )
        breakdown: str | None = None
        iterations = 0

        if abs(rz_old) < 1.0e-30 and not converged:
            breakdown = "zero_initial_rz"
        for iteration in range(1, context.max_iter + 1):
            if converged or breakdown is not None:
                break
            self._matvec_equilibrated_field(self.p, self.ap)
            denom = self._dot_value(self.p, self.ap)
            if not math.isfinite(denom) or abs(denom) < 1.0e-30:
                breakdown = "zero_or_nonfinite_denominator"
                break
            if denom <= 0.0:
                breakdown = "non_positive_curvature"
                break
            alpha = rz_old / denom
            self._cg_update_x_r(alpha)
            scaled_abs_residual = self._norm_value(self.r)
            scaled_relative_residual = scaled_abs_residual / scaled_b_norm
            residual_history.append(scaled_relative_residual)
            iterations = iteration
            if (
                scaled_abs_residual <= context.tolerance_abs
                or scaled_relative_residual <= context.tolerance_rel
            ):
                converged = True
                break
            self._copy(self.r, self.z)
            rz_new = self._dot_value(self.r, self.z)
            if not math.isfinite(rz_new) or abs(rz_old) < 1.0e-30:
                breakdown = "zero_or_nonfinite_rz"
                break
            beta = rz_new / rz_old
            self._cg_update_p(beta)
            rz_old = rz_new
        self._equilibrated_solution_to_original()
        self._matvec()
        self._residual()
        original_abs_residual = self._norm_value(self.r)
        original_relative_residual = original_abs_residual / original_b_norm
        ti.sync()

        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        original_residual_passed = original_relative_residual <= context.tolerance_rel
        status = "success" if converged and original_residual_passed else "failed"
        if not converged:
            failure_class = breakdown or "not_converged"
        elif not original_residual_passed:
            failure_class = "original_residual_above_tolerance"
        else:
            failure_class = None
        solution = self.solution_to_tuple()
        metadata = {
            "solver": plan.solver.get("name"),
            "preconditioner": plan.preconditioner.get("name"),
            "matrix_id": self.spec.metadata.get("matrix_id"),
            "operator_backend": "taichi_csr",
            "precision": self.spec.dtype,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "csr_nnz": self.nnz,
            "field": self.spec.metadata.get("field"),
            "symmetry": self.spec.symmetry,
            "absolute_residual_norm": original_abs_residual,
            "relative_residual_norm": original_relative_residual,
            "scaled_relative_residual_norm": residual_history[-1],
            "b_norm": original_b_norm,
            "scaled_b_norm": scaled_b_norm,
            "min_diagonal": min_diag,
            "equilibration": "symmetric_diagonal",
        }
        trace = RunTrace(
            run_id=f"run:{plan.plan_id}",
            plan_id=plan.plan_id,
            backend=plan.backend,
            status=status,
            failure_class=failure_class,
            setup_time_ms=0.0,
            solve_time_ms=solve_ms,
            total_time_ms=total_ms,
            gpu_kernel_time_ms=solve_ms,
            transfer_time_ms=0.0,
            num_iterations=iterations,
            final_residual_norm=original_relative_residual,
            residual_history=residual_history,
            metadata=metadata,
        )
        return SolveResult(status=status, solution=solution, trace=trace, metadata=metadata)

    def _solve_bicgstab(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
        preconditioner: str,
    ) -> SolveResult:
        if self.n_rows != self.n_cols:
            raise ValueError("Taichi CSR BiCGSTAB requires a square matrix")
        start = time.perf_counter()
        if rhs is not None:
            self.load_b(rhs)
        ilu0_pivot_tolerance = 0.0
        ilu0_min_abs_pivot: float | None = None
        if preconditioner == "ilu0":
            ilu0_pivot_tolerance = _ilu0_pivot_tolerance(plan)
            self._build_ilu0(ilu0_pivot_tolerance)
            if not self._ilu0_is_valid():
                raise ValueError("Taichi CSR ILU0 setup failed: missing or tiny pivot")
            ilu0_min_abs_pivot = float(self.ilu0_min_abs_pivot.to_numpy()[()])
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        solve_start = time.perf_counter()
        self._zero_x()
        self._copy(self.b, self.r)
        self._zero_p()
        self._zero_v()
        absolute_residual = self._norm_value(self.r)
        residual_history = [absolute_residual / b_norm]
        converged = (
            absolute_residual <= context.tolerance_abs
            or residual_history[-1] <= context.tolerance_rel
        )
        breakdown: str | None = None
        rho_old = 1.0
        alpha = 1.0
        omega = 1.0
        iterations = 0

        for iteration in range(1, context.max_iter + 1):
            if converged or breakdown is not None:
                break
            rho_new = self._dot_value(self.b, self.r)
            if not math.isfinite(rho_new) or abs(rho_new) < 1.0e-30:
                breakdown = "zero_or_nonfinite_rho"
                break
            if iteration == 1:
                self._copy(self.r, self.p)
            else:
                if abs(omega) < 1.0e-30:
                    breakdown = "zero_omega"
                    break
                beta = (rho_new / rho_old) * (alpha / omega)
                if not math.isfinite(beta):
                    breakdown = "nonfinite_beta"
                    break
                self._bicgstab_update_p(beta, omega)
            self._apply_bicgstab_preconditioner(
                preconditioner,
                self.p,
                self.z,
                ilu0_pivot_tolerance,
            )
            self._matvec_field(self.z, self.v)
            denom = self._dot_value(self.b, self.v)
            if not math.isfinite(denom) or abs(denom) < 1.0e-30:
                breakdown = "zero_or_nonfinite_alpha_denominator"
                break
            alpha = rho_new / denom
            if not math.isfinite(alpha):
                breakdown = "nonfinite_alpha"
                break
            self._bicgstab_update_x_s(alpha)
            absolute_s = self._norm_value(self.y)
            relative_s = absolute_s / b_norm
            if absolute_s <= context.tolerance_abs or relative_s <= context.tolerance_rel:
                self._copy(self.y, self.r)
                absolute_residual = absolute_s
                residual_history.append(relative_s)
                iterations = iteration
                converged = True
                break
            self._apply_bicgstab_preconditioner(
                preconditioner,
                self.y,
                self.z,
                ilu0_pivot_tolerance,
            )
            self._matvec_field(self.z, self.ap)
            tt = self._dot_value(self.ap, self.ap)
            if not math.isfinite(tt) or abs(tt) < 1.0e-30:
                breakdown = "zero_or_nonfinite_omega_denominator"
                break
            ts = self._dot_value(self.ap, self.y)
            omega = ts / tt
            if not math.isfinite(omega) or abs(omega) < 1.0e-30:
                breakdown = "zero_or_nonfinite_omega"
                break
            self._bicgstab_update_x_r(omega)
            absolute_residual = self._norm_value(self.r)
            relative_residual = absolute_residual / b_norm
            residual_history.append(relative_residual)
            iterations = iteration
            if (
                absolute_residual <= context.tolerance_abs
                or relative_residual <= context.tolerance_rel
            ):
                converged = True
                break
            rho_old = rho_new
        ti.sync()

        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        status = "success" if converged else "failed"
        failure_class = None if converged else breakdown or "not_converged"
        final_relative_residual = residual_history[-1]
        solution = self.solution_to_tuple()
        metadata = {
            "solver": plan.solver.get("name"),
            "preconditioner": plan.preconditioner.get("name", preconditioner),
            "matrix_id": self.spec.metadata.get("matrix_id"),
            "operator_backend": "taichi_csr",
            "precision": self.spec.dtype,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "csr_nnz": self.nnz,
            "field": self.spec.metadata.get("field"),
            "symmetry": self.spec.symmetry,
            "absolute_residual_norm": absolute_residual,
            "relative_residual_norm": final_relative_residual,
            "b_norm": b_norm,
        }
        if preconditioner == "ilu0":
            metadata.update(
                {
                    "ilu0_factorization": "doolittle_level_zero_csr_pattern",
                    "ilu0_apply": "taichi_gpu_serial_triangular_solve",
                    "ilu0_pivot_tolerance": ilu0_pivot_tolerance,
                    "ilu0_min_abs_pivot": ilu0_min_abs_pivot,
                    "ilu0_valid": True,
                }
            )
        trace = RunTrace(
            run_id=f"run:{plan.plan_id}",
            plan_id=plan.plan_id,
            backend=plan.backend,
            status=status,
            failure_class=failure_class,
            setup_time_ms=0.0,
            solve_time_ms=solve_ms,
            total_time_ms=total_ms,
            gpu_kernel_time_ms=solve_ms,
            transfer_time_ms=0.0,
            num_iterations=iterations,
            final_residual_norm=final_relative_residual,
            residual_history=residual_history,
            metadata=metadata,
        )
        return SolveResult(status=status, solution=solution, trace=trace, metadata=metadata)

    def _solve_bicgstab_row_column_equilibration(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        if self.n_rows != self.n_cols:
            raise ValueError("Taichi CSR row/column equilibration requires a square matrix")
        passes = _row_column_equilibration_passes(plan)
        start = time.perf_counter()
        if rhs is not None:
            self.load_b(rhs)
        original_b_norm = max(self._norm_value(self.b), 1.0e-30)
        self._build_row_column_equilibration_scale(passes)
        solve_start = time.perf_counter()
        self._zero_x()
        self._scale_b_to_rhs_scaled()
        self._copy(self.rhs_scaled, self.r)
        self._copy(self.r, self.r_hat)
        self._zero_p()
        self._zero_v()
        scaled_b_norm = max(self._norm_value(self.rhs_scaled), 1.0e-30)
        scaled_abs_residual = self._norm_value(self.r)
        residual_history = [scaled_abs_residual / scaled_b_norm]
        converged = (
            scaled_abs_residual <= context.tolerance_abs
            or residual_history[-1] <= context.tolerance_rel
        )
        breakdown: str | None = None
        rho_old = 1.0
        alpha = 1.0
        omega = 1.0
        iterations = 0

        for iteration in range(1, context.max_iter + 1):
            if converged or breakdown is not None:
                break
            rho_new = self._dot_value(self.r_hat, self.r)
            if not math.isfinite(rho_new) or abs(rho_new) < 1.0e-30:
                breakdown = "zero_or_nonfinite_rho"
                break
            if iteration == 1:
                self._copy(self.r, self.p)
            else:
                if abs(omega) < 1.0e-30:
                    breakdown = "zero_omega"
                    break
                beta = (rho_new / rho_old) * (alpha / omega)
                if not math.isfinite(beta):
                    breakdown = "nonfinite_beta"
                    break
                self._bicgstab_update_p(beta, omega)
            self._copy(self.p, self.z)
            self._matvec_row_column_equilibrated_field(self.z, self.v)
            denom = self._dot_value(self.r_hat, self.v)
            if not math.isfinite(denom) or abs(denom) < 1.0e-30:
                breakdown = "zero_or_nonfinite_alpha_denominator"
                break
            alpha = rho_new / denom
            if not math.isfinite(alpha):
                breakdown = "nonfinite_alpha"
                break
            self._bicgstab_update_x_s(alpha)
            absolute_s = self._norm_value(self.y)
            relative_s = absolute_s / scaled_b_norm
            if absolute_s <= context.tolerance_abs or relative_s <= context.tolerance_rel:
                self._copy(self.y, self.r)
                scaled_abs_residual = absolute_s
                residual_history.append(relative_s)
                iterations = iteration
                converged = True
                break
            self._copy(self.y, self.z)
            self._matvec_row_column_equilibrated_field(self.z, self.ap)
            tt = self._dot_value(self.ap, self.ap)
            if not math.isfinite(tt) or abs(tt) < 1.0e-30:
                breakdown = "zero_or_nonfinite_omega_denominator"
                break
            ts = self._dot_value(self.ap, self.y)
            omega = ts / tt
            if not math.isfinite(omega) or abs(omega) < 1.0e-30:
                breakdown = "zero_or_nonfinite_omega"
                break
            self._bicgstab_update_x_r(omega)
            scaled_abs_residual = self._norm_value(self.r)
            scaled_relative_residual = scaled_abs_residual / scaled_b_norm
            residual_history.append(scaled_relative_residual)
            iterations = iteration
            if (
                scaled_abs_residual <= context.tolerance_abs
                or scaled_relative_residual <= context.tolerance_rel
            ):
                converged = True
                break
            rho_old = rho_new
        self._row_column_equilibrated_solution_to_original()
        self._matvec()
        self._residual()
        original_abs_residual = self._norm_value(self.r)
        original_relative_residual = original_abs_residual / original_b_norm
        ti.sync()

        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        original_residual_passed = original_relative_residual <= context.tolerance_rel
        status = "success" if converged and original_residual_passed else "failed"
        if not converged:
            failure_class = breakdown or "not_converged"
        elif not original_residual_passed:
            failure_class = "original_residual_above_tolerance"
        else:
            failure_class = None
        solution = self.solution_to_tuple()
        row_scale_stats = _scale_stats(self.row_scale.to_numpy())
        col_scale_stats = _scale_stats(self.col_scale.to_numpy())
        metadata = {
            "solver": plan.solver.get("name"),
            "preconditioner": plan.preconditioner.get("name"),
            "matrix_id": self.spec.metadata.get("matrix_id"),
            "operator_backend": "taichi_csr",
            "precision": self.spec.dtype,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "csr_nnz": self.nnz,
            "field": self.spec.metadata.get("field"),
            "symmetry": self.spec.symmetry,
            "absolute_residual_norm": original_abs_residual,
            "relative_residual_norm": original_relative_residual,
            "scaled_relative_residual_norm": residual_history[-1],
            "b_norm": original_b_norm,
            "scaled_b_norm": scaled_b_norm,
            "equilibration": "row_column_l1",
            "row_column_equilibration_passes": passes,
            "row_scale_min": row_scale_stats[0],
            "row_scale_max": row_scale_stats[1],
            "col_scale_min": col_scale_stats[0],
            "col_scale_max": col_scale_stats[1],
        }
        trace = RunTrace(
            run_id=f"run:{plan.plan_id}",
            plan_id=plan.plan_id,
            backend=plan.backend,
            status=status,
            failure_class=failure_class,
            setup_time_ms=0.0,
            solve_time_ms=solve_ms,
            total_time_ms=total_ms,
            gpu_kernel_time_ms=solve_ms,
            transfer_time_ms=0.0,
            num_iterations=iterations,
            final_residual_norm=original_relative_residual,
            residual_history=residual_history,
            metadata=metadata,
        )
        return SolveResult(status=status, solution=solution, trace=trace, metadata=metadata)

    def _solve_gmres(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
        use_jacobi: bool,
    ) -> SolveResult:
        if self.n_rows != self.n_cols:
            raise ValueError("Taichi CSR GMRES requires a square matrix")
        import numpy as np

        restart_requested = int(plan.solver.get("restart", min(16, self.n_rows)))
        if restart_requested <= 0:
            raise ValueError("gmres restart must be positive")
        restart = min(restart_requested, self.n_rows, max(context.max_iter, 1))
        start = time.perf_counter()
        if rhs is not None:
            self.load_b(rhs)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        solve_start = time.perf_counter()
        transfer_time_ms = 0.0
        host_control_time_ms = 0.0

        def load_x_np(values) -> None:
            nonlocal transfer_time_ms
            transfer_start = time.perf_counter()
            self.x.from_numpy(np.asarray(values, dtype=self._numpy_float_dtype))
            ti.sync()
            transfer_time_ms += (time.perf_counter() - transfer_start) * 1000.0

        def load_p_np(values) -> None:
            nonlocal transfer_time_ms
            transfer_start = time.perf_counter()
            self.p.from_numpy(np.asarray(values, dtype=self._numpy_float_dtype))
            ti.sync()
            transfer_time_ms += (time.perf_counter() - transfer_start) * 1000.0

        def z_to_numpy():
            nonlocal transfer_time_ms
            transfer_start = time.perf_counter()
            values = self.z.to_numpy().astype(np.float64, copy=True)
            transfer_time_ms += (time.perf_counter() - transfer_start) * 1000.0
            return values

        self._zero_x()
        self._copy(self.b, self.r)
        absolute_residual = self._norm_value(self.r)
        relative_residual = absolute_residual / b_norm
        residual_history = [relative_residual]
        converged = (
            absolute_residual <= context.tolerance_abs
            or relative_residual <= context.tolerance_rel
        )
        breakdown: str | None = None
        iterations = 0
        restart_cycles = 0

        while not converged and breakdown is None and iterations < context.max_iter:
            restart_cycles += 1
            if use_jacobi:
                self._jacobi(self.r, self.z)
            else:
                self._copy(self.r, self.z)
            beta = self._norm_value(self.z)
            if not math.isfinite(beta) or beta <= 1.0e-30:
                breakdown = "zero_or_nonfinite_preconditioned_residual"
                break
            self._scale_z(1.0 / beta)
            basis = [z_to_numpy()]
            cycle_base = self.x.to_numpy().astype(np.float64, copy=True)
            cycle_candidate = cycle_base.copy()
            hessenberg = np.zeros((restart + 1, restart), dtype=np.float64)
            inner_limit = min(restart, context.max_iter - iterations)

            for inner in range(inner_limit):
                load_p_np(basis[inner])
                self._matvec_field(self.p, self.ap)
                if use_jacobi:
                    self._jacobi(self.ap, self.z)
                else:
                    self._copy(self.ap, self.z)
                for basis_index in range(inner + 1):
                    load_p_np(basis[basis_index])
                    h_value = self._dot_value(self.p, self.z)
                    hessenberg[basis_index, inner] = h_value
                    self._axpy_z_from_p(-h_value)
                next_norm = self._norm_value(self.z)
                hessenberg[inner + 1, inner] = next_norm

                host_start = time.perf_counter()
                target = np.zeros(inner + 2, dtype=np.float64)
                target[0] = beta
                try:
                    coefficients = np.linalg.lstsq(
                        hessenberg[: inner + 2, : inner + 1],
                        target,
                        rcond=None,
                    )[0]
                except np.linalg.LinAlgError:
                    host_control_time_ms += (time.perf_counter() - host_start) * 1000.0
                    breakdown = "least_squares_failed"
                    break
                cycle_candidate = cycle_base.copy()
                for coefficient, vector in zip(coefficients, basis):
                    cycle_candidate += float(coefficient) * vector
                host_control_time_ms += (time.perf_counter() - host_start) * 1000.0

                load_x_np(cycle_candidate)
                self._matvec()
                self._residual()
                absolute_residual = self._norm_value(self.r)
                relative_residual = absolute_residual / b_norm
                if not math.isfinite(relative_residual):
                    breakdown = "nonfinite_residual"
                    break
                residual_history.append(relative_residual)
                iterations += 1
                if (
                    absolute_residual <= context.tolerance_abs
                    or relative_residual <= context.tolerance_rel
                ):
                    converged = True
                    break
                if not math.isfinite(next_norm) or next_norm <= 1.0e-30:
                    breakdown = "arnoldi_breakdown_without_convergence"
                    break
                self._scale_z(1.0 / next_norm)
                basis.append(z_to_numpy())

            if breakdown is not None or converged:
                break
            load_x_np(cycle_candidate)
            self._matvec()
            self._residual()
            absolute_residual = self._norm_value(self.r)
            relative_residual = absolute_residual / b_norm
            if not math.isfinite(relative_residual):
                breakdown = "nonfinite_residual"
                break
        ti.sync()

        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        status = "success" if converged else "failed"
        failure_class = None if converged else breakdown or "not_converged"
        final_relative_residual = residual_history[-1]
        solution = self.solution_to_tuple()
        metadata = {
            "solver": plan.solver.get("name"),
            "preconditioner": plan.preconditioner.get(
                "name", "jacobi" if use_jacobi else "none"
            ),
            "matrix_id": self.spec.metadata.get("matrix_id"),
            "operator_backend": "taichi_csr",
            "precision": self.spec.dtype,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "csr_nnz": self.nnz,
            "field": self.spec.metadata.get("field"),
            "symmetry": self.spec.symmetry,
            "absolute_residual_norm": absolute_residual,
            "relative_residual_norm": final_relative_residual,
            "b_norm": b_norm,
            "restart": restart,
            "restart_requested": restart_requested,
            "restart_cycles": restart_cycles,
            "orthogonalization_backend": "taichi_gpu_modified_gram_schmidt",
            "basis_storage": "host_vectors",
            "least_squares_backend": "numpy_lstsq_dense_hessenberg",
            "host_control_time_ms": host_control_time_ms,
        }
        trace = RunTrace(
            run_id=f"run:{plan.plan_id}",
            plan_id=plan.plan_id,
            backend=plan.backend,
            status=status,
            failure_class=failure_class,
            setup_time_ms=0.0,
            solve_time_ms=solve_ms,
            total_time_ms=total_ms,
            gpu_kernel_time_ms=max(solve_ms - transfer_time_ms - host_control_time_ms, 0.0),
            transfer_time_ms=transfer_time_ms,
            num_iterations=iterations,
            final_residual_norm=final_relative_residual,
            residual_history=residual_history,
            metadata=metadata,
        )
        return SolveResult(status=status, solution=solution, trace=trace, metadata=metadata)

    def _solve_gmres_row_column_equilibration(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        if self.n_rows != self.n_cols:
            raise ValueError("Taichi CSR GMRES row/column equilibration requires a square matrix")
        import numpy as np

        restart_requested = int(plan.solver.get("restart", min(16, self.n_rows)))
        if restart_requested <= 0:
            raise ValueError("gmres restart must be positive")
        restart = min(restart_requested, self.n_rows, max(context.max_iter, 1))
        passes = _row_column_equilibration_passes(plan)
        start = time.perf_counter()
        if rhs is not None:
            self.load_b(rhs)
        original_b_norm = max(self._norm_value(self.b), 1.0e-30)
        self._build_row_column_equilibration_scale(passes)
        self._scale_b_to_rhs_scaled()
        scaled_b_norm = max(self._norm_value(self.rhs_scaled), 1.0e-30)
        solve_start = time.perf_counter()
        transfer_time_ms = 0.0
        host_control_time_ms = 0.0

        def load_x_np(values) -> None:
            nonlocal transfer_time_ms
            transfer_start = time.perf_counter()
            self.x.from_numpy(np.asarray(values, dtype=self._numpy_float_dtype))
            ti.sync()
            transfer_time_ms += (time.perf_counter() - transfer_start) * 1000.0

        def load_p_np(values) -> None:
            nonlocal transfer_time_ms
            transfer_start = time.perf_counter()
            self.p.from_numpy(np.asarray(values, dtype=self._numpy_float_dtype))
            ti.sync()
            transfer_time_ms += (time.perf_counter() - transfer_start) * 1000.0

        def z_to_numpy():
            nonlocal transfer_time_ms
            transfer_start = time.perf_counter()
            values = self.z.to_numpy().astype(np.float64, copy=True)
            transfer_time_ms += (time.perf_counter() - transfer_start) * 1000.0
            return values

        self._zero_x()
        self._copy(self.rhs_scaled, self.r)
        scaled_abs_residual = self._norm_value(self.r)
        scaled_relative_residual = scaled_abs_residual / scaled_b_norm
        residual_history = [scaled_relative_residual]
        converged = (
            scaled_abs_residual <= context.tolerance_abs
            or scaled_relative_residual <= context.tolerance_rel
        )
        breakdown: str | None = None
        iterations = 0
        restart_cycles = 0

        while not converged and breakdown is None and iterations < context.max_iter:
            restart_cycles += 1
            self._copy(self.r, self.z)
            beta = self._norm_value(self.z)
            if not math.isfinite(beta) or beta <= 1.0e-30:
                breakdown = "zero_or_nonfinite_scaled_residual"
                break
            self._scale_z(1.0 / beta)
            basis = [z_to_numpy()]
            cycle_base = self.x.to_numpy().astype(np.float64, copy=True)
            cycle_candidate = cycle_base.copy()
            hessenberg = np.zeros((restart + 1, restart), dtype=np.float64)
            inner_limit = min(restart, context.max_iter - iterations)

            for inner in range(inner_limit):
                load_p_np(basis[inner])
                self._matvec_row_column_equilibrated_field(self.p, self.z)
                for basis_index in range(inner + 1):
                    load_p_np(basis[basis_index])
                    h_value = self._dot_value(self.p, self.z)
                    hessenberg[basis_index, inner] = h_value
                    self._axpy_z_from_p(-h_value)
                next_norm = self._norm_value(self.z)
                hessenberg[inner + 1, inner] = next_norm

                host_start = time.perf_counter()
                target = np.zeros(inner + 2, dtype=np.float64)
                target[0] = beta
                try:
                    coefficients = np.linalg.lstsq(
                        hessenberg[: inner + 2, : inner + 1],
                        target,
                        rcond=None,
                    )[0]
                except np.linalg.LinAlgError:
                    host_control_time_ms += (time.perf_counter() - host_start) * 1000.0
                    breakdown = "least_squares_failed"
                    break
                cycle_candidate = cycle_base.copy()
                for coefficient, vector in zip(coefficients, basis):
                    cycle_candidate += float(coefficient) * vector
                host_control_time_ms += (time.perf_counter() - host_start) * 1000.0

                load_x_np(cycle_candidate)
                self._matvec_row_column_equilibrated_field(self.x, self.y)
                self._row_column_scaled_residual()
                scaled_abs_residual = self._norm_value(self.r)
                scaled_relative_residual = scaled_abs_residual / scaled_b_norm
                if not math.isfinite(scaled_relative_residual):
                    breakdown = "nonfinite_scaled_residual"
                    break
                residual_history.append(scaled_relative_residual)
                iterations += 1
                if (
                    scaled_abs_residual <= context.tolerance_abs
                    or scaled_relative_residual <= context.tolerance_rel
                ):
                    converged = True
                    break
                if not math.isfinite(next_norm) or next_norm <= 1.0e-30:
                    breakdown = "arnoldi_breakdown_without_convergence"
                    break
                self._scale_z(1.0 / next_norm)
                basis.append(z_to_numpy())

            if breakdown is not None or converged:
                break
            load_x_np(cycle_candidate)
            self._matvec_row_column_equilibrated_field(self.x, self.y)
            self._row_column_scaled_residual()
            scaled_abs_residual = self._norm_value(self.r)
            scaled_relative_residual = scaled_abs_residual / scaled_b_norm
            if not math.isfinite(scaled_relative_residual):
                breakdown = "nonfinite_scaled_residual"
                break
        self._row_column_equilibrated_solution_to_original()
        self._matvec()
        self._residual()
        original_abs_residual = self._norm_value(self.r)
        original_relative_residual = original_abs_residual / original_b_norm
        ti.sync()

        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        original_residual_passed = original_relative_residual <= context.tolerance_rel
        status = "success" if converged and original_residual_passed else "failed"
        if not converged:
            failure_class = breakdown or "not_converged"
        elif not original_residual_passed:
            failure_class = "original_residual_above_tolerance"
        else:
            failure_class = None
        solution = self.solution_to_tuple()
        row_scale_stats = _scale_stats(self.row_scale.to_numpy())
        col_scale_stats = _scale_stats(self.col_scale.to_numpy())
        metadata = {
            "solver": plan.solver.get("name"),
            "preconditioner": plan.preconditioner.get("name"),
            "matrix_id": self.spec.metadata.get("matrix_id"),
            "operator_backend": "taichi_csr",
            "precision": self.spec.dtype,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "csr_nnz": self.nnz,
            "field": self.spec.metadata.get("field"),
            "symmetry": self.spec.symmetry,
            "absolute_residual_norm": original_abs_residual,
            "relative_residual_norm": original_relative_residual,
            "scaled_relative_residual_norm": residual_history[-1],
            "b_norm": original_b_norm,
            "scaled_b_norm": scaled_b_norm,
            "restart": restart,
            "restart_requested": restart_requested,
            "restart_cycles": restart_cycles,
            "orthogonalization_backend": "taichi_gpu_modified_gram_schmidt",
            "basis_storage": "host_vectors",
            "least_squares_backend": "numpy_lstsq_dense_hessenberg",
            "host_control_time_ms": host_control_time_ms,
            "equilibration": "row_column_l1",
            "row_column_equilibration_passes": passes,
            "row_scale_min": row_scale_stats[0],
            "row_scale_max": row_scale_stats[1],
            "col_scale_min": col_scale_stats[0],
            "col_scale_max": col_scale_stats[1],
        }
        trace = RunTrace(
            run_id=f"run:{plan.plan_id}",
            plan_id=plan.plan_id,
            backend=plan.backend,
            status=status,
            failure_class=failure_class,
            setup_time_ms=0.0,
            solve_time_ms=solve_ms,
            total_time_ms=total_ms,
            gpu_kernel_time_ms=max(solve_ms - transfer_time_ms - host_control_time_ms, 0.0),
            transfer_time_ms=transfer_time_ms,
            num_iterations=iterations,
            final_residual_norm=original_relative_residual,
            residual_history=residual_history,
            metadata=metadata,
        )
        return SolveResult(status=status, solution=solution, trace=trace, metadata=metadata)

    def _solve_richardson(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
        use_jacobi: bool,
    ) -> SolveResult:
        if self.n_rows != self.n_cols:
            raise ValueError("Taichi CSR Richardson requires a square matrix")
        omega = float(plan.solver.get("omega", 2.0 / 3.0))
        if not 0.0 < omega < 1.0:
            raise ValueError("richardson omega must be in (0, 1)")
        start = time.perf_counter()
        if rhs is not None:
            self.load_b(rhs)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        solve_start = time.perf_counter()
        self._zero_x()
        self._copy(self.b, self.r)
        absolute_residual = self._norm_value(self.r)
        residual_history = [absolute_residual / b_norm]
        converged = (
            absolute_residual <= context.tolerance_abs
            or residual_history[-1] <= context.tolerance_rel
        )
        breakdown: str | None = None
        iterations = 0

        for iteration in range(1, context.max_iter + 1):
            if converged or breakdown is not None:
                break
            if use_jacobi:
                self._jacobi(self.r, self.z)
            else:
                self._copy(self.r, self.z)
            self._matvec_field(self.z, self.ap)
            self._richardson_update_x_r(omega)
            absolute_residual = self._norm_value(self.r)
            relative_residual = absolute_residual / b_norm
            if not math.isfinite(relative_residual):
                breakdown = "nonfinite_residual"
                break
            residual_history.append(relative_residual)
            iterations = iteration
            if (
                absolute_residual <= context.tolerance_abs
                or relative_residual <= context.tolerance_rel
            ):
                converged = True
                break
        ti.sync()

        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        status = "success" if converged else "failed"
        failure_class = None if converged else breakdown or "not_converged"
        final_relative_residual = residual_history[-1]
        solution = self.solution_to_tuple()
        metadata = {
            "solver": plan.solver.get("name"),
            "preconditioner": plan.preconditioner.get("name", "jacobi" if use_jacobi else "none"),
            "matrix_id": self.spec.metadata.get("matrix_id"),
            "operator_backend": "taichi_csr",
            "precision": self.spec.dtype,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "csr_nnz": self.nnz,
            "field": self.spec.metadata.get("field"),
            "symmetry": self.spec.symmetry,
            "absolute_residual_norm": absolute_residual,
            "relative_residual_norm": final_relative_residual,
            "b_norm": b_norm,
            "omega": omega,
        }
        trace = RunTrace(
            run_id=f"run:{plan.plan_id}",
            plan_id=plan.plan_id,
            backend=plan.backend,
            status=status,
            failure_class=failure_class,
            setup_time_ms=0.0,
            solve_time_ms=solve_ms,
            total_time_ms=total_ms,
            gpu_kernel_time_ms=solve_ms,
            transfer_time_ms=0.0,
            num_iterations=iterations,
            final_residual_norm=final_relative_residual,
            residual_history=residual_history,
            metadata=metadata,
        )
        return SolveResult(status=status, solution=solution, trace=trace, metadata=metadata)

    def _solve_chebyshev(
        self,
        rhs,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        if self.n_rows != self.n_cols:
            raise ValueError("Taichi CSR Chebyshev requires a square matrix")
        lower, upper = _chebyshev_spectral_bounds(plan)
        d = 0.5 * (upper + lower)
        c = 0.5 * (upper - lower)
        start = time.perf_counter()
        if rhs is not None:
            self.load_b(rhs)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        solve_start = time.perf_counter()
        self._zero_x()
        self._copy(self.b, self.r)
        absolute_residual = self._norm_value(self.r)
        residual_history = [absolute_residual / b_norm]
        converged = (
            absolute_residual <= context.tolerance_abs
            or residual_history[-1] <= context.tolerance_rel
        )
        breakdown: str | None = None
        iterations = 0
        alpha = 0.0

        for iteration in range(1, context.max_iter + 1):
            if converged or breakdown is not None:
                break
            self._jacobi(self.r, self.z)
            if iteration == 1:
                self._copy(self.z, self.p)
                alpha = 1.0 / d
            else:
                if abs(alpha) < 1.0e-30:
                    breakdown = "zero_alpha"
                    break
                if iteration == 2:
                    beta = 0.5 * (c * alpha) ** 2
                else:
                    beta = (0.5 * c * alpha) ** 2
                denom = d - beta / alpha
                if not math.isfinite(denom) or abs(denom) < 1.0e-30:
                    breakdown = "zero_or_nonfinite_alpha_denominator"
                    break
                alpha = 1.0 / denom
                if not math.isfinite(alpha):
                    breakdown = "nonfinite_alpha"
                    break
                self._cg_update_p(beta)
            self._matvec_field(self.p, self.ap)
            self._cg_update_x_r(alpha)
            absolute_residual = self._norm_value(self.r)
            relative_residual = absolute_residual / b_norm
            if not math.isfinite(relative_residual):
                breakdown = "nonfinite_residual"
                break
            residual_history.append(relative_residual)
            iterations = iteration
            if (
                absolute_residual <= context.tolerance_abs
                or relative_residual <= context.tolerance_rel
            ):
                converged = True
                break
        ti.sync()

        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        status = "success" if converged else "failed"
        failure_class = None if converged else breakdown or "not_converged"
        final_relative_residual = residual_history[-1]
        solution = self.solution_to_tuple()
        metadata = {
            "solver": plan.solver.get("name"),
            "preconditioner": plan.preconditioner.get("name", "jacobi"),
            "matrix_id": self.spec.metadata.get("matrix_id"),
            "operator_backend": "taichi_csr",
            "precision": self.spec.dtype,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "csr_nnz": self.nnz,
            "field": self.spec.metadata.get("field"),
            "symmetry": self.spec.symmetry,
            "absolute_residual_norm": absolute_residual,
            "relative_residual_norm": final_relative_residual,
            "b_norm": b_norm,
            "lambda_min": lower,
            "lambda_max": upper,
            "spectral_bounds": "explicit_jacobi_preconditioned",
        }
        trace = RunTrace(
            run_id=f"run:{plan.plan_id}",
            plan_id=plan.plan_id,
            backend=plan.backend,
            status=status,
            failure_class=failure_class,
            setup_time_ms=0.0,
            solve_time_ms=solve_ms,
            total_time_ms=total_ms,
            gpu_kernel_time_ms=solve_ms,
            transfer_time_ms=0.0,
            num_iterations=iterations,
            final_residual_norm=final_relative_residual,
            residual_history=residual_history,
            metadata=metadata,
        )
        return SolveResult(status=status, solution=solution, trace=trace, metadata=metadata)

    def _load_host_arrays(self) -> None:
        self.row_ptr.from_numpy(_numpy_array(self.row_ptr_host, dtype="int32"))
        self.col_ind.from_numpy(_numpy_array(self.col_ind_host, dtype="int32"))
        self.values.from_numpy(
            _numpy_array(self.values_host, dtype=self._numpy_float_dtype)
        )
        self._fill_x(0.0)
        self._zero_y()
        self._zero_b()
        self._zero_r()
        self._zero_rhs_scaled()
        self._zero_r_hat()
        self._zero_z()
        self._zero_p()
        self._zero_ap()
        self._zero_v()
        self._build_diag()
        self._build_symmetric_equilibration_scale()
        self._reset_row_column_equilibration_scale()
        self._reset_ilu0()
        self._zero_row_col_norm()
        self._zero_scalar()
        ti.sync()

    def _apply_bicgstab_preconditioner(
        self,
        preconditioner: str,
        src,
        dst,
        ilu0_pivot_tolerance: float,
    ) -> None:
        if preconditioner == "jacobi":
            self._jacobi(src, dst)
        elif preconditioner == "ilu0":
            self._ilu0_apply(src, dst, ilu0_pivot_tolerance)
        else:
            self._copy(src, dst)

    def _build_ilu0(self, pivot_tolerance: float) -> None:
        self._copy_values_to_ilu0()
        self._factor_ilu0(float(pivot_tolerance))
        ti.sync()

    def _ilu0_is_valid(self) -> bool:
        ti.sync()
        return int(self.ilu0_valid.to_numpy()[()]) == 1

    def _field_by_name(self, name: str):
        if name == "x":
            return self.x, self.n_cols
        if name == "y":
            return self.y, self.n_rows
        if name == "b":
            return self.b, self.n_rows
        if name == "r":
            return self.r, self.n_rows
        if name == "rhs_scaled":
            return self.rhs_scaled, self.n_rows
        if name == "r_hat":
            return self.r_hat, self.n_rows
        if name == "z":
            return self.z, self.n_rows
        if name == "p":
            return self.p, self.n_rows
        if name == "ap":
            return self.ap, self.n_rows
        if name == "v":
            return self.v, self.n_rows
        raise ValueError(f"unknown Taichi CSR vector field: {name}")

    def _dot_value(self, left, right) -> float:
        self._dot(left, right, self.n_rows)
        ti.sync()
        return float(self.scalar.to_numpy()[()])

    def _norm_value(self, field) -> float:
        self._dot(field, field, self.n_rows)
        ti.sync()
        return math.sqrt(max(float(self.scalar.to_numpy()[()]), 0.0))

    @ti.kernel
    def _fill_x(self, value: float):
        for i in self.x:
            self.x[i] = value

    @ti.kernel
    def _zero_x(self):
        for i in self.x:
            self.x[i] = 0.0

    @ti.kernel
    def _zero_y(self):
        for i in self.y:
            self.y[i] = 0.0

    @ti.kernel
    def _zero_b(self):
        for i in self.b:
            self.b[i] = 0.0

    @ti.kernel
    def _zero_r(self):
        for i in self.r:
            self.r[i] = 0.0

    @ti.kernel
    def _zero_rhs_scaled(self):
        for i in self.rhs_scaled:
            self.rhs_scaled[i] = 0.0

    @ti.kernel
    def _zero_r_hat(self):
        for i in self.r_hat:
            self.r_hat[i] = 0.0

    @ti.kernel
    def _zero_z(self):
        for i in self.z:
            self.z[i] = 0.0

    @ti.kernel
    def _zero_p(self):
        for i in self.p:
            self.p[i] = 0.0

    @ti.kernel
    def _zero_ap(self):
        for i in self.ap:
            self.ap[i] = 0.0

    @ti.kernel
    def _zero_v(self):
        for i in self.v:
            self.v[i] = 0.0

    @ti.kernel
    def _zero_scalar(self):
        self.scalar[None] = 0.0

    @ti.kernel
    def _build_diag(self):
        for row in range(self.n_rows):
            value = self.values[0] * 0.0
            self.diag_index[row] = -1
            for offset in range(self.row_ptr[row], self.row_ptr[row + 1]):
                if self.col_ind[offset] == row:
                    value += self.values[offset]
                    self.diag_index[row] = offset
            self.diag[row] = value

    @ti.kernel
    def _copy(self, src: ti.template(), dst: ti.template()):
        for i in src:
            dst[i] = src[i]

    @ti.kernel
    def _matvec(self):
        for row in range(self.n_rows):
            total = self.values[0] * 0.0
            for offset in range(self.row_ptr[row], self.row_ptr[row + 1]):
                total += self.values[offset] * self.x[self.col_ind[offset]]
            self.y[row] = total

    @ti.kernel
    def _matvec_field(self, vector: ti.template(), output: ti.template()):
        for row in range(self.n_rows):
            total = self.values[0] * 0.0
            for offset in range(self.row_ptr[row], self.row_ptr[row + 1]):
                total += self.values[offset] * vector[self.col_ind[offset]]
            output[row] = total

    @ti.kernel
    def _matvec_equilibrated_field(self, vector: ti.template(), output: ti.template()):
        for row in range(self.n_rows):
            total = self.values[0] * 0.0
            for offset in range(self.row_ptr[row], self.row_ptr[row + 1]):
                col = self.col_ind[offset]
                total += self.values[offset] * self.eq_scale[col] * vector[col]
            output[row] = self.eq_scale[row] * total

    @ti.kernel
    def _matvec_row_column_equilibrated_field(
        self,
        vector: ti.template(),
        output: ti.template(),
    ):
        for row in range(self.n_rows):
            total = self.values[0] * 0.0
            for offset in range(self.row_ptr[row], self.row_ptr[row + 1]):
                col = self.col_ind[offset]
                total += self.values[offset] * self.col_scale[col] * vector[col]
            output[row] = self.row_scale[row] * total

    @ti.kernel
    def _residual(self):
        for row in range(self.n_rows):
            self.r[row] = self.b[row] - self.y[row]

    @ti.kernel
    def _row_column_scaled_residual(self):
        for row in range(self.n_rows):
            self.r[row] = self.rhs_scaled[row] - self.y[row]

    @ti.kernel
    def _dot(self, left: ti.template(), right: ti.template(), n: ti.i32):
        self.scalar[None] = 0.0
        for i in range(n):
            self.scalar[None] += left[i] * right[i]

    @ti.kernel
    def _jacobi(self, src: ti.template(), dst: ti.template()):
        for i in src:
            diag = self.diag[i]
            if ti.abs(diag) > 1.0e-30:
                dst[i] = src[i] / diag
            else:
                dst[i] = src[i]

    @ti.kernel
    def _build_symmetric_equilibration_scale(self):
        for i in self.eq_scale:
            diag = self.diag[i]
            if diag > 1.0e-30:
                self.eq_scale[i] = 1.0 / ti.sqrt(diag)
            else:
                self.eq_scale[i] = 0.0

    def _build_row_column_equilibration_scale(self, passes: int) -> None:
        self._reset_row_column_equilibration_scale()
        for _ in range(passes):
            self._zero_row_col_norm()
            self._accumulate_row_column_l1_norms()
            self._update_row_column_equilibration_scale()
        ti.sync()

    @ti.kernel
    def _reset_row_column_equilibration_scale(self):
        for i in self.row_scale:
            self.row_scale[i] = 1.0
        for i in self.col_scale:
            self.col_scale[i] = 1.0

    @ti.kernel
    def _zero_row_col_norm(self):
        for i in self.row_norm:
            self.row_norm[i] = 0.0
        for i in self.col_norm:
            self.col_norm[i] = 0.0

    @ti.kernel
    def _accumulate_row_column_l1_norms(self):
        for row in range(self.n_rows):
            row_total = self.values[0] * 0.0
            for offset in range(self.row_ptr[row], self.row_ptr[row + 1]):
                col = self.col_ind[offset]
                scaled_abs = ti.abs(
                    self.row_scale[row] * self.values[offset] * self.col_scale[col]
                )
                row_total += scaled_abs
                ti.atomic_add(self.col_norm[col], scaled_abs)
            self.row_norm[row] = row_total

    @ti.kernel
    def _update_row_column_equilibration_scale(self):
        for i in self.row_scale:
            norm = self.row_norm[i]
            if norm > 1.0e-30:
                updated = self.row_scale[i] / ti.sqrt(norm)
                if updated < 1.0e-12:
                    updated = 1.0e-12
                if updated > 1.0e12:
                    updated = 1.0e12
                self.row_scale[i] = updated
        for i in self.col_scale:
            norm = self.col_norm[i]
            if norm > 1.0e-30:
                updated = self.col_scale[i] / ti.sqrt(norm)
                if updated < 1.0e-12:
                    updated = 1.0e-12
                if updated > 1.0e12:
                    updated = 1.0e12
                self.col_scale[i] = updated

    @ti.kernel
    def _reset_ilu0(self):
        for offset in self.ilu0_values:
            self.ilu0_values[offset] = self.values[offset]
        for i in self.ilu0_work:
            self.ilu0_work[i] = 0.0
        self.ilu0_valid[None] = 0
        self.ilu0_min_abs_pivot[None] = 0.0

    @ti.kernel
    def _copy_values_to_ilu0(self):
        for offset in self.ilu0_values:
            self.ilu0_values[offset] = self.values[offset]
        for i in self.ilu0_work:
            self.ilu0_work[i] = 0.0
        self.ilu0_valid[None] = 1
        self.ilu0_min_abs_pivot[None] = 1.0e30

    @ti.kernel
    def _factor_ilu0(self, pivot_tolerance: float):
        ti.loop_config(serialize=True)
        for row in range(self.n_rows):
            row_start = self.row_ptr[row]
            row_end = self.row_ptr[row + 1]
            row_diag_pos = self.diag_index[row]
            if row_diag_pos < 0:
                self.ilu0_valid[None] = 0
            for offset in range(row_start, row_end):
                col = self.col_ind[offset]
                if col < row:
                    pivot_pos = self.diag_index[col]
                    if pivot_pos < 0:
                        self.ilu0_valid[None] = 0
                    else:
                        pivot = self.ilu0_values[pivot_pos]
                        if ti.abs(pivot) <= pivot_tolerance:
                            self.ilu0_valid[None] = 0
                        else:
                            factor = self.ilu0_values[offset] / pivot
                            self.ilu0_values[offset] = factor
                            for upper_offset in range(
                                self.row_ptr[col],
                                self.row_ptr[col + 1],
                            ):
                                upper_col = self.col_ind[upper_offset]
                                if upper_col > col:
                                    row_pos = -1
                                    for scan in range(row_start, row_end):
                                        if self.col_ind[scan] == upper_col:
                                            row_pos = scan
                                    if row_pos >= 0:
                                        self.ilu0_values[row_pos] -= (
                                            factor * self.ilu0_values[upper_offset]
                                        )
            if row_diag_pos >= 0:
                abs_diag = ti.abs(self.ilu0_values[row_diag_pos])
                if abs_diag <= pivot_tolerance:
                    self.ilu0_valid[None] = 0
                else:
                    if abs_diag < self.ilu0_min_abs_pivot[None]:
                        self.ilu0_min_abs_pivot[None] = abs_diag

    @ti.kernel
    def _ilu0_apply(
        self,
        src: ti.template(),
        dst: ti.template(),
        pivot_tolerance: float,
    ):
        ti.loop_config(serialize=True)
        for row in range(self.n_rows):
            total = src[row]
            for offset in range(self.row_ptr[row], self.row_ptr[row + 1]):
                col = self.col_ind[offset]
                if col < row:
                    total -= self.ilu0_values[offset] * self.ilu0_work[col]
            self.ilu0_work[row] = total
        ti.loop_config(serialize=True)
        for reverse_index in range(self.n_rows):
            row = self.n_rows - 1 - reverse_index
            total = self.ilu0_work[row]
            diag = self.values[0] * 0.0
            diag_pos = self.diag_index[row]
            if diag_pos >= 0:
                diag = self.ilu0_values[diag_pos]
            for offset in range(self.row_ptr[row], self.row_ptr[row + 1]):
                col = self.col_ind[offset]
                if col > row:
                    total -= self.ilu0_values[offset] * dst[col]
            if ti.abs(diag) > pivot_tolerance:
                dst[row] = total / diag
            else:
                dst[row] = total

    @ti.kernel
    def _scale_b_to_r(self):
        for i in self.r:
            self.r[i] = self.eq_scale[i] * self.b[i]

    @ti.kernel
    def _scale_b_to_rhs_scaled(self):
        for i in self.rhs_scaled:
            self.rhs_scaled[i] = self.row_scale[i] * self.b[i]

    @ti.kernel
    def _equilibrated_solution_to_original(self):
        for i in self.x:
            self.x[i] = self.eq_scale[i] * self.x[i]

    @ti.kernel
    def _row_column_equilibrated_solution_to_original(self):
        for i in self.x:
            self.x[i] = self.col_scale[i] * self.x[i]

    @ti.kernel
    def _cg_update_x_r(self, alpha: float):
        for i in self.x:
            self.x[i] += alpha * self.p[i]
            self.r[i] -= alpha * self.ap[i]

    @ti.kernel
    def _cg_update_p(self, beta: float):
        for i in self.p:
            self.p[i] = self.z[i] + beta * self.p[i]

    @ti.kernel
    def _bicgstab_update_p(self, beta: float, omega: float):
        for i in self.p:
            self.p[i] = self.r[i] + beta * (self.p[i] - omega * self.v[i])

    @ti.kernel
    def _bicgstab_update_x_s(self, alpha: float):
        for i in self.x:
            self.x[i] += alpha * self.z[i]
            self.y[i] = self.r[i] - alpha * self.v[i]

    @ti.kernel
    def _bicgstab_update_x_r(self, omega: float):
        for i in self.x:
            self.x[i] += omega * self.z[i]
            self.r[i] = self.y[i] - omega * self.ap[i]

    @ti.kernel
    def _richardson_update_x_r(self, omega: float):
        for i in self.x:
            self.x[i] += omega * self.z[i]
            self.r[i] -= omega * self.ap[i]

    @ti.kernel
    def _scale_z(self, alpha: float):
        for i in self.z:
            self.z[i] *= alpha

    @ti.kernel
    def _axpy_z_from_p(self, alpha: float):
        for i in self.z:
            self.z[i] += alpha * self.p[i]


def _numpy_array(values: Iterable[float | int], dtype: str = "float32"):
    import numpy as np

    return np.asarray(tuple(values), dtype=dtype)


def _taichi_float_dtype(dtype: str):
    if dtype == "float32":
        return ti.f32
    if dtype == "float64":
        return ti.f64
    raise ValueError(f"unsupported Taichi CSR dtype: {dtype}")


def _numpy_float_dtype(dtype: str) -> str:
    if dtype == "float32":
        return "float32"
    if dtype == "float64":
        return "float64"
    raise ValueError(f"unsupported Taichi CSR dtype: {dtype}")


def _row_column_equilibration_passes(plan: PolicyPlan) -> int:
    passes = int(plan.preconditioner.get("passes", 4))
    if passes < 1:
        raise ValueError("row_column_equilibration passes must be >= 1")
    return min(passes, 16)


def _ilu0_pivot_tolerance(plan: PolicyPlan) -> float:
    tolerance = float(plan.preconditioner.get("pivot_tolerance", 1.0e-12))
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("ilu0 pivot_tolerance must be finite and nonnegative")
    return tolerance


def _scale_stats(values) -> tuple[float, float]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return (math.nan, math.nan)
    return (min(finite), max(finite))


def _chebyshev_spectral_bounds(plan: PolicyPlan) -> tuple[float, float]:
    solver = plan.solver
    if "lambda_min" not in solver or "lambda_max" not in solver:
        raise ValueError("Taichi CSR Chebyshev requires explicit lambda_min/lambda_max")
    lower = float(solver["lambda_min"])
    upper = float(solver["lambda_max"])
    if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower < upper):
        raise ValueError(f"invalid chebyshev spectral bounds: {lower}, {upper}")
    return lower, upper
