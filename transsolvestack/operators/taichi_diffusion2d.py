"""Taichi GPU 2D diffusion stencil operator and smoke solvers."""

import time
from dataclasses import dataclass
from math import cos, pi
from typing import Any

import taichi as ti

from transsolvestack.core.result import PolicyPlan, RunTrace, SolveResult
from transsolvestack.core.types import LinearOperatorSpec, SolveContext
from transsolvestack.operators.base import LinearOperator


_TAICHI_INITIALIZED = False


def ensure_taichi_cuda(device_memory_gb: float = 0.5) -> None:
    """Initialize Taichi CUDA once, with a small memory budget."""

    global _TAICHI_INITIALIZED
    if _TAICHI_INITIALIZED:
        return
    try:
        ti.init(arch=ti.cuda, device_memory_GB=device_memory_gb)
    except TypeError:
        ti.init(arch=ti.cuda)
    _TAICHI_INITIALIZED = True


@ti.data_oriented
@dataclass(eq=False)
class TaichiDiffusion2DOperator(LinearOperator):
    """GPU-resident matrix-free 2D diffusion operator.

    It solves A x = A 1, so the exact solution is the all-ones vector. This gives
    a real numerical correctness check without materializing A on CPU.
    """

    spec: LinearOperatorSpec
    nx: int
    ny: int
    ax: float = 1.0
    ay: float = 1.0

    def __post_init__(self) -> None:
        self.x = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.x_true = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.b = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.r = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.z = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.p = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.ap = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.r_hat = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.v = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.s = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.t = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.y = ti.field(dtype=ti.f32, shape=(self.nx, self.ny))
        self.scalar = ti.field(dtype=ti.f32, shape=())
        self._initialize_problem(self.ax, self.ay)
        self._apply(self.x_true, self.b, self.ax, self.ay)
        ti.sync()

    def apply(self, x: Any, y: Any | None = None) -> Any:
        if y is None:
            raise ValueError("TaichiDiffusion2DOperator.apply requires an output field")
        self._apply(x, y, self.ax, self.ay)
        return y

    def solve_cg(
        self,
        rhs: Any,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        return self._solve_cg(context=context, plan=plan, use_jacobi=False)

    def solve_pcg(
        self,
        rhs: Any,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        return self._solve_cg(context=context, plan=plan, use_jacobi=True)

    def solve_bicgstab(
        self,
        rhs: Any,
        context: SolveContext,
        plan: PolicyPlan,
    ) -> SolveResult:
        return self._solve_bicgstab(context=context, plan=plan, use_jacobi=True)

    def _solve_cg(
        self,
        context: SolveContext,
        plan: PolicyPlan,
        use_jacobi: bool,
    ) -> SolveResult:
        start = time.perf_counter()
        self._zero(self.x)
        self._copy(self.b, self.r)
        if use_jacobi:
            self._jacobi(self.r, self.z, self.ax, self.ay)
        else:
            self._copy(self.r, self.z)
        self._copy(self.z, self.p)
        rz_old = self._dot_value(self.r, self.z)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        residual_history = [self._norm_value(self.r) / b_norm]
        converged = False
        iterations = 0

        solve_start = time.perf_counter()
        for iteration in range(1, context.max_iter + 1):
            self._apply(self.p, self.ap, self.ax, self.ay)
            denom = self._dot_value(self.p, self.ap)
            if abs(denom) < 1.0e-30:
                break
            alpha = rz_old / denom
            self._cg_update_x_r(self.x, self.r, self.p, self.ap, alpha)
            res = self._norm_value(self.r) / b_norm
            residual_history.append(res)
            iterations = iteration
            if res <= context.tolerance_rel:
                converged = True
                break
            if use_jacobi:
                self._jacobi(self.r, self.z, self.ax, self.ay)
            else:
                self._copy(self.r, self.z)
            rz_new = self._dot_value(self.r, self.z)
            beta = rz_new / max(rz_old, 1.0e-30)
            self._cg_update_p(self.p, self.z, beta)
            rz_old = rz_new
        ti.sync()
        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        error = self._relative_error_to_true()
        final_residual = residual_history[-1]
        status = "success" if converged and error < 5.0e-3 else "failed"
        failure_class = None if status == "success" else "not_converged"
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
            final_residual_norm=final_residual,
            residual_history=residual_history,
            metadata={
                "solver": plan.solver.get("name"),
                "preconditioner": plan.preconditioner.get("name", "none"),
                "relative_error_to_true": error,
                "nx": self.nx,
                "ny": self.ny,
                "ax": self.ax,
                "ay": self.ay,
            },
        )
        return SolveResult(status=status, solution=self.x, trace=trace)

    def _solve_bicgstab(
        self,
        context: SolveContext,
        plan: PolicyPlan,
        use_jacobi: bool,
    ) -> SolveResult:
        start = time.perf_counter()
        self._zero(self.x)
        self._copy(self.b, self.r)
        self._copy(self.r, self.r_hat)
        self._zero(self.p)
        self._zero(self.v)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        residual_history = [self._norm_value(self.r) / b_norm]
        rho_old = 1.0
        alpha = 1.0
        omega = 1.0
        converged = False
        iterations = 0

        solve_start = time.perf_counter()
        for iteration in range(1, context.max_iter + 1):
            rho_new = self._dot_value(self.r_hat, self.r)
            if abs(rho_new) < 1.0e-30 or abs(omega) < 1.0e-30:
                break
            beta = (rho_new / rho_old) * (alpha / omega)
            self._bicgstab_update_p(self.p, self.r, self.v, beta, omega)
            if use_jacobi:
                self._jacobi(self.p, self.z, self.ax, self.ay)
            else:
                self._copy(self.p, self.z)
            self._apply(self.z, self.v, self.ax, self.ay)
            denom = self._dot_value(self.r_hat, self.v)
            if abs(denom) < 1.0e-30:
                break
            alpha = rho_new / denom
            self._bicgstab_compute_s(self.s, self.r, self.v, alpha)
            s_norm = self._norm_value(self.s) / b_norm
            if s_norm <= context.tolerance_rel:
                self._axpy(self.x, self.z, alpha)
                residual_history.append(s_norm)
                converged = True
                iterations = iteration
                break
            if use_jacobi:
                self._jacobi(self.s, self.y, self.ax, self.ay)
            else:
                self._copy(self.s, self.y)
            self._apply(self.y, self.t, self.ax, self.ay)
            tt = self._dot_value(self.t, self.t)
            if abs(tt) < 1.0e-30:
                break
            omega = self._dot_value(self.t, self.s) / tt
            self._bicgstab_update_x_r(self.x, self.r, self.z, self.y, self.s, self.t, alpha, omega)
            res = self._norm_value(self.r) / b_norm
            residual_history.append(res)
            iterations = iteration
            if res <= context.tolerance_rel:
                converged = True
                break
            rho_old = rho_new
        ti.sync()
        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        error = self._relative_error_to_true()
        final_residual = residual_history[-1]
        status = "success" if converged and error < 5.0e-3 else "failed"
        failure_class = None if status == "success" else "not_converged"
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
            final_residual_norm=final_residual,
            residual_history=residual_history,
            metadata={
                "solver": plan.solver.get("name"),
                "preconditioner": plan.preconditioner.get("name", "none"),
                "relative_error_to_true": error,
                "nx": self.nx,
                "ny": self.ny,
                "ax": self.ax,
                "ay": self.ay,
            },
        )
        return SolveResult(status=status, solution=self.x, trace=trace)

    def _dot_value(self, a: Any, b: Any) -> float:
        self._dot(a, b, self.scalar)
        ti.sync()
        return float(self.scalar[None])

    def _norm_value(self, a: Any) -> float:
        self._dot(a, a, self.scalar)
        ti.sync()
        return float(self.scalar[None]) ** 0.5

    def _relative_error_to_true(self) -> float:
        self._error_norm(self.x, self.x_true, self.scalar)
        ti.sync()
        err = float(self.scalar[None]) ** 0.5
        true_norm = max(self._norm_value(self.x_true), 1.0e-30)
        return err / true_norm

    @ti.kernel
    def _initialize_problem(self, ax: float, ay: float):
        for i, j in self.x:
            self.x[i, j] = 0.0
            self.x_true[i, j] = 1.0
            self.b[i, j] = 0.0
            self.r[i, j] = 0.0
            self.z[i, j] = 0.0
            self.p[i, j] = 0.0
            self.ap[i, j] = 0.0

    @ti.kernel
    def _zero(self, a: ti.template()):
        for i, j in a:
            a[i, j] = 0.0

    @ti.kernel
    def _copy(self, src: ti.template(), dst: ti.template()):
        for i, j in src:
            dst[i, j] = src[i, j]

    @ti.kernel
    def _apply(self, x: ti.template(), y: ti.template(), ax: float, ay: float):
        for i, j in x:
            value = 2.0 * (ax + ay) * x[i, j]
            if i > 0:
                value -= ax * x[i - 1, j]
            if i + 1 < self.nx:
                value -= ax * x[i + 1, j]
            if j > 0:
                value -= ay * x[i, j - 1]
            if j + 1 < self.ny:
                value -= ay * x[i, j + 1]
            y[i, j] = value

    @ti.kernel
    def _jacobi(self, r: ti.template(), z: ti.template(), ax: float, ay: float):
        diag = 2.0 * (ax + ay)
        for i, j in r:
            z[i, j] = r[i, j] / diag

    @ti.kernel
    def _dot(self, a: ti.template(), b: ti.template(), out: ti.template()):
        out[None] = 0.0
        for i, j in a:
            out[None] += a[i, j] * b[i, j]

    @ti.kernel
    def _error_norm(self, x: ti.template(), x_true: ti.template(), out: ti.template()):
        out[None] = 0.0
        for i, j in x:
            diff = x[i, j] - x_true[i, j]
            out[None] += diff * diff

    @ti.kernel
    def _cg_update_x_r(
        self,
        x: ti.template(),
        r: ti.template(),
        p: ti.template(),
        ap: ti.template(),
        alpha: float,
    ):
        for i, j in x:
            x[i, j] += alpha * p[i, j]
            r[i, j] -= alpha * ap[i, j]

    @ti.kernel
    def _cg_update_p(self, p: ti.template(), z: ti.template(), beta: float):
        for i, j in p:
            p[i, j] = z[i, j] + beta * p[i, j]

    @ti.kernel
    def _bicgstab_update_p(
        self,
        p: ti.template(),
        r: ti.template(),
        v: ti.template(),
        beta: float,
        omega: float,
    ):
        for i, j in p:
            p[i, j] = r[i, j] + beta * (p[i, j] - omega * v[i, j])

    @ti.kernel
    def _bicgstab_compute_s(
        self,
        s: ti.template(),
        r: ti.template(),
        v: ti.template(),
        alpha: float,
    ):
        for i, j in s:
            s[i, j] = r[i, j] - alpha * v[i, j]

    @ti.kernel
    def _axpy(self, x: ti.template(), y: ti.template(), alpha: float):
        for i, j in x:
            x[i, j] += alpha * y[i, j]

    @ti.kernel
    def _bicgstab_update_x_r(
        self,
        x: ti.template(),
        r: ti.template(),
        z: ti.template(),
        y: ti.template(),
        s: ti.template(),
        t: ti.template(),
        alpha: float,
        omega: float,
    ):
        for i, j in x:
            x[i, j] += alpha * z[i, j] + omega * y[i, j]
            r[i, j] = s[i, j] - omega * t[i, j]


@ti.data_oriented
@dataclass(eq=False)
class TaichiDiffusionFlatStencilOperator(LinearOperator):
    """GPU-resident flattened 2D/3D diffusion stencil operator."""

    spec: LinearOperatorSpec
    nx: int
    ny: int
    nz: int = 1
    ax: float = 1.0
    ay: float = 1.0
    az: float = 0.0

    def __post_init__(self) -> None:
        self.n = self.nx * self.ny * self.nz
        self.x = ti.field(dtype=ti.f32, shape=(self.n,))
        self.x_true = ti.field(dtype=ti.f32, shape=(self.n,))
        self.b = ti.field(dtype=ti.f32, shape=(self.n,))
        self.r = ti.field(dtype=ti.f32, shape=(self.n,))
        self.z = ti.field(dtype=ti.f32, shape=(self.n,))
        self.p = ti.field(dtype=ti.f32, shape=(self.n,))
        self.ap = ti.field(dtype=ti.f32, shape=(self.n,))
        self.r_hat = ti.field(dtype=ti.f32, shape=(self.n,))
        self.v = ti.field(dtype=ti.f32, shape=(self.n,))
        self.s = ti.field(dtype=ti.f32, shape=(self.n,))
        self.t = ti.field(dtype=ti.f32, shape=(self.n,))
        self.y = ti.field(dtype=ti.f32, shape=(self.n,))
        self.scalar = ti.field(dtype=ti.f32, shape=())
        self.coeff_ax = ti.field(dtype=ti.f32, shape=())
        self.coeff_ay = ti.field(dtype=ti.f32, shape=())
        self.coeff_az = ti.field(dtype=ti.f32, shape=())
        self.coeff_ax[None] = self.ax
        self.coeff_ay[None] = self.ay
        self.coeff_az[None] = self.az
        self._initialize_problem()
        self._apply(self.x_true, self.b)
        ti.sync()

    def update_coefficients(self, ax: float, ay: float, az: float = 0.0) -> None:
        self.ax = ax
        self.ay = ay
        self.az = az
        self.coeff_ax[None] = ax
        self.coeff_ay[None] = ay
        self.coeff_az[None] = az
        self._zero(self.x)
        self._apply(self.x_true, self.b)
        ti.sync()

    def apply(self, x: Any, y: Any | None = None) -> Any:
        if y is None:
            raise ValueError("TaichiDiffusionFlatStencilOperator.apply requires y")
        self._apply(x, y)
        return y

    def solve_cg(self, rhs: Any, context: SolveContext, plan: PolicyPlan) -> SolveResult:
        return self._solve_cg(context=context, plan=plan, use_jacobi=False)

    def solve_pcg(self, rhs: Any, context: SolveContext, plan: PolicyPlan) -> SolveResult:
        return self._solve_cg(context=context, plan=plan, use_jacobi=True)

    def solve_bicgstab(
        self, rhs: Any, context: SolveContext, plan: PolicyPlan
    ) -> SolveResult:
        return self._solve_bicgstab(context=context, plan=plan, use_jacobi=True)

    def solve_richardson(
        self, rhs: Any, context: SolveContext, plan: PolicyPlan
    ) -> SolveResult:
        return self._solve_richardson_jacobi(context=context, plan=plan)

    def solve_chebyshev(
        self, rhs: Any, context: SolveContext, plan: PolicyPlan
    ) -> SolveResult:
        return self._solve_chebyshev_jacobi(context=context, plan=plan)

    def _solve_cg(
        self, context: SolveContext, plan: PolicyPlan, use_jacobi: bool
    ) -> SolveResult:
        start = time.perf_counter()
        self._zero(self.x)
        self._copy(self.b, self.r)
        if use_jacobi:
            self._jacobi(self.r, self.z)
        else:
            self._copy(self.r, self.z)
        self._copy(self.z, self.p)
        rz_old = self._dot_value(self.r, self.z)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        residual_history = [self._norm_value(self.r) / b_norm]
        converged = False
        iterations = 0
        solve_start = time.perf_counter()
        for iteration in range(1, context.max_iter + 1):
            self._apply(self.p, self.ap)
            denom = self._dot_value(self.p, self.ap)
            if abs(denom) < 1.0e-30:
                break
            alpha = rz_old / denom
            self._cg_update_x_r(self.x, self.r, self.p, self.ap, alpha)
            res = self._norm_value(self.r) / b_norm
            residual_history.append(res)
            iterations = iteration
            if res <= context.tolerance_rel:
                converged = True
                break
            if use_jacobi:
                self._jacobi(self.r, self.z)
            else:
                self._copy(self.r, self.z)
            rz_new = self._dot_value(self.r, self.z)
            beta = rz_new / max(rz_old, 1.0e-30)
            self._cg_update_p(self.p, self.z, beta)
            rz_old = rz_new
        ti.sync()
        return self._make_result(
            context=context,
            plan=plan,
            start=start,
            solve_start=solve_start,
            converged=converged,
            iterations=iterations,
            residual_history=residual_history,
        )

    def _solve_richardson_jacobi(
        self, context: SolveContext, plan: PolicyPlan
    ) -> SolveResult:
        start = time.perf_counter()
        self._zero(self.x)
        self._copy(self.b, self.r)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        residual_history = [self._norm_value(self.r) / b_norm]
        omega = float(plan.solver.get("omega", 2.0 / 3.0))
        if not 0.0 < omega < 1.0:
            raise ValueError("richardson omega must be in (0, 1)")
        converged = False
        iterations = 0
        solve_start = time.perf_counter()
        for iteration in range(1, context.max_iter + 1):
            self._jacobi(self.r, self.z)
            self._apply(self.z, self.ap)
            self._cg_update_x_r(self.x, self.r, self.z, self.ap, omega)
            res = self._norm_value(self.r) / b_norm
            residual_history.append(res)
            iterations = iteration
            if res <= context.tolerance_rel:
                converged = True
                break
        ti.sync()
        return self._make_result(
            context=context,
            plan=plan,
            start=start,
            solve_start=solve_start,
            converged=converged,
            iterations=iterations,
            residual_history=residual_history,
            extra_metadata={"omega": omega},
        )

    def _solve_chebyshev_jacobi(
        self, context: SolveContext, plan: PolicyPlan
    ) -> SolveResult:
        start = time.perf_counter()
        self._zero(self.x)
        self._copy(self.b, self.r)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        residual_history = [self._norm_value(self.r) / b_norm]
        lower, upper = self._chebyshev_spectral_bounds(plan)
        d = 0.5 * (upper + lower)
        c = 0.5 * (upper - lower)
        alpha = 0.0
        converged = False
        iterations = 0
        solve_start = time.perf_counter()
        for iteration in range(1, context.max_iter + 1):
            self._jacobi(self.r, self.z)
            if iteration == 1:
                self._copy(self.z, self.p)
                alpha = 1.0 / d
            else:
                if iteration == 2:
                    beta = 0.5 * (c * alpha) ** 2
                else:
                    beta = (0.5 * c * alpha) ** 2
                alpha = 1.0 / (d - beta / alpha)
                self._cg_update_p(self.p, self.z, beta)
            self._apply(self.p, self.ap)
            self._cg_update_x_r(self.x, self.r, self.p, self.ap, alpha)
            res = self._norm_value(self.r) / b_norm
            residual_history.append(res)
            iterations = iteration
            if res <= context.tolerance_rel:
                converged = True
                break
        ti.sync()
        return self._make_result(
            context=context,
            plan=plan,
            start=start,
            solve_start=solve_start,
            converged=converged,
            iterations=iterations,
            residual_history=residual_history,
            extra_metadata={
                "lambda_min": lower,
                "lambda_max": upper,
                "spectral_bounds": "jacobi_preconditioned_stencil_estimate",
            },
        )

    def _solve_bicgstab(
        self, context: SolveContext, plan: PolicyPlan, use_jacobi: bool
    ) -> SolveResult:
        start = time.perf_counter()
        self._zero(self.x)
        self._copy(self.b, self.r)
        self._copy(self.r, self.r_hat)
        self._zero(self.p)
        self._zero(self.v)
        b_norm = max(self._norm_value(self.b), 1.0e-30)
        residual_history = [self._norm_value(self.r) / b_norm]
        rho_old = 1.0
        alpha = 1.0
        omega = 1.0
        converged = False
        iterations = 0
        solve_start = time.perf_counter()
        for iteration in range(1, context.max_iter + 1):
            rho_new = self._dot_value(self.r_hat, self.r)
            if abs(rho_new) < 1.0e-30 or abs(omega) < 1.0e-30:
                break
            beta = (rho_new / rho_old) * (alpha / omega)
            self._bicgstab_update_p(self.p, self.r, self.v, beta, omega)
            if use_jacobi:
                self._jacobi(self.p, self.z)
            else:
                self._copy(self.p, self.z)
            self._apply(self.z, self.v)
            denom = self._dot_value(self.r_hat, self.v)
            if abs(denom) < 1.0e-30:
                break
            alpha = rho_new / denom
            self._bicgstab_compute_s(self.s, self.r, self.v, alpha)
            s_norm = self._norm_value(self.s) / b_norm
            if s_norm <= context.tolerance_rel:
                self._axpy(self.x, self.z, alpha)
                residual_history.append(s_norm)
                converged = True
                iterations = iteration
                break
            if use_jacobi:
                self._jacobi(self.s, self.y)
            else:
                self._copy(self.s, self.y)
            self._apply(self.y, self.t)
            tt = self._dot_value(self.t, self.t)
            if abs(tt) < 1.0e-30:
                break
            omega = self._dot_value(self.t, self.s) / tt
            self._bicgstab_update_x_r(
                self.x, self.r, self.z, self.y, self.s, self.t, alpha, omega
            )
            res = self._norm_value(self.r) / b_norm
            residual_history.append(res)
            iterations = iteration
            if res <= context.tolerance_rel:
                converged = True
                break
            rho_old = rho_new
        ti.sync()
        return self._make_result(
            context=context,
            plan=plan,
            start=start,
            solve_start=solve_start,
            converged=converged,
            iterations=iterations,
            residual_history=residual_history,
        )

    def _make_result(
        self,
        context: SolveContext,
        plan: PolicyPlan,
        start: float,
        solve_start: float,
        converged: bool,
        iterations: int,
        residual_history: list[float],
        extra_metadata: dict[str, Any] | None = None,
    ) -> SolveResult:
        solve_ms = (time.perf_counter() - solve_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0
        error = self._relative_error_to_true()
        final_residual = residual_history[-1]
        status = "success" if converged and error < 5.0e-3 else "failed"
        failure_class = None if status == "success" else "not_converged"
        metadata = {
            "solver": plan.solver.get("name"),
            "preconditioner": plan.preconditioner.get("name", "none"),
            "relative_error_to_true": error,
            "nx": self.nx,
            "ny": self.ny,
            "nz": self.nz,
            "ax": self.ax,
            "ay": self.ay,
            "az": self.az,
        }
        metadata.update(extra_metadata or {})
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
            final_residual_norm=final_residual,
            residual_history=residual_history,
            metadata=metadata,
        )
        return SolveResult(status=status, solution=self.x, trace=trace)

    def _chebyshev_spectral_bounds(self, plan: PolicyPlan) -> tuple[float, float]:
        solver = plan.solver
        if "lambda_min" in solver and "lambda_max" in solver:
            lower = float(solver["lambda_min"])
            upper = float(solver["lambda_max"])
        else:
            weighted_cos = 0.0
            coeffs = ((self.ax, self.nx), (self.ay, self.ny), (self.az, self.nz))
            for coeff, size in coeffs:
                if coeff > 0.0 and size > 1:
                    weighted_cos += coeff * cos(pi / float(size + 1))
            denom = max(self.ax + self.ay + self.az, 1.0e-30)
            radius = min(weighted_cos / denom, 0.999999)
            lower = max(1.0 - radius, 1.0e-6)
            upper = 1.0 + radius
        if not 0.0 < lower < upper:
            raise ValueError(f"invalid chebyshev spectral bounds: {lower}, {upper}")
        return lower, upper

    def _dot_value(self, a: Any, b: Any) -> float:
        self._dot(a, b, self.scalar)
        ti.sync()
        return float(self.scalar[None])

    def _norm_value(self, a: Any) -> float:
        self._dot(a, a, self.scalar)
        ti.sync()
        return float(self.scalar[None]) ** 0.5

    def _relative_error_to_true(self) -> float:
        self._error_norm(self.x, self.x_true, self.scalar)
        ti.sync()
        err = float(self.scalar[None]) ** 0.5
        true_norm = max(self._norm_value(self.x_true), 1.0e-30)
        return err / true_norm

    @ti.kernel
    def _initialize_problem(self):
        for idx in self.x:
            self.x[idx] = 0.0
            self.x_true[idx] = 1.0
            self.b[idx] = 0.0
            self.r[idx] = 0.0
            self.z[idx] = 0.0
            self.p[idx] = 0.0
            self.ap[idx] = 0.0

    @ti.kernel
    def _zero(self, a: ti.template()):
        for idx in a:
            a[idx] = 0.0

    @ti.kernel
    def _copy(self, src: ti.template(), dst: ti.template()):
        for idx in src:
            dst[idx] = src[idx]

    @ti.kernel
    def _apply(self, x: ti.template(), y: ti.template()):
        for idx in x:
            ax = self.coeff_ax[None]
            ay = self.coeff_ay[None]
            az = self.coeff_az[None]
            plane = self.ny * self.nz
            i = idx // plane
            rem = idx - i * plane
            j = rem // self.nz
            k = rem - j * self.nz
            value = 2.0 * (ax + ay + az) * x[idx]
            if i > 0:
                value -= ax * x[idx - plane]
            if i + 1 < self.nx:
                value -= ax * x[idx + plane]
            if j > 0:
                value -= ay * x[idx - self.nz]
            if j + 1 < self.ny:
                value -= ay * x[idx + self.nz]
            if k > 0:
                value -= az * x[idx - 1]
            if k + 1 < self.nz:
                value -= az * x[idx + 1]
            y[idx] = value

    @ti.kernel
    def _jacobi(self, r: ti.template(), z: ti.template()):
        diag = 2.0 * (self.coeff_ax[None] + self.coeff_ay[None] + self.coeff_az[None])
        for idx in r:
            z[idx] = r[idx] / diag

    @ti.kernel
    def _dot(self, a: ti.template(), b: ti.template(), out: ti.template()):
        out[None] = 0.0
        for idx in a:
            out[None] += a[idx] * b[idx]

    @ti.kernel
    def _error_norm(self, x: ti.template(), x_true: ti.template(), out: ti.template()):
        out[None] = 0.0
        for idx in x:
            diff = x[idx] - x_true[idx]
            out[None] += diff * diff

    @ti.kernel
    def _cg_update_x_r(
        self,
        x: ti.template(),
        r: ti.template(),
        p: ti.template(),
        ap: ti.template(),
        alpha: float,
    ):
        for idx in x:
            x[idx] += alpha * p[idx]
            r[idx] -= alpha * ap[idx]

    @ti.kernel
    def _cg_update_p(self, p: ti.template(), z: ti.template(), beta: float):
        for idx in p:
            p[idx] = z[idx] + beta * p[idx]

    @ti.kernel
    def _bicgstab_update_p(
        self,
        p: ti.template(),
        r: ti.template(),
        v: ti.template(),
        beta: float,
        omega: float,
    ):
        for idx in p:
            p[idx] = r[idx] + beta * (p[idx] - omega * v[idx])

    @ti.kernel
    def _bicgstab_compute_s(
        self,
        s: ti.template(),
        r: ti.template(),
        v: ti.template(),
        alpha: float,
    ):
        for idx in s:
            s[idx] = r[idx] - alpha * v[idx]

    @ti.kernel
    def _axpy(self, x: ti.template(), y: ti.template(), alpha: float):
        for idx in x:
            x[idx] += alpha * y[idx]

    @ti.kernel
    def _bicgstab_update_x_r(
        self,
        x: ti.template(),
        r: ti.template(),
        z: ti.template(),
        y: ti.template(),
        s: ti.template(),
        t: ti.template(),
        alpha: float,
        omega: float,
    ):
        for idx in x:
            x[idx] += alpha * z[idx] + omega * y[idx]
            r[idx] = s[idx] - omega * t[idx]


def diffusion_operator_from_system(system: Any) -> TaichiDiffusionFlatStencilOperator:
    metadata = system.operator.metadata
    grid_size = tuple(int(v) for v in metadata["grid_size"])
    family_id = metadata["family_id"]
    if family_id == "poisson_2d_stencil":
        ax, ay, az = 1.0, 1.0, 0.0
    elif family_id == "anisotropic_diffusion_2d":
        ax, ay, az = 8.0, 1.0, 0.0
    elif family_id == "strong_anisotropic_diffusion_2d":
        ax, ay, az = 32.0, 1.0, 0.0
    elif family_id == "poisson_3d_stencil":
        ax, ay, az = 1.0, 1.0, 1.0
    else:
        raise ValueError(f"unsupported Taichi diffusion smoke family: {family_id}")
    return TaichiDiffusionFlatStencilOperator(
        spec=system.operator,
        nx=grid_size[0],
        ny=grid_size[1],
        nz=grid_size[2] if len(grid_size) == 3 else 1,
        ax=ax,
        ay=ay,
        az=az,
    )
