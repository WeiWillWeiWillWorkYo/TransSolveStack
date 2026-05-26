# Solver Functional Smoke

- rows: `2`
- success: `2`

| candidate | solver | preconditioner | status | iters | residual | rel_error | residual_drop |
|---|---|---|---|---:|---:|---:|---:|
| taichi_richardson_jacobi_f32 | richardson | jacobi | success | 225 | 9.60454e-07 | 3.1388e-06 | 9.60454e-07 |
| taichi_chebyshev_jacobi_f32 | chebyshev | jacobi | success | 41 | 8.14687e-07 | 8.60631e-07 | 8.14687e-07 |
