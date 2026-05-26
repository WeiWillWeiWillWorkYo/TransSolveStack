# Sequence Benchmark Report

- candidates: `2`
- step_records: `8`

## Candidate Summary

| candidate | status | steps | total_ms | iterations | max_residual | max_rel_error |
|---|---|---:|---:|---:|---:|---:|
| taichi_pcg_jacobi_f32 | success | 4 | 1629.79 | 314 | 8.8661e-07 | 8.14888e-07 |
| taichi_cg_none_f32 | success | 4 | 447.202 | 314 | 8.86366e-07 | 8.28163e-07 |

## Steps

- taichi_pcg_jacobi_f32 / s0: status=success, ax=2.0, total_ms=1443.3576869778335, iters=68, residual=6.602019543851204e-07
- taichi_pcg_jacobi_f32 / s1: status=success, ax=4.0, total_ms=47.38681297749281, iters=74, residual=7.856146171144502e-07
- taichi_pcg_jacobi_f32 / s2: status=success, ax=8.0, total_ms=57.53311701118946, iters=80, residual=8.144958672246536e-07
- taichi_pcg_jacobi_f32 / s3: status=success, ax=16.0, total_ms=81.51285536587238, iters=92, residual=8.866103582365057e-07
- taichi_cg_none_f32 / s0: status=success, ax=2.0, total_ms=135.59590838849545, iters=68, residual=6.592345519761027e-07
- taichi_cg_none_f32 / s1: status=success, ax=4.0, total_ms=83.18850072100759, iters=74, residual=7.715857420809986e-07
- taichi_cg_none_f32 / s2: status=success, ax=8.0, total_ms=106.91078286617994, iters=80, residual=8.109249599547151e-07
- taichi_cg_none_f32 / s3: status=success, ax=16.0, total_ms=121.5070141479373, iters=92, residual=8.86365956105642e-07