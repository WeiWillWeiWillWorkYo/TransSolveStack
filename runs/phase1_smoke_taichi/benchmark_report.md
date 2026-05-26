# Benchmark Evaluation Report

- rows: `21`
- systems: `7`
- candidates: `3`

## Per-System Candidate Table

| system | candidate | status | oracle | time_ms | regret | iters | residual | rel_error | repeats |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_bicgstab_jacobi_f32 | success | no | 59.9656 | 0.499966 | 61 | 7.45011e-07 | 1.31331e-05 | 3 |
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_cg_none_f32 | success | yes | 39.978 | 0 | 80 | 8.11369e-07 | 6.43306e-07 | 3 |
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_pcg_jacobi_f32 | success | no | 151.699 | 2.79455 | 80 | 8.08145e-07 | 6.49251e-07 | 3 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_bicgstab_jacobi_f32 | success | no | 129.39 | 0.668293 | 115 | 2.17362e-07 | 7.79839e-06 | 3 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_cg_none_f32 | success | no | 84.2416 | 0.0861674 | 156 | 9.42596e-07 | 1.37622e-06 | 3 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_pcg_jacobi_f32 | success | yes | 77.5586 | 0 | 156 | 9.4253e-07 | 1.35155e-06 | 3 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_bicgstab_jacobi_f32 | success | no | 67.0894 | 1.32633 | 45 | 5.7659e-07 | 1.02834e-05 | 3 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_cg_none_f32 | success | no | 81.6669 | 1.8318 | 53 | 7.94753e-07 | 3.81533e-07 | 3 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_pcg_jacobi_f32 | success | yes | 28.8392 | 0 | 53 | 8.07905e-07 | 6.24472e-07 | 3 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_bicgstab_jacobi_f32 | success | no | 81.0769 | 0.408559 | 79 | 7.87466e-07 | 4.1308e-05 | 3 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_cg_none_f32 | success | yes | 57.5602 | 0 | 104 | 8.53145e-07 | 7.39298e-07 | 3 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_pcg_jacobi_f32 | success | no | 58.2409 | 0.0118255 | 104 | 8.53184e-07 | 4.99677e-07 | 3 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_bicgstab_jacobi_f32 | success | no | 17.0543 | 0.384062 | 18 | 7.84133e-07 | 6.92337e-07 | 3 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_cg_none_f32 | success | no | 13.1404 | 0.0664246 | 26 | 4.34768e-07 | 2.22448e-07 | 3 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_pcg_jacobi_f32 | success | yes | 12.3219 | 0 | 26 | 4.34633e-07 | 3.25636e-07 | 3 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_bicgstab_jacobi_f32 | success | no | 26.1541 | 0.44001 | 25 | 5.06173e-07 | 1.78575e-06 | 3 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_cg_none_f32 | success | yes | 18.1625 | 0 | 35 | 6.14577e-07 | 2.50771e-07 | 3 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_pcg_jacobi_f32 | success | no | 19.1958 | 0.0568941 | 35 | 6.14345e-07 | 4.11375e-07 | 3 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_bicgstab_jacobi_f32 | success | no | 68.869 | 0.240321 | 71 | 9.63736e-07 | 1.76366e-05 | 3 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_cg_none_f32 | success | no | 57.6753 | 0.0387233 | 108 | 8.09603e-07 | 8.8221e-07 | 3 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_pcg_jacobi_f32 | success | yes | 55.5251 | 0 | 107 | 9.99853e-07 | 1.07736e-06 | 3 |
