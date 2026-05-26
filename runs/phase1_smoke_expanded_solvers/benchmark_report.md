# Benchmark Evaluation Report

- rows: `35`
- systems: `7`
- candidates: `5`

## Per-System Candidate Table

| system | candidate | status | oracle | time_ms | regret | iters | residual | rel_error | repeats |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_bicgstab_jacobi_f32 | success | no | 61.0476 | 0.465078 | 62 | 9.69985e-07 | 1.33234e-05 | 3 |
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_cg_none_f32 | success | yes | 41.6685 | 0 | 80 | 8.10043e-07 | 7.28616e-07 | 3 |
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_chebyshev_jacobi_f32 | success | no | 48.2818 | 0.158712 | 148 | 9.46844e-07 | 1.82188e-06 | 3 |
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_pcg_jacobi_f32 | success | no | 43.5736 | 0.0457191 | 80 | 8.14599e-07 | 6.77248e-07 | 3 |
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_richardson_jacobi_f32 | failed | no | 82.1953 |  | 300 | 0.0100581 | 0.246699 | 3 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_bicgstab_jacobi_f32 | success | no | 127.643 | 0.551444 | 121 | 4.35257e-07 | 1.70681e-05 | 3 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_cg_none_f32 | success | yes | 82.2735 | 0 | 156 | 9.42677e-07 | 1.30819e-06 | 3 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_chebyshev_jacobi_f32 | success | no | 96.459 | 0.172419 | 292 | 9.27588e-07 | 1.51661e-06 | 3 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_pcg_jacobi_f32 | success | no | 86.7485 | 0.054392 | 156 | 9.42557e-07 | 1.66891e-06 | 3 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_richardson_jacobi_f32 | failed | no | 79.5383 |  | 300 | 0.0106024 | 0.621719 | 3 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_bicgstab_jacobi_f32 | success | no | 92.2278 | 2.41687 | 44 | 7.96295e-07 | 1.38596e-05 | 3 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_cg_none_f32 | success | yes | 26.9919 | 0 | 53 | 8.02823e-07 | 4.00677e-07 | 3 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_chebyshev_jacobi_f32 | success | no | 57.832 | 1.14257 | 150 | 8.70048e-07 | 1.09353e-06 | 3 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_pcg_jacobi_f32 | success | no | 28.9846 | 0.0738272 | 53 | 8.21569e-07 | 4.38771e-07 | 3 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_richardson_jacobi_f32 | failed | no | 82.975 |  | 300 | 0.0121863 | 0.245163 | 3 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_bicgstab_jacobi_f32 | success | no | 101.283 | 0.839151 | 85 | 6.23166e-07 | 3.28837e-05 | 3 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_cg_none_f32 | success | yes | 55.0708 | 0 | 104 | 8.50671e-07 | 4.89969e-07 | 3 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_chebyshev_jacobi_f32 | success | no | 110.603 | 1.00838 | 294 | 8.49542e-07 | 1.17988e-06 | 3 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_pcg_jacobi_f32 | success | no | 58.0154 | 0.0534704 | 104 | 8.49442e-07 | 5.16311e-07 | 3 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_richardson_jacobi_f32 | failed | no | 92.191 |  | 300 | 0.0133215 | 0.605605 | 3 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_bicgstab_jacobi_f32 | success | no | 16.4135 | 0.287303 | 18 | 6.27677e-07 | 9.32126e-07 | 3 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_cg_none_f32 | success | no | 13.0922 | 0.0268167 | 26 | 4.34546e-07 | 3.41925e-07 | 3 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_chebyshev_jacobi_f32 | success | no | 52.1506 | 3.09014 | 58 | 9.57074e-07 | 1.24405e-06 | 3 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_pcg_jacobi_f32 | success | yes | 12.7503 | 0 | 26 | 4.35134e-07 | 3.2133e-07 | 3 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_richardson_jacobi_f32 | failed | no | 77.3275 |  | 300 | 6.10572e-05 | 0.000285605 | 3 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_bicgstab_jacobi_f32 | success | no | 24.0404 | 0.386319 | 25 | 5.55866e-07 | 2.0164e-06 | 3 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_cg_none_f32 | success | yes | 17.3412 | 0 | 35 | 6.13995e-07 | 2.17114e-07 | 3 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_chebyshev_jacobi_f32 | success | no | 27.9211 | 0.610103 | 76 | 9.83904e-07 | 1.17664e-06 | 3 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_pcg_jacobi_f32 | success | no | 18.0582 | 0.0413493 | 35 | 6.16132e-07 | 2.96225e-07 | 3 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_richardson_jacobi_f32 | failed | no | 132.059 |  | 300 | 0.00115016 | 0.00770745 | 3 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_bicgstab_jacobi_f32 | success | no | 83.9297 | 0.756869 | 82 | 6.75235e-07 | 1.33079e-05 | 3 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_cg_none_f32 | success | no | 56.8921 | 0.1909 | 108 | 8.0963e-07 | 9.11304e-07 | 3 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_chebyshev_jacobi_f32 | success | yes | 47.7723 | 0 | 149 | 8.90405e-07 | 1.5892e-06 | 3 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_pcg_jacobi_f32 | success | no | 58.199 | 0.218257 | 108 | 8.09568e-07 | 8.77116e-07 | 3 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_richardson_jacobi_f32 | failed | no | 76.9511 |  | 300 | 0.00965605 | 0.253369 | 3 |
