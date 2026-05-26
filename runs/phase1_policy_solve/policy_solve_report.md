# Policy-Driven Solve Report

- rows: `7`
- success: `7`

| system | context | candidate | status | reason | ms | iters | residual | rel_error | fallbacks |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| synthetic:poisson_2d_stencil:32x32:float32 | default_f32 | taichi_pcg_jacobi_f32 | success | profiled_success | 365.241 | 53 | 8.35059e-07 | 5.09875e-07 | 2 |
| synthetic:poisson_2d_stencil:64x64:float32 | default_f32 | taichi_cg_none_f32 | success | profiled_success | 438.923 | 104 | 8.46922e-07 | 6.57624e-07 | 2 |
| synthetic:anisotropic_diffusion_2d:32x32:float32 | default_f32 | taichi_cg_none_f32 | success | profiled_success | 647.911 | 80 | 8.12183e-07 | 6.35484e-07 | 2 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | default_f32 | taichi_pcg_jacobi_f32 | success | profiled_success | 477.201 | 156 | 9.38611e-07 | 1.63572e-06 | 2 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | default_f32 | taichi_pcg_jacobi_f32 | success | profiled_success | 347.839 | 108 | 8.09745e-07 | 9.78169e-07 | 2 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | default_f32 | taichi_pcg_jacobi_f32 | success | profiled_success | 373.359 | 26 | 4.34557e-07 | 3.47238e-07 | 2 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | default_f32 | taichi_cg_none_f32 | success | profiled_success | 315.455 | 35 | 6.14778e-07 | 3.06038e-07 | 2 |
