# Policy Selection Report

- rows: `7`
- systems: `7`

| system | context | candidate | reason | oracle | time_ms | regret | fallbacks |
|---|---|---|---|---:|---:|---:|---:|
| synthetic:anisotropic_diffusion_2d:32x32:float32 | default_f32 | taichi_cg_none_f32 | profiled_success | yes | 39.978 | 0 | 2 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | default_f32 | taichi_pcg_jacobi_f32 | profiled_success | yes | 77.5586 | 0 | 2 |
| synthetic:poisson_2d_stencil:32x32:float32 | default_f32 | taichi_pcg_jacobi_f32 | profiled_success | yes | 28.8392 | 0 | 2 |
| synthetic:poisson_2d_stencil:64x64:float32 | default_f32 | taichi_cg_none_f32 | profiled_success | yes | 57.5602 | 0 | 2 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | default_f32 | taichi_pcg_jacobi_f32 | profiled_success | yes | 12.3219 | 0 | 2 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | default_f32 | taichi_cg_none_f32 | profiled_success | yes | 18.1625 | 0 | 2 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | default_f32 | taichi_pcg_jacobi_f32 | profiled_success | yes | 55.5251 | 0 | 2 |
