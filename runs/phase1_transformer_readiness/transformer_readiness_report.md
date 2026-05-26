# Transformer Readiness Export

- schema_version: `phase1_policy_features_v1`
- rows: `21`
- systems: `7`
- candidates: `3`
- oracle_rows: `7`

| system | candidate | oracle | regret | total_ms |
|---|---|---:|---:|---:|
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_bicgstab_jacobi_f32 | no | 0.499966 | 59.9656 |
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_cg_none_f32 | yes | 0 | 39.978 |
| synthetic:anisotropic_diffusion_2d:32x32:float32 | taichi_pcg_jacobi_f32 | no | 2.79455 | 151.699 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_bicgstab_jacobi_f32 | no | 0.668293 | 129.39 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_cg_none_f32 | no | 0.0861674 | 84.2416 |
| synthetic:anisotropic_diffusion_2d:64x64:float32 | taichi_pcg_jacobi_f32 | yes | 0 | 77.5586 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_bicgstab_jacobi_f32 | no | 1.32633 | 67.0894 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_cg_none_f32 | no | 1.8318 | 81.6669 |
| synthetic:poisson_2d_stencil:32x32:float32 | taichi_pcg_jacobi_f32 | yes | 0 | 28.8392 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_bicgstab_jacobi_f32 | no | 0.408559 | 81.0769 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_cg_none_f32 | yes | 0 | 57.5602 |
| synthetic:poisson_2d_stencil:64x64:float32 | taichi_pcg_jacobi_f32 | no | 0.0118255 | 58.2409 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_bicgstab_jacobi_f32 | no | 0.384062 | 17.0543 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_cg_none_f32 | no | 0.0664246 | 13.1404 |
| synthetic:poisson_3d_stencil:12x12x12:float32 | taichi_pcg_jacobi_f32 | yes | 0 | 12.3219 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_bicgstab_jacobi_f32 | no | 0.44001 | 26.1541 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_cg_none_f32 | yes | 0 | 18.1625 |
| synthetic:poisson_3d_stencil:16x16x16:float32 | taichi_pcg_jacobi_f32 | no | 0.0568941 | 19.1958 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_bicgstab_jacobi_f32 | no | 0.240321 | 68.869 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_cg_none_f32 | no | 0.0387233 | 57.6753 |
| synthetic:strong_anisotropic_diffusion_2d:32x32:float32 | taichi_pcg_jacobi_f32 | yes | 0 | 55.5251 |
