# Candidate Performance Summary

- num_records: `21`
- num_success: `21`
- num_failed: `0`
- num_dry_run: `0`
- best_candidate_id: `taichi_pcg_jacobi_f32`
- best_total_time_ms: `12.321929913014174`
- num_systems: `7`

## Status Counts

- success: `21`

## Oracle Plans

- synthetic:anisotropic_diffusion_2d:32x32:float32 / default_f32: `taichi_cg_none_f32` (39.978009182959795)
- synthetic:anisotropic_diffusion_2d:64x64:float32 / default_f32: `taichi_pcg_jacobi_f32` (77.55860686302185)
- synthetic:poisson_2d_stencil:32x32:float32 / default_f32: `taichi_pcg_jacobi_f32` (28.839170932769775)
- synthetic:poisson_2d_stencil:64x64:float32 / default_f32: `taichi_cg_none_f32` (57.56019614636898)
- synthetic:poisson_3d_stencil:12x12x12:float32 / default_f32: `taichi_pcg_jacobi_f32` (12.321929913014174)
- synthetic:poisson_3d_stencil:16x16x16:float32 / default_f32: `taichi_cg_none_f32` (18.162461929023266)
- synthetic:strong_anisotropic_diffusion_2d:32x32:float32 / default_f32: `taichi_pcg_jacobi_f32` (55.52513478323817)

## Raw Summary

```text
num_records: 21
status_counts: {'success': 21}
backend_counts: {'taichi_gpu': 21}
system_counts: {'synthetic:poisson_2d_stencil:32x32:float32': 3, 'synthetic:poisson_2d_stencil:64x64:float32': 3, 'synthetic:anisotropic_diffusion_2d:32x32:float32': 3, 'synthetic:anisotropic_diffusion_2d:64x64:float32': 3, 'synthetic:strong_anisotropic_diffusion_2d:32x32:float32': 3, 'synthetic:poisson_3d_stencil:12x12x12:float32': 3, 'synthetic:poisson_3d_stencil:16x16x16:float32': 3}
context_counts: {'default_f32': 21}
candidate_counts: {'taichi_pcg_jacobi_f32': 7, 'taichi_cg_none_f32': 7, 'taichi_bicgstab_jacobi_f32': 7}
num_success: 21
num_failed: 0
num_dry_run: 0
best_total_time_ms: 12.321929913014174
best_candidate_id: taichi_pcg_jacobi_f32
metadata: {}
```
