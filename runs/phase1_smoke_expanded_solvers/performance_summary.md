# Candidate Performance Summary

- num_records: `35`
- num_success: `28`
- num_failed: `7`
- num_dry_run: `0`
- best_candidate_id: `taichi_pcg_jacobi_f32`
- best_total_time_ms: `12.75030616670847`
- num_systems: `7`

## Status Counts

- success: `28`
- failed: `7`

## Oracle Plans

- synthetic:anisotropic_diffusion_2d:32x32:float32 / default_f32: `taichi_cg_none_f32` (41.6685058735311)
- synthetic:anisotropic_diffusion_2d:64x64:float32 / default_f32: `taichi_cg_none_f32` (82.27352285757661)
- synthetic:poisson_2d_stencil:32x32:float32 / default_f32: `taichi_cg_none_f32` (26.991874910891056)
- synthetic:poisson_2d_stencil:64x64:float32 / default_f32: `taichi_cg_none_f32` (55.070784874260426)
- synthetic:poisson_3d_stencil:12x12x12:float32 / default_f32: `taichi_pcg_jacobi_f32` (12.75030616670847)
- synthetic:poisson_3d_stencil:16x16x16:float32 / default_f32: `taichi_cg_none_f32` (17.341171856969595)
- synthetic:strong_anisotropic_diffusion_2d:32x32:float32 / default_f32: `taichi_chebyshev_jacobi_f32` (47.77232790365815)

## Raw Summary

```text
num_records: 35
status_counts: {'success': 28, 'failed': 7}
backend_counts: {'taichi_gpu': 35}
system_counts: {'synthetic:poisson_2d_stencil:32x32:float32': 5, 'synthetic:poisson_2d_stencil:64x64:float32': 5, 'synthetic:anisotropic_diffusion_2d:32x32:float32': 5, 'synthetic:anisotropic_diffusion_2d:64x64:float32': 5, 'synthetic:strong_anisotropic_diffusion_2d:32x32:float32': 5, 'synthetic:poisson_3d_stencil:12x12x12:float32': 5, 'synthetic:poisson_3d_stencil:16x16x16:float32': 5}
context_counts: {'default_f32': 35}
candidate_counts: {'taichi_pcg_jacobi_f32': 7, 'taichi_cg_none_f32': 7, 'taichi_bicgstab_jacobi_f32': 7, 'taichi_richardson_jacobi_f32': 7, 'taichi_chebyshev_jacobi_f32': 7}
num_success: 28
num_failed: 7
num_dry_run: 0
best_total_time_ms: 12.75030616670847
best_candidate_id: taichi_pcg_jacobi_f32
metadata: {}
```
