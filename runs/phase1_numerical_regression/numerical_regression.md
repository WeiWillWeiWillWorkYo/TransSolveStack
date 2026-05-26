# Numerical Regression

- records: `21`
- passed: `21`
- failed: `0`

## Cases

- pass: synthetic:poisson_2d_stencil:32x32:float32 / taichi_pcg_jacobi_f32 res=8.013951093414108e-07 err=4.102938405042675e-07 iters=53
- pass: synthetic:poisson_2d_stencil:32x32:float32 / taichi_cg_none_f32 res=8.033326888792124e-07 err=3.440855667351031e-07 iters=53
- pass: synthetic:poisson_2d_stencil:32x32:float32 / taichi_bicgstab_jacobi_f32 res=5.021938269546062e-07 err=8.08765971864649e-06 iters=44
- pass: synthetic:poisson_2d_stencil:64x64:float32 / taichi_pcg_jacobi_f32 res=8.51449515876856e-07 err=4.7818857128685e-07 iters=104
- pass: synthetic:poisson_2d_stencil:64x64:float32 / taichi_cg_none_f32 res=8.539261423826615e-07 err=5.009970130657688e-07 iters=104
- pass: synthetic:poisson_2d_stencil:64x64:float32 / taichi_bicgstab_jacobi_f32 res=9.723291515752736e-07 err=1.663359430552477e-05 iters=81
- pass: synthetic:anisotropic_diffusion_2d:32x32:float32 / taichi_pcg_jacobi_f32 res=8.143684234731121e-07 err=6.12646047570045e-07 iters=80
- pass: synthetic:anisotropic_diffusion_2d:32x32:float32 / taichi_cg_none_f32 res=8.141236135593142e-07 err=6.457793687257099e-07 iters=80
- pass: synthetic:anisotropic_diffusion_2d:32x32:float32 / taichi_bicgstab_jacobi_f32 res=3.7680783756689733e-07 err=6.88845714705829e-06 iters=64
- pass: synthetic:anisotropic_diffusion_2d:64x64:float32 / taichi_pcg_jacobi_f32 res=9.426485163520828e-07 err=1.3833595415837275e-06 iters=156
- pass: synthetic:anisotropic_diffusion_2d:64x64:float32 / taichi_cg_none_f32 res=9.426103054862897e-07 err=1.3286937999930076e-06 iters=156
- pass: synthetic:anisotropic_diffusion_2d:64x64:float32 / taichi_bicgstab_jacobi_f32 res=8.487162461565792e-07 err=4.2471912977772214e-05 iters=115
- pass: synthetic:strong_anisotropic_diffusion_2d:32x32:float32 / taichi_pcg_jacobi_f32 res=8.09823387862436e-07 err=9.360009644297503e-07 iters=108
- pass: synthetic:strong_anisotropic_diffusion_2d:32x32:float32 / taichi_cg_none_f32 res=8.096584979969986e-07 err=8.983261610225201e-07 iters=108
- pass: synthetic:strong_anisotropic_diffusion_2d:32x32:float32 / taichi_bicgstab_jacobi_f32 res=9.968129275204381e-07 err=2.1809949315999624e-05 iters=79
- pass: synthetic:poisson_3d_stencil:12x12x12:float32 / taichi_pcg_jacobi_f32 res=4.3448363595880674e-07 err=3.917968617829153e-07 iters=26
- pass: synthetic:poisson_3d_stencil:12x12x12:float32 / taichi_cg_none_f32 res=4.3562864384584794e-07 err=1.8439524970407312e-07 iters=26
- pass: synthetic:poisson_3d_stencil:12x12x12:float32 / taichi_bicgstab_jacobi_f32 res=8.366309997629105e-07 err=7.653132198954188e-07 iters=18
- pass: synthetic:poisson_3d_stencil:16x16x16:float32 / taichi_pcg_jacobi_f32 res=6.140190028735674e-07 err=4.197941024806469e-07 iters=35
- pass: synthetic:poisson_3d_stencil:16x16x16:float32 / taichi_cg_none_f32 res=6.161052789193321e-07 err=2.3980606871046684e-07 iters=35
- pass: synthetic:poisson_3d_stencil:16x16x16:float32 / taichi_bicgstab_jacobi_f32 res=8.391229490230421e-07 err=3.6078033759791037e-06 iters=25
