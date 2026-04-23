# gswarp

**Pure-Python NVIDIA Warp backend for 3D Gaussian Splatting.**

gswarp reimplements the three core CUDA modules used in 3DGS training — the rasterizer, SSIM loss, and KNN initialization — using [NVIDIA Warp](https://nvidia.github.io/warp/). No C++/CUDA compilation required; install via pip and drop it in as a replacement for the original CUDA packages.

> Full documentation and source: **https://github.com/fancifulland2718/gswarp**

---

## Three Replacement Modules

| Module | Replaces | Import Path |
|--------|----------|-------------|
| **Rasterizer** | `diff_gaussian_rasterization` | `gswarp` |
| **SSIM** | `fused_ssim` | `gswarp.fused_ssim` |
| **KNN** | `simple_knn` | `gswarp.knn` |

---

## Requirements

- Python 3.10+
- NVIDIA GPU with compute capability ≥ 7.0 (Volta)
- PyTorch 1.13+ (with CUDA support)
- NVIDIA Warp 1.8.0+

---

## Installation

```bash
pip install gswarp
```

`warp-lang` is installed automatically as a dependency. No compilation steps needed after installation.

---

## Quick Start

### Standard Rasterizer

Replace the original import in `gaussian_renderer/__init__.py`:

```python
# before
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

# after
from gswarp import GaussianRasterizationSettings, GaussianRasterizer
```

`GaussianRasterizer.forward()` returns a three-element tuple `(color, radii, meta)`. The `meta` object is a `RasterizerMeta` NamedTuple with five auxiliary fields:

```python
color, radii, meta = rasterizer(means3D=..., means2D=..., ...)

# meta fields:
#   meta.depth          (1, H, W)  — per-pixel accumulated depth
#   meta.alpha          (1, H, W)  — per-pixel accumulated opacity
#   meta.proj_2D        (N, 2)     — 2D projected positions
#   meta.conic_2D       (N, 3)     — 2D conic representation
#   meta.conic_2D_inv   (N, 3)     — inverse 2D conic

# If you only need color and radii:
color, radii, _ = rasterizer(means3D=..., means2D=..., ...)
```

**Optional runtime tuning** (call once before the training loop):

```python
from gswarp import initialize_runtime_tuning, set_binning_sort_mode

initialize_runtime_tuning(device="cuda:0", verbose=True)
set_binning_sort_mode("warp_depth_stable_tile")  # recommended for large scenes
```

### Flow Backend (optical-flow pipelines)

A dedicated backend adds per-pixel top-K contributor tracking:

```python
from gswarp.rasterizer_flow import GaussianRasterizationSettings, GaussianRasterizer

raster_settings = GaussianRasterizationSettings(
    ...,
    enable_flow_grad=True,   # default True — enables flow gradients
    compute_flow_aux=True,   # default None (runtime True) — fills top-K outputs
)

color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, \
    gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(...)
```

| Aux output | Shape | dtype | Meaning |
|------------|-------|-------|---------|
| `gs_per_pixel` | `(K, H, W)` | `int32` | Index of each top-K Gaussian per pixel (`-1` = empty) |
| `weight_per_gs_pixel` | `(K, H, W)` | `float32` | Alpha-compositing weight |
| `x_mu` | `(2, K, H, W)` | `float32` | Offset `(dx, dy)` from projected center to pixel |

`K` defaults to 20; adjust with `set_flow_topk(k)` from `gswarp.rasterizer_flow`.

### SSIM and KNN

```python
# SSIM — identical signature to fused_ssim
from gswarp.fused_ssim import fused_ssim
loss_ssim = fused_ssim(img1, img2, padding="same", train=True)

# KNN — identical signature to simple_knn
from gswarp.knn import distCUDA2
dist2 = distCUDA2(points)  # points: (N, 3) float32 CUDA tensor
```

---

## Performance

Full 30K-step training, RTX 5090D V2 (sm_120), PyTorch 2.11.0+cu130, Warp 1.12.0. All three modules use the Warp backend.

**NeRF Synthetic (8 scenes):** average **×1.08** speedup over CUDA baseline.  
**Tanks & Temples / Deep Blending (4 large scenes):** average **×1.03** speedup.

Sample results:

| Dataset | CUDA (it/s) | Warp (it/s) | Speedup |
|---------|------------|------------|---------|
| chair | 103.6 | 113.1 | ×1.09 |
| drums | 103.0 | 115.3 | ×1.12 |
| hotdog | 144.5 | 156.5 | ×1.08 |
| mic | 95.4 | 105.1 | ×1.10 |
| truck | 39.4 | 40.1 | ×1.02 |
| drjohnson | 30.8 | 32.0 | ×1.04 |

---

## Quality

NeRF Synthetic 8-scene average after 30K steps:

| Metric | CUDA | Warp | Δ |
|--------|------|------|---|
| PSNR (dB) | 33.31 | 33.33 | +0.02 |
| SSIM | 0.9692 | 0.9693 | +0.0001 |
| LPIPS | 0.0303 | 0.0302 | −0.0001 |

Per-scene PSNR differences are within ±0.25 dB. Training quality is equivalent to the CUDA baseline across all tested scenes.

---

## Links

- **GitHub repository & full documentation:** https://github.com/fancifulland2718/gswarp
- **License:** Apache 2.0

---

## Acknowledgements

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (INRIA / MPII)
- [fused-ssim](https://github.com/rahul-goel/fused-ssim) (Rahul Goel et al.)
- [simple-knn](https://github.com/camenduru/simple-knn) (graphdeco-inria)
- [NVIDIA Warp](https://nvidia.github.io/warp/)
