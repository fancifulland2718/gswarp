# gswarp

**English** · [中文](README_zh.md)

gswarp is a pure-Python **NVIDIA Warp** backend for 3D Gaussian Splatting, reimplementing the three core CUDA modules used in 3DGS training — the rasterizer, SSIM loss, and KNN initialization. No C++/CUDA compilation required; install via pip and swap out the original CUDA implementations directly.

> **License**: [Apache License 2.0](LICENSE). Third-party attributions in [NOTICE](NOTICE).

---

## Table of Contents

- [Three Replacement Modules](#three-replacement-modules)
- [Requirements](#requirements)
- [Installation](#installation)
- [Replacing CUDA Backends in a 3DGS Project](#replacing-cuda-backends-in-a-3dgs-project)
  - [Rasterizer](#rasterizer)
  - [SSIM](#ssim)
  - [KNN](#knn)
  - [Recommended: Replace All at Once](#recommended-replace-all-at-once)
- [Performance](#performance)
- [Quality Metrics](#quality-metrics)
- [Detailed Documentation](#detailed-documentation)
- [Acknowledgements](#acknowledgements)

---

## Three Replacement Modules

| Module | Replaces | Import Path | Notes |
|--------|----------|-------------|-------|
| **Rasterizer** | `diff_gaussian_rasterization` | `gswarp` | Full differentiable Gaussian rasterization + auto-tuning |
| **SSIM** | `fused_ssim` | `gswarp.fused_ssim` | Separable Gaussian convolution with launch caching |
| **KNN** | `simple_knn` | `gswarp.knn` | Morton-sort + bounding-box pruning 3-NN |

---

## Requirements

| Component | Minimum |
|-----------|---------|
| Python | 3.10+ |
| NVIDIA GPU | Compute capability ≥ 7.0 (Volta) |
| PyTorch | 1.13+ (with CUDA support) |
| NVIDIA Warp | 1.8.0+ |

---

## Installation

```bash
pip install gswarp
```

This installs `warp-lang` automatically via package dependencies. If you want to pin the Warp version explicitly, use:

```bash
pip install "warp-lang>=1.8.0" gswarp
```

Or install from source:

```bash
git clone https://github.com/fancifulland2718/gswarp.git
cd gswarp
pip install .
```

No compilation steps are needed after installation. The first call to any Warp kernel triggers JIT compilation (a few seconds); subsequent runs use the cache.

---

## Replacing CUDA Backends in a 3DGS Project

The examples below follow the [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) reference implementation.

### Rasterizer

Original (`gaussian_renderer/__init__.py`):

```python
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
```

Replace with:

```python
from gswarp import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
```

`GaussianRasterizationSettings` is a `NamedTuple` with the same fields as the original (Warp-specific fields have defaults). `GaussianRasterizer.forward()` **returns a three-element tuple `(color, radii, meta)`**, where `meta` is a `RasterizerMeta` NamedTuple carrying five auxiliary outputs:

| `meta` field | Meaning |
|--------------|---------|
| `meta.depth` | Per-pixel accumulated depth |
| `meta.alpha` | Per-pixel accumulated opacity |
| `meta.proj_2D` | 2D projected positions of each Gaussian |
| `meta.conic_2D` | Conic representation of the 2D covariance |
| `meta.conic_2D_inv` | Inverse of the 2D conic |

```python
# Original CUDA (two outputs only):
color, radii = rasterizer(means3D=..., means2D=..., ...)

# gswarp (three-element tuple):
color, radii, meta = rasterizer(means3D=..., means2D=..., ...)

# Accessing meta fields:
depth        = meta.depth          # (1, H, W)
alpha        = meta.alpha          # (1, H, W)
proj_2D      = meta.proj_2D        # (N, 2)
conic_2D     = meta.conic_2D       # (N, 3)
conic_2D_inv = meta.conic_2D_inv   # (N, 3)

# If you only need color and radii, discard meta:
color, radii, _ = rasterizer(means3D=..., means2D=..., ...)
```

### Flow Backend (optional)

For optical-flow pipelines that need per-pixel top-K contributor information, use the dedicated flow backend `gswarp.rasterizer_flow` instead.

**Key differences from the standard backend:**

**1. `GaussianRasterizationSettings` adds two flow-specific fields**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_flow_grad` | `bool` | `True` | Enable flow gradient computation (absent in the standard backend) |
| `compute_flow_aux` | `bool \| None` | `None` (runtime default: `True`) | Whether to fill the top-K auxiliary outputs |

**2. `forward()` returns a flat 10-tuple — no `meta`**

```python
from gswarp.rasterizer_flow import GaussianRasterizationSettings, GaussianRasterizer

raster_settings = GaussianRasterizationSettings(
    ...,                    # same shared fields as the standard backend
    enable_flow_grad=True,  # default True
    compute_flow_aux=True,  # default None (runtime True)
)

color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, \
    gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(...)
```

Shapes and dtypes of the three auxiliary outputs:

| Output | Shape | dtype | Meaning |
|--------|-------|-------|---------|
| `gs_per_pixel` | `(K, H, W)` | `int32` | Index of the top-K contributing Gaussian per pixel (`-1` for unfilled slots) |
| `weight_per_gs_pixel` | `(K, H, W)` | `float32` | Alpha-compositing weight of each contributing Gaussian |
| `x_mu` | `(2, K, H, W)` | `float32` | Offset `(dx, dy)` from each Gaussian's projected center to the pixel center |

`K` defaults to `20` and can be changed at runtime:

```python
from gswarp.rasterizer_flow import set_flow_topk, set_compute_flow_aux

set_flow_topk(32)            # change K (clear launch cache; call before first render)
set_compute_flow_aux(False)  # temporarily disable aux outputs to save VRAM
```

**Optional runtime configuration** (call once before the training loop):

```python
from gswarp import initialize_runtime_tuning, set_binning_sort_mode

# Detect GPU and select optimal block_dim automatically (recommended)
initialize_runtime_tuning(device="cuda:0", verbose=True)

# Choose a sort mode (default warp_depth_stable_tile is usually best)
set_binning_sort_mode("warp_depth_stable_tile")  # recommended for large scenes
# set_binning_sort_mode("warp_radix")            # alternative
# set_binning_sort_mode("torch")                 # fallback
```

### SSIM

Original (`train.py`):

```python
from fused_ssim import fused_ssim
```

Replace with:

```python
from gswarp.fused_ssim import fused_ssim
```

The function signature is identical:

```python
loss_ssim = fused_ssim(img1, img2, padding="same", train=True)
```

### KNN

Original (`scene/gaussian_model.py`):

```python
from simple_knn._C import distCUDA2
```

Replace with:

```python
from gswarp.knn import distCUDA2
```

The function signature is identical:

```python
dist2 = distCUDA2(points)  # points: (N, 3) float32 CUDA tensor
```

### Recommended: Replace All at Once

Add the following near the top of `train.py`:

```python
try:
    from gswarp import GaussianRasterizationSettings, GaussianRasterizer
    from gswarp.fused_ssim import fused_ssim
    from gswarp.knn import distCUDA2
    GSWARP_AVAILABLE = True
except ImportError:
    GSWARP_AVAILABLE = False
```

Then switch backends at each usage site using the `GSWARP_AVAILABLE` flag. A reference integration is available in [gaussian-splatting/train.py](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py).

---

## Performance

Results from full 30K-step training on 12 standard 3DGS datasets. Hardware: RTX 5090D V2 (sm_120, 24 GiB), Python 3.14, PyTorch 2.11.0+cu130, Warp 1.12.0. **All three modules use the Warp backend**, with Python-layer overhead optimizations applied.

| Dataset | CUDA (it/s) | Warp (it/s) | Speedup |
|---------|------------|------------|---------|
| chair | 103.6 | 113.1 | ×1.09 |
| drums | 103.0 | 115.3 | ×1.12 |
| ficus | 139.5 | 148.2 | ×1.06 |
| hotdog | 144.5 | 156.5 | ×1.08 |
| lego | 117.5 | 126.5 | ×1.08 |
| materials | 134.0 | 144.9 | ×1.08 |
| mic | 95.4 | 105.1 | ×1.10 |
| ship | 107.0 | 113.2 | ×1.06 |
| train | 55.6 | 58.3 | ×1.05 |
| truck | 39.4 | 40.1 | ×1.02 |
| drjohnson | 30.8 | 32.0 | ×1.04 |
| playroom | 46.9 | 47.5 | ×1.01 |

**NeRF Synthetic (8 scenes)**: average ×1.08 speedup. **Tanks & Temples / Deep Blending (4 large scenes)**: average ×1.03 speedup. The smaller gains on large scenes (drjohnson, playroom) are explained by the higher Gaussian counts diluting the rasterizer kernel advantage — see the [rasterizer documentation](docs/rasterizer.md) for a per-phase breakdown.

---

## Quality Metrics

Test-set evaluation after 30K training steps:

**NeRF Synthetic (8-scene average)**

| Metric | CUDA | Warp | Δ |
|--------|------|------|---|
| PSNR (dB) | 33.31 | 33.33 | +0.02 |
| SSIM | 0.9692 | 0.9693 | +0.0001 |
| LPIPS | 0.0303 | 0.0302 | −0.0001 |

**Tanks & Temples (2-scene average)**

| Metric | CUDA | Warp | Δ |
|--------|------|------|---|
| PSNR (dB) | 23.74 | 23.79 | +0.04 |
| SSIM | 0.8512 | 0.8515 | +0.0003 |
| LPIPS | 0.1711 | 0.1707 | −0.0004 |

**Deep Blending (2-scene average)**

| Metric | CUDA | Warp | Δ |
|--------|------|------|---|
| PSNR (dB) | 29.77 | 30.01 | +0.04 |
| SSIM | 0.9062 | 0.9063 | +0.0001 |
| LPIPS | 0.2390 | 0.2388 | −0.0002 |

Per-scene PSNR differences are within ±0.25 dB; SSIM differences are < 0.001. The Warp backend produces training quality equivalent to the CUDA baseline across all tested scenes.

---

## Detailed Documentation

| Document | Contents |
|----------|----------|
| [docs/rasterizer.md](docs/rasterizer.md) | Architecture, CUDA implementation differences, micro-benchmarks, correctness, known limitations |
| [docs/ssim.md](docs/ssim.md) | SSIM kernel optimizations, performance analysis, correctness |
| [docs/knn.md](docs/knn.md) | KNN algorithm, Morton sorting, performance analysis |

---

## Acknowledgements

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (INRIA / MPII)
- [fused-ssim](https://github.com/rahul-goel/fused-ssim) (Rahul Goel et al.)
- [simple-knn](https://github.com/camenduru/simple-knn) (graphdeco-inria)
- [Fast Converging 3DGS](https://arxiv.org/abs/2601.19489) (Zhang et al., 2025) — inspiration for compact AABB culling
- [NVIDIA Warp](https://nvidia.github.io/warp/)

