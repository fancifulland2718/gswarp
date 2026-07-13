# gswarp

**English** · [中文](README_zh.md)

gswarp is a pure-Python **NVIDIA Warp** backend for 3D Gaussian Splatting, reimplementing the three core CUDA modules used in 3DGS training — the rasterizer, SSIM loss, and KNN initialization. No C++/CUDA compilation required; install via pip and swap out the original CUDA implementations directly.

> **License**: [Apache License 2.0](LICENSE). Third-party attributions in [NOTICE](NOTICE).

---

## Table of Contents

- [Three Replacement Modules](#three-replacement-modules)
- [Current Architecture](#current-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Replacing CUDA Backends in a 3DGS Project](#replacing-cuda-backends-in-a-3dgs-project)
  - [Rasterizer](#rasterizer)
  - [SSIM](#ssim)
  - [KNN](#knn)
  - [Recommended: Replace All at Once](#recommended-replace-all-at-once)
- [Benchmarks](#benchmarks)
- [Detailed Documentation](#detailed-documentation)
- [Acknowledgements](#acknowledgements)

---

## Three Replacement Modules

| Module | Replaces | Import Path | Notes |
|--------|----------|-------------|-------|
| **Rasterizer** | `diff_gaussian_rasterization` | `gswarp` | Typed staged rasterization with manual backward and stream-safe caches |
| **SSIM** | `fused_ssim` | `gswarp.fused_ssim` | Separable Gaussian convolution with graph-owned reusable plans |
| **KNN** | `simple_knn` | `gswarp.knn` | Morton-sort + bounding-box pruning 3-NN on the active PyTorch stream |

---

## Current Architecture

The public modules remain small compatibility layers. Internally, the rasterizer resolves a cached immutable method plan and executes explicit stages:

```text
public API -> validation/autograd -> method plan
           -> preprocess -> features -> binning -> render -> typed forward state
           -> manual backward
```

Standard 3DGS and the optional flow backend share preprocessing, binning, runtime options, stream interop, workspace management, and most backward operations. Method-specific backend modules only adapt stage inputs, outputs, auxiliary data, and retained backward state.

Every call snapshots its runtime options and binds Warp to the current PyTorch CUDA device and stream. Reusable workspaces and recorded launches are bounded by device/stream keys; forward-owned tensors remain attached to their autograd graph until backward releases them. `clear_warp_caches()` and `get_warp_cache_report()` provide explicit cache lifecycle control.

The stable backend targets the declared minimum Warp version. Optional advanced backends are selected only when both their minimum version and required Warp capabilities are available; no advanced backend is enabled in the current release.

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

`GaussianRasterizationSettings` is a `NamedTuple` containing the CUDA-compatible fields plus Warp-specific fields with defaults. `GaussianRasterizer.forward()` **returns a three-element tuple `(color, radii, meta)`**, where `meta` is a `RasterizerMeta` NamedTuple carrying five auxiliary outputs:

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

Use keyword arguments when migrating existing code. `shs`, `colors_precomp`, `scales`, `rotations`, and `cov3D_precomp` follow the CUDA rasterizer's mutually exclusive input rules; `dc` is accepted as a compatibility alias for `shs`. The current stable backend does not implement `prefiltered=True`, and accepts but does not apply the CUDA antialiasing path.

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

# Initialize Warp and record the per-device launch/tuning report
initialize_runtime_tuning(device="cuda:0", verbose=True)

# The stable default is a 32-bit depth sort followed by a stable tile sort
set_binning_sort_mode("warp_depth_stable_tile")
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

## Benchmarks

The current package artifact was retested from a locally built installation, not from a checkout import. The CUDA reference was the native `diff_gaussian_rasterization` extension; gswarp was installed into a separate target directory and its source hash was checked before training. This prevents either package from shadowing the other through `PYTHONPATH`.

**Environment.** NVIDIA GeForce RTX 5090 (32 GiB, sm_120), NVIDIA driver 610.62, Python 3.14.3, PyTorch 2.11.0+cu130, Warp 1.12.0, and gswarp 1.0.5 built from the current working tree. All runs use original 3DGS default optimization settings, `--data_device cpu`, default Adam, and 30,000 iterations. The Warp run selects the gswarp rasterizer, fused SSIM, and KNN; depth accumulation is disabled because this reference training loss does not consume depth.

The current matrix covers all three standard evaluation families: eight NeRF Synthetic scenes, two Tanks and Temples scenes, and two Deep Blending scenes. This is a completed 12-scene, 30K-iteration suite rather than a representative subset.

### End-to-End Training

Wall throughput includes the complete training loop, final evaluation, and checkpoint saving. Stable GPU throughput is derived from CUDA events over iterations 25,000-29,999, excluding final evaluation and saving. Peak memory is PyTorch peak allocated memory after the training script resets its counter.

| Scene | Family | CUDA min | Warp min | Wall ratio | CUDA stable it/s | Warp stable it/s | CUDA/Warp final Gaussians | CUDA/Warp peak MiB |
|-------|--------|---------:|---------:|-----------:|----------------:|----------------:|--------------------------:|-------------------:|
| chair | NeRF Synthetic | 6.95 | 6.69 | 1.04x | 73.33 | 73.32 | 298,936 / 297,746 | 616 / 778 |
| drums | NeRF Synthetic | 6.58 | 6.61 | 1.00x | 77.61 | 74.90 | 319,786 / 319,116 | 649 / 797 |
| ficus | NeRF Synthetic | 5.60 | 5.93 | 0.95x | 90.87 | 84.89 | 177,831 / 177,639 | 401 / 563 |
| hotdog | NeRF Synthetic | 7.40 | 6.02 | 1.23x | 73.45 | 83.88 | 157,488 / 157,839 | 433 / 529 |
| lego | NeRF Synthetic | 6.78 | 6.54 | 1.04x | 76.70 | 75.82 | 298,102 / 299,285 | 648 / 770 |
| materials | NeRF Synthetic | 6.27 | 6.07 | 1.03x | 82.93 | 82.57 | 238,715 / 236,993 | 520 / 670 |
| mic | NeRF Synthetic | 6.58 | 6.92 | 0.95x | 76.49 | 70.99 | 277,299 / 276,471 | 606 / 730 |
| ship | NeRF Synthetic | 9.31 | 6.80 | 1.37x | 61.01 | 73.29 | 309,749 / 310,845 | 811 / 789 |
| train | Tanks and Temples | 11.22 | 9.29 | 1.21x | 44.88 | 50.35 | 1,094,613 / 1,089,398 | 1,902 / 2,108 |
| truck | Tanks and Temples | 12.47 | 11.38 | 1.10x | 39.89 | 41.49 | 2,051,896 / 2,052,183 | 3,351 / 3,696 |
| drjohnson | Deep Blending | 23.34 | 16.58 | 1.41x | 21.34 | 27.93 | 3,109,310 / 3,109,155 | 5,270 / 5,639 |
| playroom | Deep Blending | 18.31 | 20.38 | 0.90x | 28.12 | 37.85 | 1,842,952 / 1,842,580 | 3,149 / 3,474 |

Across the complete suite, CUDA takes 120.81 minutes and all-Warp takes 109.23 minutes, a 1.106x wall-clock ratio. This is a suite aggregate, not an arithmetic mean of per-scene ratios. Warp is faster in 8 of 12 complete jobs and in 6 of 12 stable iteration windows. Training peak allocated memory is usually higher on Warp; this is a measured tradeoff, not a memory-reduction claim. The [rasterizer documentation](docs/rasterizer.md) reports phase splits and current Nsight evidence.

### Warm Inference

For a frozen Warp-trained 30K checkpoint, three warmed full-test-view passes were measured and the median CUDA-event time is reported. Each pass warms 100 views first. This is the actual integration configuration, including depth disabled on Warp when the caller does not consume it.

| Scene | Backend | GPU ms/view | Warp/CUDA ratio | Peak allocated |
|-------|---------|------------:|----------------:|---------------:|
| Lego | CUDA | 1.8224 | 1.00x | 290 MiB |
| Lego | Warp | 1.6382 | 1.11x | 206 MiB |
| Truck | CUDA | 4.2975 | 1.00x | 1,317 MiB |
| Truck | Warp | 4.0478 | 1.06x | 1,207 MiB |

### Independent Training Quality

The following test-image metrics come from the original 3DGS `render.py` and `metrics.py` workflow after separately training CUDA and all-Warp configurations for 30K steps. Both configurations use the same optimization settings and the same native CUDA renderer for final evaluation. Their training gradients are not required to be bitwise identical: floating-point reduction order can change optimizer and densification decisions over a non-convex trajectory. These values therefore measure end-to-end training outcomes, not pixelwise renderer equivalence.

| Scene | CUDA PSNR | Warp PSNR | Delta PSNR | CUDA SSIM | Warp SSIM | CUDA LPIPS | Warp LPIPS |
|-------|----------:|----------:|-----------:|----------:|----------:|-----------:|-----------:|
| chair | 35.6934 | 35.7688 | +0.0754 | 0.987447 | 0.987523 | 0.011773 | 0.011646 |
| drums | 26.1614 | 26.1687 | +0.0073 | 0.954811 | 0.954778 | 0.036494 | 0.036379 |
| ficus | 34.8947 | 34.9049 | +0.0102 | 0.987307 | 0.987330 | 0.011735 | 0.011737 |
| hotdog | 37.6701 | 37.7385 | +0.0684 | 0.985379 | 0.985416 | 0.019953 | 0.019955 |
| lego | 35.9071 | 35.9231 | +0.0161 | 0.983264 | 0.983284 | 0.015307 | 0.015234 |
| materials | 30.1175 | 30.1108 | -0.0066 | 0.961664 | 0.961612 | 0.032918 | 0.032874 |
| mic | 35.8806 | 35.6974 | -0.1832 | 0.992079 | 0.991871 | 0.005762 | 0.005878 |
| ship | 31.0945 | 31.0405 | -0.0540 | 0.907392 | 0.907365 | 0.105282 | 0.105641 |
| train | 22.2565 | 21.9888 | -0.2677 | 0.821652 | 0.820027 | 0.195832 | 0.196647 |
| truck | 25.5052 | 25.5192 | +0.0141 | 0.884745 | 0.884810 | 0.142212 | 0.142430 |
| drjohnson | 29.3829 | 29.4829 | +0.1000 | 0.904947 | 0.905391 | 0.236216 | 0.235705 |
| playroom | 30.1062 | 30.1593 | +0.0531 | 0.909675 | 0.908975 | 0.241131 | 0.240396 |

Warp has higher PSNR in 8 of 12 independently trained scenes. The largest negative delta is train at -0.2677 dB, and the largest positive delta is drjohnson at +0.1000 dB. The suite does not show a uniform quality direction. Scene-level deltas must be interpreted together with the frozen-checkpoint and module-level checks below.

### Frozen-Checkpoint Equivalence

To isolate the rasterizer from training dynamics, the same Warp-trained checkpoint was rendered through both the native CUDA extension and gswarp. The comparison uses identical camera, Gaussian, and background inputs for each view.

| Scene | Test views | Image MAE | CUDA/Warp PSNR | Visibility Jaccard | Max abs. error |
|-------|-----------:|----------:|---------------:|-------------------:|---------------:|
| Lego | 200 | 1.79e-7 | 105.54 dB | 1.000000 | 0.00996 |
| Truck | 32 | 4.64e-7 | 100.43 dB | 1.000000 | 0.01326 |
| Train | 38 | 5.10e-7 | 100.33 dB | 0.99999992 | 0.00478 |

Lego and Truck have identical visible-Gaussian sets. On Train, the Jaccard difference from 1.0 corresponds to two one-sided visibility decisions across approximately 25.6 million Gaussian-view observations; the native CUDA and Warp images also have the same 20.955083 dB global PSNR against ground truth to the shown precision. The CUDA/Warp PSNR column measures agreement between the two renderer outputs, not reconstruction quality against ground truth.

These results support rasterizer equivalence within the measured FP32 tolerance and do not indicate systematic missed coverage. They do not imply that independently trained, non-convex trajectories must converge to identical checkpoints. For the phase timing, provenance rules, and interpretation limits, see [docs/rasterizer.md](docs/rasterizer.md); for controlled SSIM gradient and training-path evidence, see [docs/ssim.md](docs/ssim.md).

---

## Detailed Documentation

| Document | Contents |
|----------|----------|
| [docs/rasterizer.md](docs/rasterizer.md) | Architecture, CUDA implementation differences, micro-benchmarks, correctness, known limitations |
| [docs/ssim.md](docs/ssim.md) | SSIM kernel optimizations, performance analysis, correctness |
| [docs/knn.md](docs/knn.md) | KNN algorithm, Morton sorting, execution model, correctness |

---

## Acknowledgements

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) (INRIA / MPII)
- [fused-ssim](https://github.com/rahul-goel/fused-ssim) (Rahul Goel et al.)
- [simple-knn](https://github.com/camenduru/simple-knn) (graphdeco-inria)
- [Fast Converging 3DGS](https://arxiv.org/abs/2601.19489) (Zhang et al., 2025) — inspiration for compact AABB culling
- [NVIDIA Warp](https://nvidia.github.io/warp/)

