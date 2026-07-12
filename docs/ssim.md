# SSIM Module

[中文](ssim_zh.md) · **English**

This document covers the implementation details, kernel performance, and correctness of the `gswarp.fused_ssim` module.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Implementation Details](#implementation-details)
- [Python Overhead and Launch Caching](#python-overhead-and-launch-caching)
- [Kernel Performance Analysis](#kernel-performance-analysis)
- [End-to-End Training Impact](#end-to-end-training-impact)
- [Correctness](#correctness)
- [Known Limitations](#known-limitations)

---

## Overview

`gswarp.fused_ssim` is a Warp-based drop-in replacement for [fused-ssim](https://github.com/rahul-goel/fused-ssim). It computes the differentiable structural similarity loss L_SSIM = 1 − SSIM(rendered, gt), which corresponds to the λ = 0.2 term in the standard 3DGS loss.

The API is fully compatible with `fused_ssim.fused_ssim()`:

```python
from gswarp.fused_ssim import fused_ssim
loss_ssim = warp_ssim(image, gt_image)  # drop-in replacement
```

---

## Algorithm

SSIM is defined in terms of Gaussian-weighted means (μ), variances (σ²), and cross-covariance (σ₁₂):

$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

Gaussian smoothing (σ=1.5, window=11) is applied to the means and variances. Separable convolution factorizes the 2D 11×11 Gaussian into two 1D passes, yielding 4 kernels:

| Kernel | Operation | Direction |
|--------|-----------|-----------|
| `fwd_h` | Row-wise Gaussian mean | → |
| `fwd_v` | Column mean from row means; also computes SSIM values | ↓ |
| `bwd_h` | Backward pass along rows | → |
| `bwd_v` | Backward pass along columns; outputs gradients | ↓ |

`fwd_v` simultaneously computes SSIM values and the scalar loss; `bwd_v` computes SSIM gradients and applies the chain rule.

---

## Implementation Details

The code lives in `gswarp/fused_ssim.py` and applies four main optimizations:

### C0: Compile-Time Gaussian Weights

The 11 Gaussian weights (σ=1.5) are unrolled as compile-time constants. Warp embeds them directly as PTX immediates, eliminating runtime weight-table loads.

### C4: `_SSIMPlan` Launch Cache (Most Impactful)

Training uses a bounded pool of `_SSIMPlan` objects keyed by device, current PyTorch stream, shape, padding, constants, and tuned block dimensions. A plan owns its intermediate buffers and four recorded commands.

Each forward acquires a private lease that remains attached to its autograd context until the graph is released. This prevents a later forward from overwriting saved workspace and supports multiple live graphs, `retain_graph=True`, and non-default streams. Returned gradients are newly allocated and never stored in the reusable pool.

Dynamic image, upstream-gradient, and output pointers are passed through call-scoped ownerless descriptors. Recorded commands never retain the input autograd graph.

### M1: Flat Array Layout

Intermediate buffers (row means, column means, etc.) use 1D flat layout (`[C, H, W]`) rather than structured tensors, enabling direct pointer-offset access in kernels and reducing index computation overhead.

### O3: Lifetime-Based Workspace Aliasing

The horizontal backward workspace aliases three channels of the horizontal forward buffer. The forward values are dead before backward writes begin on the same stream, so the plan reduces retained storage without overlapping live values.

---

## Python Overhead and Launch Caching

Forward and backward enter the same call-scoped PyTorch/Warp execution context used by the rasterizer. Recorded commands keep static workspace parameters, while only dynamic pointers are updated for each invocation. The free-plan pool is bounded per key and per device; retirement and explicit cache clearing synchronize only the affected Warp streams.

Current-device timing is intentionally omitted from this description. The numeric sections below are historical until CUDA fused-ssim and Warp SSIM are rerun on the new GPU.

---

## Kernel Performance Analysis

> **Historical benchmark pending refresh.** The following values were collected on the previous GPU and an earlier implementation snapshot.

**Platform**: NVIDIA RTX 5090D V2 (sm_120), PyTorch 2.11.0+cu130, Warp 1.12.0.

### Per-Kernel GPU Time (drjohnson, 1332×876)

| Kernel | Operation | GPU Time | Share of Kernel Total |
|--------|-----------|----------|----------------------|
| `fwd_h` | Horizontal Gaussian convolution | 0.040 ms | 16% |
| `fwd_v` | Vertical convolution + SSIM computation | 0.101 ms | 40% |
| `bwd_h` | Backward horizontal propagation | 0.036 ms | 14% |
| `bwd_v` | Backward vertical propagation | 0.073 ms | 29% |
| **Total (kernels)** | | **0.250 ms** | 100% |
| **End-to-end (with Python overhead)** | | **0.303 ms** | — |

`fwd_v` is the hot kernel: at drjohnson resolution, the V-pass intermediate buffer is ~124 MB, exceeding the L2 cache (64 MB on sm_120) and making bandwidth the primary bottleneck.

### Comparison with CUDA fused-ssim

| Resolution | fused-ssim e2e | Warp SSIM e2e | Ratio |
|------------|---------------|--------------|-------|
| 1080p (1920×1080) | 0.647 ms | 0.738 ms | 1.14× |
| drjohnson (1332×876) | — | 0.303 ms | — |

At drjohnson resolution, Warp kernel execution time (0.25 ms) is already faster than fused-ssim's kernel total (~0.35 ms), but Python overhead (0.053 ms) brings the e2e total slightly above. At 1080p, V-pass buffer overflow beyond L2 adds extra DRAM traffic, raising the ratio to 1.14×.

---

## End-to-End Training Impact

> **Historical benchmark pending refresh.** CUDA and Warp loss backends must be rerun under the same new-hardware training configuration before these values are used as current claims.

> **Benchmark conditions**: The ablation data below was collected with the rasterizer fixed to the Warp backend and SSIM/KNN each taking two values. Python-layer overhead optimizations had not yet been applied. The numbers reflect SSIM's isolated contribution to training speed, not the final all-Warp stack performance.

Ablation runs on drjohnson and playroom with rasterizer fixed to Warp (to isolate SSIM/KNN effects), comparing 4 combinations.

**drjohnson (1332×876, ~3.1M Gaussians, 30K iters)**

| SSIM backend | KNN backend | Throughput (it/s) | Wall time (s) | PSNR@30K |
|-------------|------------|-------------------|--------------|---------|
| cuda-fused | cuda | 30.0 | 853 | 29.504 |
| cuda-fused | warp | 29.9 | 879 | 29.465 |
| **warp** | cuda | **29.6** | **854** | **29.462** |
| warp | warp | 29.1 | 900 | 29.435 |

**playroom (1584×1008, ~1.9M Gaussians, 30K iters)**

| SSIM backend | KNN backend | Throughput (it/s) | Wall time (s) | PSNR@30K |
|-------------|------------|-------------------|--------------|---------|
| cuda-fused | cuda | 45.0 | 619 | 30.458 |
| cuda-fused | warp | 46.3 | 620 | 30.326 |
| **warp** | cuda | **45.0** | **622** | **30.420** |
| warp | warp | 46.7 | 578 | 30.332 |

**Interpretation**:
- In drjohnson (large scene), Warp SSIM is ~**1.3% slower** than cuda-fused (30.0 → 29.6 it/s)
- In playroom, both backends are equal (45.0 → 45.0 it/s)
- PSNR differences (~±0.05 dB) are within training noise; not attributable to SSIM backend choice
- In the full 12-dataset bench30k comparison (all three modules Warp), NeRF Synthetic scenes average ~8% faster — well above the ~1% SSIM overhead. The backward-render warp shuffle and compact AABB binning gains outweigh the SSIM cost.

---

## Correctness

> **Historical numeric snapshot pending refresh.** Current tests cover multiple live forwards, `retain_graph`, non-default/two-stream execution, plan-pool bounds, cache clearing, and returned-gradient ownership. CUDA/PyTorch numerical tables will be regenerated on the new GPU.

**Forward (loss value accuracy)**:

Using PyTorch reference implementation (`F.conv2d` with same Gaussian kernel) as ground truth:

| Dataset | Resolution | L_SSIM diff |
|---------|------------|-------------|
| drjohnson (random frame) | 1332×876 | 0.00e+00 (bit-exact) |
| playroom (random frame) | 1584×1008 | 0.00e+00 (bit-exact) |
| Synthetic (800×800) | 800×800 | 0.00e+00 (bit-exact) |

Forward is bit-exact with PyTorch reference in most test conditions: Gaussian kernel coefficients are constant, and Warp and PyTorch FP32 Conv2d follow the same computation path.

**Backward (gradient accuracy)**:

Using `torch.autograd.gradcheck` finite differences as reference:

| Dataset | Resolution | Gradient L∞ difference |
|---------|------------|----------------------|
| drjohnson | 1332×876 | ~1e-11 |
| playroom | 1584×1008 | ~1e-11 |

Gradient errors are within numerical analysis machine precision and have no impact on training convergence.

---

## Known Limitations

- The stable implementation keeps the four-pass separable algorithm and therefore materializes intermediate image-sized buffers.
- Training plans are bounded, but each live autograd graph requires a private lease until that graph is released.
- `padding="valid"` requires both spatial dimensions to be at least 11.
- Inputs must be same-shape, same-device CUDA `float32` tensors.
- Performance and memory behavior depend on image shape, concurrent live graphs, stream count, and the current Warp/PyTorch versions; old single-device ratios are not portable claims.
