# SSIM Module

**中文** · [English](ssim.md)

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

Each Warp kernel invocation requires ~37 `set_param_at_index` calls, ~10 `wp.from_torch` calls, and several `torch.empty` allocations in Python.

`_SSIMPlan` caches all intermediate buffers and kernel parameter references on the first forward pass. Subsequent iterations update only the few pointer slots that actually change (~5 `set_param_at_index` calls), bypassing the reconstruction overhead.

Effect (chair 30K training, RTX 5090D V2): −7.1 s total loss-phase time, approximately **−2.8%** in overall training time.

### M1: Flat Array Layout

Intermediate buffers (row means, column means, etc.) use 1D flat layout (`[C, H, W]`) rather than structured tensors, enabling direct pointer-offset access in kernels and reducing index computation overhead.

### O3: Dead-Buffer Reuse

The `bwd_h` output buffer can reuse an expired intermediate tensor from `fwd_h`, eliminating one `torch.empty` allocation per backward call.

---

## Python Overhead and Launch Caching

The fixed Python overhead per SSIM call (Python→Warp parameter marshaling, kernel launch, result collection) is approximately **0.05 ms/call**, resolution-independent.

On drjohnson (1332×876), total e2e per SSIM call is ~0.30 ms: 0.25 ms kernel execution + 0.05 ms Python overhead (17%).

The C4 launch cache reduces Python overhead from ~0.08 ms/call to ~0.05 ms/call — the dominant contributor to the 2.8% training speedup.

---

## Kernel Performance Analysis

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

**Bandwidth bottleneck**: Separable convolution intermediate buffers exceed L2 cache at large resolutions (V-pass buffer ~220 MB at 1080p). CUDA fused-ssim keeps more computation on-chip via `__shared__` memory; Warp lacks explicit shared memory management and must rely on L2/DRAM. This is the root cause of the 14% e2e slowdown at 1080p.

**Theoretical ceiling**: Without shared memory, the achievable bandwidth efficiency is at most ~1/1.2× of CUDA fused-ssim — meaning Warp SSIM's theoretical performance ceiling at large resolutions is approximate parity with fused-ssim. The observed 1.14× ratio is close to this limit.
