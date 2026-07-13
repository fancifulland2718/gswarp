# SSIM Module

[中文](ssim_zh.md) · **English**

This document covers the implementation details, kernel performance, and correctness of the `gswarp.fused_ssim` module.

---

## Table of Contents

- [Overview](#overview)
- [Algorithm](#algorithm)
- [Implementation Details](#implementation-details)
- [Python Overhead and Launch Caching](#python-overhead-and-launch-caching)
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

### Compile-Time Gaussian Weights

The 11 Gaussian weights (σ=1.5) are unrolled as compile-time constants. Warp embeds them directly as PTX immediates, eliminating runtime weight-table loads.

### `_SSIMPlan` Launch Cache

Training uses a bounded pool of `_SSIMPlan` objects keyed by device, current PyTorch stream, shape, padding, constants, and tuned block dimensions. A plan owns its intermediate buffers and four recorded commands.

Each forward acquires a private lease that remains attached to its autograd context until the graph is released. This prevents a later forward from overwriting saved workspace and supports multiple live graphs, `retain_graph=True`, and non-default streams. Returned gradients are newly allocated and never stored in the reusable pool.

Dynamic image, upstream-gradient, and output pointers are passed through call-scoped ownerless descriptors. Recorded commands never retain the input autograd graph.

### Flat Array Layout

Intermediate buffers (row means, column means, etc.) use 1D flat layout (`[C, H, W]`) rather than structured tensors, enabling direct pointer-offset access in kernels and reducing index computation overhead.

### Lifetime-Based Workspace Aliasing

The horizontal backward workspace aliases three channels of the horizontal forward buffer. The forward values are dead before backward writes begin on the same stream, so the plan reduces retained storage without overlapping live values.

---

## Python Overhead and Launch Caching

Forward and backward enter the same call-scoped PyTorch/Warp execution context used by the rasterizer. Recorded commands keep static workspace parameters, while only dynamic pointers are updated for each invocation. The free-plan pool is bounded per key and per device; retirement and explicit cache clearing synchronize only the affected Warp streams.

### Current RTX 5090 Controlled Timing

The package artifact was measured on an RTX 5090 (32 GiB), Python 3.14.3, PyTorch 2.11.0+cu130, and Warp 1.12.0 at 800x800. The benchmark uses 30 warmups and 200 repetitions, reports CUDA-event medians, and keeps inputs and reduction semantics fixed. It is a loss-stack microbenchmark, not an end-to-end training throughput claim.

| Operation | CUDA fused SSIM | Warp SSIM | Warp/CUDA |
|-----------|----------------:|----------:|----------:|
| Forward | 0.166 ms | 0.143 ms | 0.86x |
| Backward only | 0.193 ms | 0.240 ms | 1.24x |
| SSIM training path | 0.280 ms | 0.359 ms | 1.28x |
| L1 plus SSIM training path | 0.469 ms | 0.624 ms | 1.33x |

Warp forward is lower in this controlled workload, while the backward path is higher. The result supports treating SSIM backward and loss-stack orchestration as a material training hotspot; it does not justify attributing an end-to-end scene result to SSIM alone. The current full 12-scene training matrix is in the [README](../README.md#benchmarks).

### Current Numerical Behavior

CUDA fused SSIM and Warp SSIM use the same SSIM formula, Gaussian coefficients, and padding semantics, but their FP32 reduction and backward execution orders are not bitwise identical. The following comparison uses the gradient with respect to the rendered image; relative L1 is defined as the sum of absolute gradient differences divided by the sum of absolute CUDA-gradient values.

| Input | SSIM forward abs. delta | Gradient max abs. delta | Gradient relative L1 |
|-------|------------------------:|------------------------:|---------------------:|
| Random 800x800 image pair | 0.00 | 3.18e-12 | 2.04e-7 |
| Train rendered training view, 545x980 | 1.79e-7 | 4.83e-9 | 1.45e-5 |

For the Train view, CUDA fused SSIM is 0.9185836315 and Warp SSIM is 0.9185838103. These are small single-operation differences, but repeated application in a non-convex optimizer can change later parameter updates and densification decisions.

A fixed-seed 30K Train component comparison was evaluated with the original 3DGS `render.py` and `metrics.py`; every final checkpoint was rendered through the same native CUDA rasterizer:

| Training rasterizer | Training SSIM/KNN | PSNR | SSIM | LPIPS | Final Gaussians |
|---------------------|-------------------|-----:|-----:|------:|----------------:|
| CUDA | CUDA / CUDA | 22.256477 | 0.821652 | 0.195832 | 1,094,613 |
| Warp | CUDA / CUDA | 22.123102 | 0.821906 | 0.194714 | 1,094,296 |
| CUDA | Warp / Warp | 21.958277 | 0.820587 | 0.196497 | 1,091,900 |
| Warp | Warp / Warp | 21.988764 | 0.820027 | 0.196647 | 1,089,398 |

The actual Train initialization point cloud contains 182,686 points, and Warp KNN matched native `simple-knn` bitwise for every squared-distance output. KNN therefore does not explain the auxiliary-path difference in this comparison. The table instead shows that small SSIM backward differences can select a different optimization trajectory. The rasterizer and auxiliary effects are not additive, and this single-seed component comparison is not a statistical estimate of expected scene quality.

These results do not indicate an SSIM formula or differentiation defect. They establish a narrower conclusion: CUDA and Warp agree closely for a single forward/backward operation, while bitwise-identical 30K checkpoints are not a valid requirement across the two FP32 execution paths.

---

## Correctness

The current package artifact is checked at both the individual-operation and training-path levels. The numerical tables above compare CUDA fused SSIM and Warp SSIM on a random 800x800 input and on an actual 545x980 Train view. Forward absolute differences are 0.00 and 1.79e-7; gradient maximum absolute differences are 3.18e-12 and 4.83e-9 respectively.

Regression tests additionally cover multiple live forwards, retained graphs, non-default and concurrent streams, bounded plan pools, cache clearing, and returned-gradient ownership. CUDA and Warp use the same SSIM formula, Gaussian coefficients, and padding semantics. Their FP32 execution order is not bitwise identical, so correctness is assessed with bounded numerical comparisons rather than checkpoint identity after non-convex training.

---

## Known Limitations

- The stable implementation keeps the four-pass separable algorithm and therefore materializes intermediate image-sized buffers.
- Training plans are bounded, but each live autograd graph requires a private lease until that graph is released.
- `padding="valid"` requires both spatial dimensions to be at least 11.
- Inputs must be same-shape, same-device CUDA `float32` tensors.
- Performance and memory behavior depend on image shape, concurrent live graphs, stream count, and the current Warp/PyTorch versions; single-device ratios are not portable claims.
