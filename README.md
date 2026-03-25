# gswarp

[中文](README_zh.md) · **English**

A **pure Python + NVIDIA Warp** reimplementation of the differentiable Gaussian rasterization pipeline. The original version was written in CUDA C++ for [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/). This backend provides a **drop-in replacement** for the native CUDA rasterizer — no C++/CUDA compilation is required.

> **License**: This project inherits the [Gaussian-Splatting License](LICENSE) (Inria and MPII, for non-commercial research use only).

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
- [Architecture Overview](#architecture-overview)
- [Differences from the CUDA Baseline](#differences-from-the-cuda-baseline)
- [Correctness](#correctness)
- [Performance Characteristics](#performance-characteristics)
- [Known Limitations](#known-limitations)
- [Future Optimization Directions](#future-optimization-directions)
- [Acknowledgments](#acknowledgments)

---

## Features

- **Complete rasterization pipeline**: Preprocess → Binning → Forward render → Backward render, all implemented by Warp kernels.
- **Drop-in replacement**: The API is compatible with the native CUDA backend — you only need to change the import path to switch backends.
- **No compilation required**: Pure Python + Warp JIT. No `setup.py build_ext`, no CUDA toolkit headers, and no platform-specific build issues.
- **Spherical Harmonics**: Supports degrees 0–3, consistent with the original implementation.
- **Auto-tuning**: Occupancy-aware kernel block-dimension selection based on GPU SM architecture (Volta through Blackwell).
- **Multiple binning sort modes**: `warp_depth_stable_tile` (default, recommended), `warp_radix`, `torch`, `torch_count`.
- **Fused backward kernels**: Merges allocation and gradient-accumulation steps to reduce kernel launch overhead.
- **Tight AABB tile culling**: Uses per-axis 3σ bounding boxes for Gaussian-to-tile assignment (inspired by [Zhang et al., 2025](https://arxiv.org/abs/2601.19489)), reducing unnecessary tile overlap for elongated Gaussians.
- **Forward-state packing**: Efficient forward-state serialization for backward reuse, avoiding redundant recomputation.

---

## Requirements

| Component | Minimum Version | Tested Version |
|------|---------|---------|
| **Python** | 3.10+ | 3.10 |
| **NVIDIA GPU** | Compute Capability ≥ 7.0 (Volta) | RTX 4060 Laptop |
| **NVIDIA Driver** | Compatible with CUDA 12.x | 13.2 |
| **PyTorch** | 2.0+ (with CUDA support) | 2.7.0+cu126 |
| **NVIDIA Warp** | 1.12.0+ | 1.12.0 |

The SM-architecture auto-tuning table covers the following architectures:

| Architecture | Compute Capability |
|------|---------|
| Volta | 7.0 |
| Turing | 7.5 |
| Ampere (GA100) | 8.0 |
| Ampere (GA10x) | 8.6 |
| Ada Lovelace | 8.9 |
| Hopper | 9.0 |
| Blackwell | 10.0 |

GPUs outside this table still work correctly — the auto-tuning logic falls back to conservative default parameters.

---

## Installation

### 1. Install dependencies

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install warp-lang>=1.12.0
```

### 2. Clone and integrate

Add this repository as a submodule of a 3DGS project (or any project that uses `gswarp`):

```bash
git clone https://github.com/fancifulland2718/gswarp.git submodules/gswarp
```

### 3. (Optional) Build the native CUDA baseline for comparison

If you also need the native CUDA rasterizer (for A/B testing or fallback):

```bash
cd submodules/gswarp
pip install .
```

> **Note**: The Warp backend itself does **not** require `pip install .` or any native compilation. It can be run directly through Python imports.

---

## Quick Start

### Using the Warp backend

```python
# Import the Warp backend (instead of the native CUDA backend)
from gswarp.warp import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
    initialize_runtime_tuning,
)

# (Optional) Initialize runtime auto-tuning — detect the GPU and choose optimal parameters
initialize_runtime_tuning(device="cuda:0", verbose=True)

# Configure rasterization settings
raster_settings = GaussianRasterizationSettings(
    image_height=height,
    image_width=width,
    tanfovx=tanfovx,
    tanfovy=tanfovy,
    bg=bg_color,          # [3] tensor, background color
    scale_modifier=1.0,
    viewmatrix=viewmatrix,     # [4, 4] world-to-camera matrix
    projmatrix=projmatrix,     # [4, 4] full projection matrix
    sh_degree=active_sh_degree,
    campos=campos,             # [3] camera position in world space
    prefiltered=False,
    # Warp-specific optional fields:
    backward_mode="manual",              # Only "manual" is supported
    binning_sort_mode="warp_depth_stable_tile",  # or: "warp_radix", "torch", "torch_count"
    auto_tune=True,
    auto_tune_verbose=True,
)

# Create the rasterizer and run the forward pass
rasterizer = GaussianRasterizer(raster_settings=raster_settings)
color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(
    means3D=means3D,           # [N, 3]
    means2D=means2D,           # [N, 3] (screen space, receives gradients)
    opacities=opacities,       # [N, 1]
    shs=shs,                   # [N, K, 3] spherical-harmonic coefficients
    scales=scales,             # [N, 3]
    rotations=rotations,       # [N, 4] quaternions
)

# Backward propagation is handled automatically by PyTorch autograd
loss = compute_loss(color, target)
loss.backward()
```

### Switching from native CUDA to Warp

Replace your import statements:

```python
# Before (native CUDA):
from gswarp import GaussianRasterizationSettings, GaussianRasterizer

# After (Warp backend):
from gswarp.warp import GaussianRasterizationSettings, GaussianRasterizer
```

The Warp backend returns more forward outputs than the native version:

```python
# Native CUDA outputs:
color, radii = rasterizer(...)

# Warp backend outputs:
color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu = rasterizer(...)
```

---

## API Reference

### Core Functions

| Function | Description |
|------|------|
| `rasterize_gaussians(...)` | Full forward pass: preprocess + binning + rendering |
| `rasterize_gaussians_backward(...)` | Full backward pass: backward render + backward preprocess |
| `mark_visible(positions, viewmatrix, projmatrix)` | Returns the visibility mask of 3D positions |
| `preprocess_gaussians(...)` | Preprocess only (without rendering), useful for analysis |

### Runtime Configuration

| Function | Description |
|------|------|
| `initialize_runtime_tuning(device, verbose)` | One-time GPU detection and parameter tuning |
| `get_runtime_tuning_report(device)` | Returns the current tuning report (memory, tile size, sort mode) |
| `get_runtime_auto_tuning_config()` | Returns the auto-tuning switch status |
| `set_binning_sort_mode(mode)` | Sets the binning sort mode at runtime |
| `get_default_parameter_info()` | Returns compile-time constants (`TOP_K`, `BLOCK_X`, etc.) |
| `is_available()` | Checks whether Warp can be imported |

### GaussianRasterizationSettings Fields

| Field | Type | Description |
|------|------|------|
| `image_height` | `int` | Output image height in pixels |
| `image_width` | `int` | Output image width in pixels |
| `tanfovx` | `float` | tan(FoV_x / 2) |
| `tanfovy` | `float` | tan(FoV_y / 2) |
| `bg` | `Tensor[3]` | Background color |
| `scale_modifier` | `float` | Global scale multiplier |
| `viewmatrix` | `Tensor[4,4]` | World-to-camera transform matrix |
| `projmatrix` | `Tensor[4,4]` | Full projection matrix |
| `sh_degree` | `int` | Active SH degree (0–3) |
| `campos` | `Tensor[3]` | Camera position in world space |
| `prefiltered` | `bool` | Whether points are prefiltered |
| `backward_mode` | `str \| None` | `"manual"` (the only supported mode) |
| `binning_sort_mode` | `str \| None` | Binning sort algorithm |
| `auto_tune` | `bool` | Enables auto-tuning (default: `True`) |
| `auto_tune_verbose` | `bool` | Prints tuning information (default: `True`) |

---

## Architecture Overview

### Pipeline Stages

```
Input Gaussians (means3D, SH, scales, rotations, opacities)
    │
    ▼
┌─────────────────────────────────┐
│  1. PREPROCESS                  │
│  - Cov3D from scale+rotation    │
│  - Project to 2D (cov2d, conic) │
│  - Frustum + near-plane culling │
│  - SH → RGB color evaluation    │
│  - Tight AABB tile rectangle    │
│  - Forward-state packing        │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  2. BINNING                     │
│  - Tile-overlap counting (scan) │
│  - Gaussian→tile duplication    │
│  - Depth sort + tile sort       │
│  - Tile-range identification    │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  3. FORWARD RENDER              │
│  - Per-pixel alpha blending     │
│  - Front-to-back compositing    │
│  - TOP_K early termination      │
│  - Color, depth, alpha outputs  │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  4. BACKWARD RENDER             │
│  - Gradients w.r.t. conic,      │
│    opacity, color, and pos      │
│  - atomic_add accumulation      │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────┐
│  5. BACKWARD PREPROCESS             │
│  - Gradients w.r.t. means3D, scales,│
│    and rotations                    │
│  - SH backward                      │
│  - Cov3D → scale/rotation gradients │
└─────────────────────────────────────┘
```

### Single-File Design

The entire Warp backend is contained in a single Python file (about 4400 lines), including:

- all Warp kernel definitions (`@wp.kernel`)
- all Warp helper functions (`@wp.func`)
- runtime auto-tuning
- public API functions

This is intentional — Warp's JIT compilation model requires all kernel code and dependencies to remain within the same `wp.Module` scope. Splitting the code across multiple files would break Warp's ability to resolve `@wp.func` cross-references during JIT.

### Key Constants

| Constant | Value | Description |
|------|---|------|
| `TOP_K` | 20 | Maximum number of Gaussians per pixel (early termination) |
| `BLOCK_X` | 16 | Tile width in pixels |
| `BLOCK_Y` | 16 | Tile height in pixels |
| `NUM_CHANNELS` | 3 | RGB output channels |
| `PREPROCESS_CULL_SIGMA` | 3.0 | Frustum-culling sigma multiplier |
| `PREPROCESS_CULL_FOV_SCALE` | 1.3 | FoV-boundary scale used for culling |
| `VISIBILITY_NEAR_PLANE` | 0.2 | Near-plane distance used for culling |

---

## Differences from the CUDA Baseline

### 1. Tight AABB vs Isotropic Radius

The tight AABB bounding-box method used in this project is inspired by *[Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction](https://arxiv.org/abs/2601.19489)* (Ziyu Zhang, Tianle Liu, Diantao Tu, Shuhan Shen, arXiv:2601.19489). This technique replaces the isotropic square radius with per-axis extents derived from the 2D covariance matrix, yielding tighter tile assignment for anisotropic Gaussians. This project introduces no extra code dependency — only the bounding-box computation logic is adopted.

**CUDA baseline** computes a square bounding box using the maximum of the two eigenvalue-derived radii:

```c
// CUDA: auxiliary.h getRect()
int max_radius = ...;  // max(ceil(3σ_max), 0), isotropic
rect_min = {min(grid.x, max((int)((point.x - max_radius) / BLOCK_X), 0)), ...};
rect_max = {min(grid.x, max((int)((point.x + max_radius + BLOCK_X - 1) / BLOCK_X), 0)), ...};
```

**Warp backend** computes a per-axis tight bounding box using the diagonal elements of the 2D covariance matrix:

```python
# Warp: _compute_tile_rect_tight_wp()
radius_x = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(cov_xx, 0.01))))
radius_y = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(cov_yy, 0.01))))
```

**Impact**:
- For **elongated Gaussians** (high anisotropy), the Warp backend assigns fewer tiles, reduces `num_rendered`, and improves binning/render efficiency.
- For **circular Gaussians**, the two approaches are equivalent.
- This introduces a small mismatch: some boundary tiles included by the CUDA baseline due to its overly conservative isotropic radius are excluded by Warp's tighter bounds. The visual difference is negligible, but measurable in numerical comparisons.

### 2. No Shared-Memory Cooperative Fetch

**CUDA baseline** uses `__shared__` memory for cooperative tile-level data fetching:

```c
// CUDA: forward.cu renderCUDA()
__shared__ int collected_id[BLOCK_SIZE];
__shared__ float2 collected_xy[BLOCK_SIZE];
__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

// All threads in the tile cooperatively load a batch of Gaussians from global
// memory into shared memory, and then iterate over that batch.
for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++) {
    // Read from shared memory — fast, broadcast to all threads in the tile
    float2 xy = collected_xy[j];
    float4 con_o = collected_conic_opacity[j];
    ...
}
```

In the **Warp backend**, using `__shared__` memory is currently relatively difficult. Each thread reads independently from global memory:

```python
# Warp: each thread reads independently
g_idx = point_list[tile_start + j]
xy_x = means2d_x[g_idx]
xy_y = means2d_y[g_idx]
con_x = conic_x[g_idx]
...
```

**Impact**:
- This is the **largest performance gap** between Warp and CUDA at small and medium scales. In CUDA's shared-memory pattern, a single load is shared by all 256 threads in a tile, whereas Warp issues 256 independent global reads per Gaussian.
- In large scenes (`num_rendered > ~30K`), blending computation and memory bandwidth dominate performance, so this gap becomes smaller.

### 3. Sorting Differences

**CUDA baseline** uses a single CUB `DeviceRadixSort` pass with a packed 64-bit key (`(tile_id << 32) | depth_bits`).

**Warp backend** provides four sort modes. They share the same **preprocess / render / backward-render** kernels, and differ only in **how the binning stage builds and sorts `(tile, point)` pairs**:

- `warp_radix`: directly duplicates packed 64-bit `(tile_id, depth_bits)` keys inside Warp and runs a **single Warp radix sort**. This is the shortest path and uses the lightest scratch, but its tie-break behavior is not fully identical to CUDA/CUB.
- `warp_depth_stable_tile`: first sorts points by depth with a **Warp i32 radix sort**, then duplicates `(tile_id, point_id)` in that depth order, and finally performs a **stable Warp radix sort by tile id**, preserving depth order inside each tile.
- `torch`: first generates `tile_id / point_id` in Warp, then performs a **stable PyTorch argsort**. When `num_rendered <= TORCH_SINGLE_SORT_THRESHOLD`, it uses a single packed-key sort; above that threshold, it falls back to a two-stage stable sort (`depth`, then `tile`).
- `torch_count`: also generates `tile_id / point_id` in Warp, but first stable-sorts by depth and then sorts by `tile_id` in PyTorch; when the tile count is small, it can take an `int16` fast path to reduce the key width of the second sort.

In other words, the four Warp paths do not disagree on whether preprocess / render are mathematically correct. They differ in **the order in which the same Gaussians are fed into each tile**. That changes the floating-point accumulation order during alpha blending and backward `atomic_add`, which is why the remaining differences appear as sparse numeric residuals instead of large systematic errors.

### 4. Culling Parameters

The Warp backend applies explicit frustum-culling parameters:
- `PREPROCESS_CULL_SIGMA = 3.0`: Gaussians whose 3σ bounding boxes lie fully outside the image are culled.
- `PREPROCESS_CULL_FOV_SCALE = 1.3`: Slightly enlarges the FoV to avoid over-aggressive boundary culling.

This differs slightly from the CUDA baseline's implicit culling behavior.

### 5. Per-Pixel vs Per-Tile Dispatch

The CUDA baseline dispatches the render kernel on a 2D grid of tile blocks:

```c
dim3 grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
dim3 block(BLOCK_X, BLOCK_Y, 1);
```

The Warp backend dispatches in 1D, with one thread per pixel:

```python
wp.launch(kernel, dim=image_height * image_width, ...)
```

The tile index is derived inside the kernel using arithmetic on `wp.tid()`. The functionality is equivalent, but occupancy and scheduling characteristics differ.

---

## Correctness

The test configuration is as follows:

- `mode = sh_scale_rotation`
- `backward_mode="manual"`
- `binning_sort_mode ∈ {"warp_radix", "torch", "warp_depth_stable_tile", "torch_count"}` (explicitly set per run)
- `auto_tune=True`
- `auto_tune_verbose=True`
- test scales: **4,096 / 16,384 / 65,536 / 262,144** particles
- evaluation entry: `tests/evaluate_warp_backend_sort_modes.py --warp-backend-module-path diff_gaussian_rasterization/release.py`

Across the full 4-sort-mode sweep, the preprocess diagnostics stay aligned and the final forward outputs remain exact at the checked tolerance for all 16 combinations. The only remaining differences are sparse backward outliers, and their pattern depends on both the point count and the chosen sort backend.

### How the four `release.py` sort backends work

All four modes share the exact same **preprocess / render / backward-render** kernels. The only thing that changes is **how the binning stage builds and sorts `(tile, point)` pairs**:

- `warp_radix`: duplicates packed 64-bit keys of the form `(tile_id, depth_bits)` inside Warp and runs a **single Warp radix sort**. This is the leanest path, but its tie-break behavior is not identical to CUDA/CUB.
- `warp_depth_stable_tile`: first does a **Warp i32 radix sort by depth**, then duplicates `(tile_id, point_id)` in that depth order, and finally performs a **stable Warp radix sort by tile id**, preserving depth order inside each tile.
- `torch`: generates `tile_id / point_id` in Warp and then switches to **stable PyTorch argsort**. Below `TORCH_SINGLE_SORT_THRESHOLD`, it uses one packed-key stable sort; above that threshold, it falls back to a two-pass stable sort (`depth`, then `tile`).
- `torch_count`: also generates `tile_id / point_id` in Warp, but explicitly stable-sorts by depth first and then sorts by `tile_id` in PyTorch; when the tile count is small, it takes an `int16` fast path for the second sort.

So the four backends do **not** disagree on the preprocess math or the rendering formula. They only disagree on **the order in which the same Gaussians are fed into each tile**, which is why the remaining differences show up only in sparse backward outliers.

### `release.py` sort-mode sweep

> In this experiment each run explicitly forces `binning_sort_mode`, so the requested mode and the actually executed mode would be redundant in the README tables.

<table>
    <thead>
        <tr>
            <th>Points</th>
            <th>Sort mode</th>
            <th>Preprocess</th>
            <th>Backward out-of-threshold coverage</th>
            <th>Backward max-diff</th>
        </tr>
    </thead>
    <tbody>
        <tr><td rowspan="4"><strong>4,096</strong></td><td><code>warp_radix</code></td><td>aligned</td><td>clean</td><td>0.037109</td></tr>
        <tr><td><code>torch</code></td><td>aligned</td><td>clean</td><td><strong>0.011719</strong></td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>aligned</td><td><code>means3D</code>: 1 / 12,288 (0.008138%)</td><td>0.039062</td></tr>
        <tr><td><code>torch_count</code></td><td>aligned</td><td><code>opacity</code>: 1 / 4,096 (0.024414%)</td><td>0.048828</td></tr>
        <tr><td rowspan="4"><strong>16,384</strong></td><td><code>warp_radix</code></td><td>aligned</td><td>clean</td><td>0.014648</td></tr>
        <tr><td><code>torch</code></td><td>aligned</td><td>clean</td><td>0.015625</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>aligned</td><td>clean</td><td><strong>0.010742</strong></td></tr>
        <tr><td><code>torch_count</code></td><td>aligned</td><td>clean</td><td>0.011230</td></tr>
        <tr><td rowspan="4"><strong>65,536</strong></td><td><code>warp_radix</code></td><td>aligned</td><td><code>means3D</code>: 3 / 196,608 (0.001526%)<br><code>opacity</code>: 2 / 65,536 (0.003052%)<br><code>shs</code>: 7 / 3,145,728 (0.000223%)</td><td>0.375000</td></tr>
        <tr><td><code>torch</code></td><td>aligned</td><td><code>means3D</code>: 3 / 196,608 (0.001526%)<br><code>opacity</code>: 1 / 65,536 (0.001526%)<br><code>scales</code>: 2 / 196,608 (0.001017%)<br><code>rotations</code>: 4 / 262,144 (0.001526%)</td><td><strong>0.171875</strong></td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>aligned</td><td><code>means3D</code>: 3 / 196,608 (0.001526%)<br><code>opacity</code>: 1 / 65,536 (0.001526%)<br><code>shs</code>: 7 / 3,145,728 (0.000223%)</td><td>0.343750</td></tr>
        <tr><td><code>torch_count</code></td><td>aligned</td><td><code>means3D</code>: 6 / 196,608 (0.003052%)<br><code>opacity</code>: 2 / 65,536 (0.003052%)<br><code>shs</code>: 13 / 3,145,728 (0.000413%)<br><code>rotations</code>: 3 / 262,144 (0.001144%)</td><td>0.218750</td></tr>
        <tr><td rowspan="4"><strong>262,144</strong></td><td><code>warp_radix</code></td><td>aligned</td><td><code>means3D</code>: 6 / 786,432 (0.000763%)<br><code>means2D</code>: 5 / 786,432 (0.000636%)<br><code>opacity</code>: 2 / 262,144 (0.000763%)<br><code>shs</code>: 49 / 12,582,912 (0.000389%)<br><code>scales</code>: 5 / 786,432 (0.000636%)<br><code>rotations</code>: 5 / 1,048,576 (0.000477%)</td><td>0.578125</td></tr>
        <tr><td><code>torch</code></td><td>aligned</td><td><code>means3D</code>: 3 / 786,432 (0.000381%)<br><code>means2D</code>: 2 / 786,432 (0.000254%)<br><code>opacity</code>: 3 / 262,144 (0.001144%)<br><code>shs</code>: 7 / 12,582,912 (0.000056%)<br><code>scales</code>: 3 / 786,432 (0.000381%)<br><code>rotations</code>: 4 / 1,048,576 (0.000381%)</td><td><strong>0.156250</strong></td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>aligned</td><td><code>means3D</code>: 9 / 786,432 (0.001144%)<br><code>means2D</code>: 5 / 786,432 (0.000636%)<br><code>opacity</code>: 2 / 262,144 (0.000763%)<br><code>shs</code>: 79 / 12,582,912 (0.000628%)<br><code>scales</code>: 6 / 786,432 (0.000763%)<br><code>rotations</code>: 9 / 1,048,576 (0.000858%)</td><td>0.429688</td></tr>
        <tr><td><code>torch_count</code></td><td>aligned</td><td><code>means3D</code>: 7 / 786,432 (0.000890%)<br><code>means2D</code>: 4 / 786,432 (0.000509%)<br><code>opacity</code>: 3 / 262,144 (0.001144%)<br><code>shs</code>: 70 / 12,582,912 (0.000556%)<br><code>scales</code>: 4 / 786,432 (0.000509%)<br><code>rotations</code>: 7 / 1,048,576 (0.000668%)</td><td>0.734375</td></tr>
    </tbody>
</table>

Attribution:

- All four modes keep preprocess tensors aligned and preserve exact forward outputs at the checked tolerance, so the remaining issue is **not** a preprocess or render-path bug.
- The out-of-threshold backward coverage stays extremely small: the worst 4K case is only **1 / 4,096 = 0.024414%**, and most large-scale fields drop into the **$10^{-3}\%$ to $10^{-4}\%$** range.
- The residuals track the sort path, which strongly suggests a **binning-order effect**: once the tile-local traversal order changes, alpha compositing and backward `atomic_add` accumulate floating-point terms in a different order, producing a few sparse gradient spikes.
- `warp_radix` also keeps `WARP_RADIX_DETERMINISTIC_TIEBREAK = False`, so equal-key tie breaks are more likely to differ from the CUDA/CUB reference; `warp_depth_stable_tile` and `torch_count` add extra reorder steps that can also perturb a few boundary samples.
- Empirically, `torch` gives the smallest residuals at 65K and 262K, suggesting that its stable-sort path stays closer to the CUDA baseline traversal order at large scale — but that is an observed trend, not a different preprocessing formula.

---

## Performance Characteristics

The following data also comes from the **current code state**. The test platform is **NVIDIA GeForce RTX 4060 Laptop GPU** (sm_89, 8 GiB, 24 SMs), **Warp 1.12.0**, and **PyTorch 2.7.0+cu126**.

Methodology:

- **Steady-state runtime**: measured via the public API (`diff_gaussian_rasterization.GaussianRasterizer` and `diff_gaussian_rasterization.warp.GaussianRasterizer`) with dedicated warmup runs first, then a batched **CUDA-event** timing pass over the measured iterations; the `Public API FW / Public API BW / Total iteration` columns below all come from this path and are the right numbers for end-to-end comparisons.
- **Memory usage**: after warmup, one forward stage and one backward stage are measured separately, recording the CUDA allocator peak increment (`peak_allocated_delta_mib`).


For the 4K / 16K / 65K / 262K cases, the evaluation uses **12+24 / 10+20 / 6+12 / 4+8** (warmup count + measured count), respectively, for each sort mode.

### CUDA baseline + `release.py` sort-mode sweep

<table>
    <thead>
        <tr>
            <th>Points</th>
            <th>Backend / sort mode</th>
            <th>Public API FW</th>
            <th>Public API BW</th>
            <th>Total iteration</th>
            <th>Internal binning GPU time</th>
        </tr>
    </thead>
    <tbody>
        <tr><td rowspan="5"><strong>4,096</strong></td><td><strong>CUDA baseline</strong></td><td>4.421 ms</td><td>3.082 ms</td><td>7.503 ms</td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td>3.829 ms</td><td>2.850 ms</td><td>6.679 ms</td><td><strong>1.046 ms</strong></td></tr>
        <tr><td><code>torch</code></td><td>3.896 ms</td><td><strong>2.554 ms</strong></td><td>6.449 ms</td><td>2.559 ms</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>3.034 ms</strong></td><td>3.119 ms</td><td>6.153 ms</td><td>1.066 ms</td></tr>
        <tr><td><code>torch_count</code></td><td>3.150 ms</td><td>2.989 ms</td><td><strong>6.139 ms</strong></td><td>2.504 ms</td></tr>
        <tr><td rowspan="5"><strong>16,384</strong></td><td><strong>CUDA baseline</strong></td><td>3.647 ms</td><td>2.865 ms</td><td>6.511 ms</td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td>4.084 ms</td><td>3.811 ms</td><td>7.894 ms</td><td><strong>1.774 ms</strong></td></tr>
        <tr><td><code>torch</code></td><td>3.796 ms</td><td><strong>2.522 ms</strong></td><td><strong>6.319 ms</strong></td><td>2.649 ms</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>3.351 ms</td><td>3.738 ms</td><td>7.089 ms</td><td>2.047 ms</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>3.312 ms</strong></td><td>3.413 ms</td><td>6.725 ms</td><td>2.836 ms</td></tr>
        <tr><td rowspan="5"><strong>65,536</strong></td><td><strong>CUDA baseline</strong></td><td>11.788 ms</td><td>3.343 ms</td><td>15.131 ms</td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td><strong>11.391 ms</strong></td><td>3.043 ms</td><td><strong>14.433 ms</strong></td><td>14.769 ms</td></tr>
        <tr><td><code>torch</code></td><td>13.141 ms</td><td>3.046 ms</td><td>16.187 ms</td><td>27.256 ms</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>14.015 ms</td><td><strong>3.019 ms</strong></td><td>17.034 ms</td><td><strong>11.108 ms</strong></td></tr>
        <tr><td><code>torch_count</code></td><td>13.211 ms</td><td>3.196 ms</td><td>16.408 ms</td><td>29.091 ms</td></tr>
        <tr><td rowspan="5"><strong>262,144</strong></td><td><strong>CUDA baseline</strong></td><td>94.492 ms</td><td><strong>7.900 ms</strong></td><td>102.392 ms</td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td><strong>94.002 ms</strong></td><td>8.143 ms</td><td><strong>102.145 ms</strong></td><td>138.323 ms</td></tr>
        <tr><td><code>torch</code></td><td>96.775 ms</td><td>7.922 ms</td><td>104.697 ms</td><td>571.229 ms</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td>100.043 ms</td><td>8.416 ms</td><td>108.459 ms</td><td><strong>85.941 ms</strong></td></tr>
        <tr><td><code>torch_count</code></td><td>261.481 ms</td><td>597.331 ms</td><td>858.813 ms</td><td>689.049 ms</td></tr>
    </tbody>
</table>

### Public API peak memory + internal binning scratch

<table>
    <thead>
        <tr>
            <th>Points</th>
            <th>Backend / sort mode</th>
            <th>Warp FW peak</th>
            <th>Warp BW peak</th>
            <th>Internal binning peak</th>
        </tr>
    </thead>
    <tbody>
        <tr><td rowspan="5"><strong>4,096</strong></td><td><strong>CUDA baseline</strong></td><td><strong>1.57 MiB</strong></td><td><strong>4.41 MiB</strong></td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td><strong>1.57 MiB</strong></td><td>4.42 MiB</td><td><strong>0.00 MiB</strong></td></tr>
        <tr><td><code>torch</code></td><td><strong>1.57 MiB</strong></td><td>4.42 MiB</td><td>4.69 MiB</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>1.57 MiB</strong></td><td>4.42 MiB</td><td>0.02 MiB</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>1.57 MiB</strong></td><td>4.42 MiB</td><td>5.34 MiB</td></tr>
        <tr><td rowspan="5"><strong>16,384</strong></td><td><strong>CUDA baseline</strong></td><td><strong>5.18 MiB</strong></td><td><strong>17.63 MiB</strong></td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td><strong>5.18 MiB</strong></td><td>17.69 MiB</td><td><strong>0.00 MiB</strong></td></tr>
        <tr><td><code>torch</code></td><td><strong>5.18 MiB</strong></td><td>17.69 MiB</td><td>18.98 MiB</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>5.18 MiB</strong></td><td>17.69 MiB</td><td>0.06 MiB</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>5.18 MiB</strong></td><td>17.69 MiB</td><td>22.13 MiB</td></tr>
        <tr><td rowspan="5"><strong>65,536</strong></td><td><strong>CUDA baseline</strong></td><td><strong>41.79 MiB</strong></td><td><strong>70.57 MiB</strong></td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td><strong>41.79 MiB</strong></td><td>70.82 MiB</td><td><strong>0.00 MiB</strong></td></tr>
        <tr><td><code>torch</code></td><td><strong>41.79 MiB</strong></td><td>70.82 MiB</td><td>348.00 MiB</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>41.79 MiB</strong></td><td>70.82 MiB</td><td>0.25 MiB</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>41.79 MiB</strong></td><td>70.82 MiB</td><td>330.50 MiB</td></tr>
        <tr><td rowspan="5"><strong>262,144</strong></td><td><strong>CUDA baseline</strong></td><td><strong>302.13 MiB</strong></td><td><strong>282.00 MiB</strong></td><td>—</td></tr>
        <tr><td><code>warp_radix</code></td><td>303.13 MiB</td><td>284.00 MiB</td><td><strong>0.00 MiB</strong></td></tr>
        <tr><td><code>torch</code></td><td><strong>302.13 MiB</strong></td><td>283.00 MiB</td><td>3038.64 MiB</td></tr>
        <tr><td><code>warp_depth_stable_tile</code></td><td><strong>302.13 MiB</strong></td><td>283.00 MiB</td><td>1.00 MiB</td></tr>
        <tr><td><code>torch_count</code></td><td><strong>302.13 MiB</strong></td><td>284.32 MiB</td><td>2891.15 MiB</td></tr>
    </tbody>
</table>

From the public-API perspective, CUDA and all four Warp sort modes still sit in roughly the same forward peak-memory range, while CUDA remains consistently lower on backward peak memory. The real spread is still the **diagnostic internal binning scratch**: `warp_radix` and `warp_depth_stable_tile` remain tiny, whereas `torch` / `torch_count` grow sharply at 65K and 262K.



`warp_depth_stable_tile` is still kept as the default not because it is absolutely best on every metric, but because within the four Warp modes it continues to offer the most balanced mix of internal binning GPU time, near-zero scratch memory, and relatively small correctness residuals; if you only optimize end-to-end throughput, the best choice already changes with scale.

---

## Known Limitations

### 1. No Shared Memory (Warp Limitation)

NVIDIA Warp lacks an explicit and flexible way to control shared memory. Alignment, padding, and construction of complex structures are difficult, so CUDA `__shared__` memory cannot be handled freely. This prevents the cooperative tile-level fetch pattern used by the CUDA baseline, where 256 threads in a tile collaboratively load Gaussian data from global memory into shared memory and then iterate over the shared buffer. Instead, each thread reads from global memory independently, causing:

- many more global-memory transactions per tile for the same data
- poor memory coalescing (scattered reads indexed by Gaussian)
- NCU profiling to show that **almost all kernels are bottlenecked by memory**

### 2. No Warp-Level Intrinsics

Warp does not expose CUDA warp-level intrinsics (`__shfl_sync`, `__ballot_sync`, `__any_sync`). The available synchronization mechanisms are relatively limited and not compatible with some advanced patterns. This prevents:

- warp-level reductions for gradient accumulation
- warp-level voting for early termination
- cross-lane communication patterns used in advanced CUDA rasterizers

### 3. `tile_atomic_add` Only Supports Scalars

`wp.tile_atomic_add` only supports a few scalar operands such as `float32`. Vector- and matrix-level atomic reductions must be decomposed into separate scalar atomic operations, which introduces too much synchronization overhead. This has been verified experimentally — the tile-reduced backward kernel (C2) is **2–3× slower** than the non-Warp version and requires 10 scalar atomic operations per Gaussian.

### 4. Compile-Time Tile Shape

`BLOCK_X` and `BLOCK_Y` are defined as `wp.constant()` values (16×16). Changing the tile shape requires modifying the source code and triggering Warp module recompilation. The CUDA baseline also uses fixed tile sizes, but Warp's JIT model makes this limitation more obvious because the constants are baked into the kernel at compile time.

### 5. First-Run JIT Compilation Overhead

The first call to any Warp kernel triggers JIT compilation of the whole module. On a typical system, this takes several seconds (depending on the number of kernels and the GPU). Subsequent calls use the Warp kernel cache and complete almost instantly.

### 6. Backward Non-Determinism

The backward render kernel uses `wp.atomic_add` to accumulate gradients. When multiple threads write to the same address, this is inherently non-deterministic. This means:

- two runs with identical inputs may produce slightly different gradient values
- the difference is usually within FP32 precision (most gradients < 1e-4)
- in extreme cases, `conic_opacity` gradients can reach max-diff values as high as ~1.6e+05 because the gradient magnitude itself is large
- this is consistent with the CUDA baseline (`atomicAdd` is also non-deterministic)

### 7. Python-Level Orchestration Overhead

Unlike the CUDA baseline, where the full pipeline is orchestrated in C++ with very little Python interaction, the Warp backend uses Python to:

- allocate intermediate tensors (through PyTorch)
- launch Warp kernels sequentially
- pass data between stages through Python variables

This introduces measurable fixed overhead. It is significant at small scales and negligible at large scales. **This may cause large fluctuations in Warp training throughput.**

### 8. Single Backward Mode

Only `backward_mode="manual"` is supported. The CUDA baseline's `autograd`-level differentiation is not applicable because Warp kernels are not natively integrated into PyTorch's autograd graph — the backward pass is explicitly encoded with manually derived gradients.

---

## Future Optimization Directions

The following are the most impactful potential improvements, roughly ordered by expected benefit:

### 1. Shared-Memory Support

If NVIDIA Warp adds richer `__shared__` memory support in a future release, the render and backward-render kernels could be rewritten into cooperative tile-level fetch kernels, potentially closing the **2–3× gap** at small and medium scales. This is the single most important optimization. It is still difficult to achieve on the current Warp version.

### 2. Runtime Tile-Shape Adaptation

Currently `BLOCK_X=16, BLOCK_Y=16` is fixed at compile time. Allowing runtime selection of tile shapes (for example, 8×8 for small images and 32×16 for wide images) could improve occupancy and reduce tile-boundary overhead. This would require Warp to support dynamic kernel parameterization or template-like mechanisms.

### 3. TOP_K Externalization

The `TOP_K = 20` constant controls the maximum number of Gaussians considered per pixel before early termination. Exposing it as a runtime parameter would allow:

- lowering TOP_K to speed up training at acceptable quality loss
- increasing TOP_K for quality-critical rendering
- adapting TOP_K based on scene complexity

### 4. Warp-Level Intrinsics (Pending Future Warp Features)

If warp-level intrinsics become available in Warp, per-warp reduction could replace `atomic_add` in the backward kernel, potentially removing the non-determinism issue and reducing LG-throttle stalls.

---

## Acknowledgments

This Warp backend builds upon the following projects:

- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), by Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis (INRIA, MPII). The original CUDA rasterizer is the reference implementation.
- [Fast Converging 3D Gaussian Splatting for 1-Minute Reconstruction](https://arxiv.org/abs/2601.19489), by Ziyu Zhang, Tianle Liu, Diantao Tu, and Shuhan Shen. The tight AABB bounding-box technique used in this project is inspired by this work.
- [NVIDIA Warp](https://nvidia.github.io/warp/) — the Python framework for high-performance GPU simulation and compute.
- [PyTorch](https://pytorch.org/) — used for tensor management and CUDA memory allocation.


