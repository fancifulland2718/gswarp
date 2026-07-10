# Profiled Performance Backlog

**Status:** deferred optimization work, not a release claim.

This backlog records candidates found after the rasterizer architecture split at
commit `fd68fb8`. Raw Nsight reports and benchmark logs are deliberately kept
outside Git because they are machine-specific generated artifacts.

## Evidence Scope

- Hardware: RTX 5090 (sm_120), PyTorch 2.11.0+cu130, Warp 1.12.0.
- Production path: standard 3DGS Lego with the normal
  `GaussianModel`/`Scene`/renderer training path, Warp rasterizer, Warp SSIM,
  and Warp KNN.
- Valid 50k Lego comparison against native CUDA: Warp is 2.99% slower end to
  end and 4.11% slower in steady iterations. The delta is concentrated in
  backward (+0.525 ms) and loss (+0.086 ms); residual non-phase time is about
  5.93 ms for both CUDA and Warp.
- Nsight Compute samples include a 301,290-Gaussian saved-model render and
  short production training. Nsight Systems traced five training iterations
  plus Warp warm-up, so one-time initialization rows are not steady-state
  optimization evidence.

## Priority Backlog

### PERF-001: Backward Tile Render (P0)

**Owner:** `gswarp/_internal/backends/warp/backward_kernels.py`,
`_backward_render_tiles_warp32_kernel`.

- Nsight Systems: 46.6% of traced GPU kernel time, 2.891 ms average over six
  calls.
- Nsight Compute: 3.053 ms, block size 32, 61 registers/thread, 46.19%
  achieved occupancy versus 50% theoretical, 84.70% memory throughput and
  87.34% L1/TEX throughput.

Investigate per-tile workload imbalance and early exit, repeated tile
extraction/reduction in reverse traversal, and gradient accumulation traffic.
Require a native-CUDA 50k win on Lego and Truck with no quality or peak-memory
regression.

### PERF-002: Forward Tile Render at Production Scale (P1)

**Owner:** `gswarp/_internal/backends/warp/render_kernels.py`,
`_render_tiles_tiled256_warp_kernel`.

- 301,290-Gaussian Lego render: 0.725 ms, 50.87% achieved occupancy versus
  83.33% theoretical, with low compute (23.10%) and memory (11.76%) use.
- It is not the current end-to-end regression: Warp render phase is slightly
  faster than CUDA in the valid 50k comparison.

Investigate sparse versus dense tile occupancy and batch size 256 alternatives.
Do not add tile variants without a large-scene profile and end-to-end win.

### PERF-003: Fused Preprocess Paths (P1)

**Owners:** `preprocess_kernels.py` and `backward_kernels.py`.

- Forward preprocess: 65.78% L2/memory throughput, 10.89% compute throughput,
  32.91% achieved occupancy versus 75% theoretical.
- Backward preprocess: 67.75% DRAM/memory throughput, 11.96% compute
  throughput, 35.11% achieved occupancy versus 66.67% theoretical.

Absolute duration is small in this 100k-Gaussian trace. Investigate global
intermediate traffic, structure-of-arrays packing, and recompute versus storage
only after a 300k+ reprofile.

### PERF-004: Warp SSIM Backward (P1)

**Owner:** `gswarp/fused_ssim.py`, `_bwd_v`.

- 50k loss phase is 15.4% slower than CUDA fused SSIM (+0.086 ms).
- Nsight Compute: 45.12 us, 89.60% memory/L2 throughput, 89.19% achieved
  occupancy.

This is memory-saturated. Investigate vertical-pass intermediate traffic only
if Warp can express safe on-chip reuse; block-size tuning alone is unlikely to
help.

## Guardrails

- Keep generated wheels, logs, Nsight reports, and helper scripts out of Git.
- Rebuild a clean isolated package before every comparison and record Git SHA,
  wheel SHA-256, import path, and backend choices in benchmark metadata.
