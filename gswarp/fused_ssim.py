"""Warp kernel implementation of fused-ssim (SSIM, sigma=1.5, window=11).

Optimizations (following gswarp/_warp_backend.py patterns):
  C0 - wp.constant Gaussian weights: eliminate per-pixel array reads
  C4 - Launch object caching: record_cmd + set_param_at_index bypass Python overhead
  M1 - Flat 1D arrays with manual offsets: remove 4D stride arithmetic
  O3 - torch.empty for always-overwritten outputs: skip GPU memset

GPU kernels use NVIDIA Warp.  torch is used only for:
  - autograd Function wrapper
  - GPU memory allocation
No C++/CUDA compilation required.
"""

import torch
import warp as wp

from ._stream import ensure_aligned
from ._tuning import register_kernel_class, get_tuned_block_dim, FAMILY_MEMORY

wp.init()

__all__ = ["fused_ssim"]

# Register SSIM kernel classes with estimated register usage.
# These are 1D separable Gaussian convolution kernels — streaming memory-bound.
register_kernel_class("ssim_fwd_h", 48, FAMILY_MEMORY)
register_kernel_class("ssim_fwd_v", 64, FAMILY_MEMORY)
register_kernel_class("ssim_bwd_h", 48, FAMILY_MEMORY)
register_kernel_class("ssim_bwd_v", 48, FAMILY_MEMORY)

# ---------------------------------------------------------------------------
# C0: 11-tap Gaussian weights as compile-time constants (sigma=1.5)
# Eliminates per-pixel array reads; values become immediate operands.
# ---------------------------------------------------------------------------
G00 = wp.constant(wp.float32(0.001028380123898387))
G01 = wp.constant(wp.float32(0.0075987582094967365))
G02 = wp.constant(wp.float32(0.036000773310661316))
G03 = wp.constant(wp.float32(0.10936068743467331))
G04 = wp.constant(wp.float32(0.21300552785396576))
G05 = wp.constant(wp.float32(0.26601171493530273))
G06 = wp.constant(wp.float32(0.21300552785396576))
G07 = wp.constant(wp.float32(0.10936068743467331))
G08 = wp.constant(wp.float32(0.036000773310661316))
G09 = wp.constant(wp.float32(0.0075987582094967365))
G10 = wp.constant(wp.float32(0.001028380123898387))


@wp.func
def _gw(i: wp.int32) -> wp.float32:
    if i == 0:
        return G00
    if i == 1:
        return G01
    if i == 2:
        return G02
    if i == 3:
        return G03
    if i == 4:
        return G04
    if i == 5:
        return G05
    if i == 6:
        return G06
    if i == 7:
        return G07
    if i == 8:
        return G08
    if i == 9:
        return G09
    return G10


# ---------------------------------------------------------------------------
# C4++: Pre-recorded launch plan for fixed image shapes (training fast path)
#
# All workspace buffers and launch commands are allocated/recorded ONCE.
# Subsequent calls only update the 2-3 parameters that actually change
# (img1, img2, upstream grad), reducing set_param_at_index from ~37 to 5
# and eliminating per-call torch.empty / wp.from_torch for intermediates.
# ---------------------------------------------------------------------------

_PLAN_CACHE: dict = {}   # (dev, B, CH, H, W, padding) -> _SSIMPlan

# Inference launch caches (rarely used, kept lightweight)
_C4_FWD_H_INF: dict = {}
_C4_FWD_VI: dict = {}
_C4_FWD_VIV: dict = {}


def _launch_cached(cache, kernel, dim, inputs, device, block_dim=256):
    key = (device, dim)
    cmd = cache.get(key)
    if cmd is None:
        cmd = wp.launch(kernel=kernel, dim=dim, inputs=inputs,
                        device=device, record_cmd=True, block_dim=block_dim)
        cache[key] = cmd
    else:
        for i, v in enumerate(inputs):
            cmd.set_param_at_index(i, v)
    cmd.launch()


class _SSIMPlan:
    """Pre-recorded launch plan for fixed-shape SSIM training.

    All workspace buffers and launch commands are allocated/recorded once.
    Subsequent calls only update img1, img2, and upstream grad pointers.
    """

    def __init__(self, B, CH, H, W, C1, C2, padding, device):
        dev = str(device)
        N = B * CH * H * W
        valid = (padding == "valid")

        # --- Pre-allocate ALL workspace buffers ---
        # Shared buffer for H-pass forward and H-pass backward workspaces.
        #
        # Data flow (all sequential on one CUDA stream):
        #   fwd_h  → writes w_h[0..3]  (shared_buf[0..4N])
        #   fwd_v  → reads  w_h[0..3]; writes w_ssim, w_dm[0..2]
        #   bwd_h  → reads  w_dm[0..2]; writes w_t[0..2] (shared_buf[0..3N])
        #   bwd_v  → reads  w_t[0..2]; writes w_dL
        #
        # w_t aliases w_h[0..2]: safe because fwd_v completes before bwd_h
        # starts, so the H-pass values are never live when bwd_h overwrites them.
        # Saving: 3*N float32 elements vs. two separate allocations.
        shared_buf = torch.empty(4 * N, dtype=torch.float32, device=device)
        w_h = [wp.from_torch(shared_buf[i * N:(i + 1) * N]) for i in range(4)]
        w_t = [wp.from_torch(shared_buf[i * N:(i + 1) * N]) for i in range(3)]

        # dm workspace (3 x N) - written by fwd_v, read by bwd_h
        dm_buf = torch.empty(3 * N, dtype=torch.float32, device=device)
        w_dm = [wp.from_torch(dm_buf[i * N:(i + 1) * N]) for i in range(3)]
        self.t_dm = [dm_buf[i * N:(i + 1) * N].view(B, CH, H, W) for i in range(3)]

        # SSIM output
        if valid:
            Hv, Wv = H - 10, W - 10
            Nv = B * CH * Hv * Wv
            self.t_ssim = torch.empty(Nv, dtype=torch.float32, device=device)
        else:
            Nv = N
            Hv = Wv = 0
            self.t_ssim = torch.empty(N, dtype=torch.float32, device=device)
        w_ssim = wp.from_torch(self.t_ssim)

        # Backward output (dL/dimg1)
        self.t_dL = torch.empty(B, CH, H, W, dtype=torch.float32, device=device)
        w_dL = wp.from_torch(self.t_dL.view(-1))

        # Placeholder for upstream grad
        ph_up = wp.from_torch(torch.ones(1, dtype=torch.float32, device=device))

        # --- Record all 4 launch commands ---
        # Use a workspace array as placeholder for img1/img2 (same dtype/size)
        ph = w_h[0]

        bd_fwd_h = get_tuned_block_dim("ssim_fwd_h", device)
        bd_fwd_v = get_tuned_block_dim("ssim_fwd_v", device)
        bd_bwd_h = get_tuned_block_dim("ssim_bwd_h", device)
        bd_bwd_v = get_tuned_block_dim("ssim_bwd_v", device)

        # FWD_H: params [0]=img1, [1]=img2 change each call
        self._cmd_fwd_h = wp.launch(
            kernel=_fwd_h, dim=N,
            inputs=[ph, ph, w_h[0], w_h[1], w_h[2], w_h[3], W],
            device=dev, record_cmd=True, block_dim=bd_fwd_h)

        # FWD_V: ALL params fixed (workspace -> workspace)
        if valid:
            self._cmd_fwd_v = wp.launch(
                kernel=_fwd_v_train_valid, dim=Nv,
                inputs=[w_h[0], w_h[1], w_h[2], w_h[3],
                        w_ssim, w_dm[0], w_dm[1], w_dm[2],
                        H, W, Hv, Wv, float(C1), float(C2)],
                device=dev, record_cmd=True, block_dim=bd_fwd_v)
        else:
            self._cmd_fwd_v = wp.launch(
                kernel=_fwd_v_train, dim=N,
                inputs=[w_h[0], w_h[1], w_h[2], w_h[3],
                        w_ssim, w_dm[0], w_dm[1], w_dm[2],
                        H, W, float(C1), float(C2)],
                device=dev, record_cmd=True, block_dim=bd_fwd_v)

        # BWD_H: only upstream param changes
        if valid:
            N_valid = B * CH * (H - 10) * (W - 10)
            inv_N = 1.0 / float(N_valid)
            self._cmd_bwd_h = wp.launch(
                kernel=_bwd_h_scalar_valid, dim=N,
                inputs=[w_dm[0], w_dm[1], w_dm[2],
                        w_t[0], w_t[1], w_t[2],
                        H, W, inv_N, ph_up],
                device=dev, record_cmd=True, block_dim=bd_bwd_h)
            self._bwd_h_up_idx = 9
        else:
            inv_N = 1.0 / float(N)
            self._cmd_bwd_h = wp.launch(
                kernel=_bwd_h_scalar, dim=N,
                inputs=[w_dm[0], w_dm[1], w_dm[2],
                        w_t[0], w_t[1], w_t[2],
                        W, inv_N, ph_up],
                device=dev, record_cmd=True, block_dim=bd_bwd_h)
            self._bwd_h_up_idx = 8

        # BWD_V: params [0]=img1, [1]=img2 change each call
        self._cmd_bwd_v = wp.launch(
            kernel=_bwd_v, dim=N,
            inputs=[ph, ph, w_t[0], w_t[1], w_t[2], w_dL, H, W],
            device=dev, record_cmd=True, block_dim=bd_bwd_v)

        # Keep references to prevent GC of backing torch buffers
        # (Warp arrays hold raw CUDA pointers, not Python refs)
        self._refs = (shared_buf, dm_buf, ph_up)

    def forward(self, img1, img2):
        """Run forward. Returns (loss_scalar, w_img1, w_img2)."""
        w_img1 = wp.from_torch(img1.view(-1))
        w_img2 = wp.from_torch(img2.view(-1))

        self._cmd_fwd_h.set_param_at_index(0, w_img1)
        self._cmd_fwd_h.set_param_at_index(1, w_img2)
        self._cmd_fwd_h.launch()

        self._cmd_fwd_v.launch()

        return self.t_ssim.mean(), w_img1, w_img2

    def backward(self, w_img1, w_img2, opt_grad):
        """Run backward. Returns dL/dimg1 tensor."""
        w_up = wp.from_torch(opt_grad.reshape(1))

        self._cmd_bwd_h.set_param_at_index(self._bwd_h_up_idx, w_up)
        self._cmd_bwd_h.launch()

        self._cmd_bwd_v.set_param_at_index(0, w_img1)
        self._cmd_bwd_v.set_param_at_index(1, w_img2)
        self._cmd_bwd_v.launch()

        return self.t_dL


# ---------------------------------------------------------------------------
# Separable kernels
# ---------------------------------------------------------------------------

@wp.kernel
def _fwd_h(
    img1: wp.array(dtype=wp.float32),
    img2: wp.array(dtype=wp.float32),
    h0: wp.array(dtype=wp.float32),
    h1: wp.array(dtype=wp.float32),
    h2: wp.array(dtype=wp.float32),
    h3: wp.array(dtype=wp.float32),
    W: wp.int32,
):
    tid = wp.tid()
    x = tid % W
    row = tid - x

    a0 = wp.float32(0.0)
    a1 = wp.float32(0.0)
    a2 = wp.float32(0.0)
    a3 = wp.float32(0.0)

    # Gaussian symmetry: w[k] == w[10-k] for k=0..4
    for k in range(5):
        nl = x + k - 5
        nr = x + 5 - k
        g = _gw(k)

        if nl >= 0 and nl < W:
            il = row + nl
            v1l = img1[il]
            v2l = img2[il]
        else:
            v1l = wp.float32(0.0)
            v2l = wp.float32(0.0)

        if nr >= 0 and nr < W:
            ir = row + nr
            v1r = img1[ir]
            v2r = img2[ir]
        else:
            v1r = wp.float32(0.0)
            v2r = wp.float32(0.0)

        a0 = a0 + g * (v1l + v1r)
        a1 = a1 + g * (v2l + v2r)
        a2 = a2 + g * (v1l * v1l + v2l * v2l + v1r * v1r + v2r * v2r)
        a3 = a3 + g * (v1l * v2l + v1r * v2r)

    # Center tap (k=5)
    ic = row + x
    v1c = img1[ic]
    v2c = img2[ic]
    a0 = a0 + G05 * v1c
    a1 = a1 + G05 * v2c
    a2 = a2 + G05 * (v1c * v1c + v2c * v2c)
    a3 = a3 + G05 * v1c * v2c

    h0[tid] = a0
    h1[tid] = a1
    h2[tid] = a2
    h3[tid] = a3


@wp.kernel
def _fwd_v_train(
    h0: wp.array(dtype=wp.float32),
    h1: wp.array(dtype=wp.float32),
    h2: wp.array(dtype=wp.float32),
    h3: wp.array(dtype=wp.float32),
    ssim_out: wp.array(dtype=wp.float32),
    dm0: wp.array(dtype=wp.float32),
    dm1: wp.array(dtype=wp.float32),
    dm2: wp.array(dtype=wp.float32),
    H: wp.int32,
    W: wp.int32,
    C1: wp.float32,
    C2: wp.float32,
):
    tid = wp.tid()
    y = (tid / W) % H
    col = tid - y * W

    a0 = wp.float32(0.0)
    a1 = wp.float32(0.0)
    a2 = wp.float32(0.0)
    a3 = wp.float32(0.0)

    # Gaussian symmetry: w[k] == w[10-k] for k=0..4
    for k in range(5):
        nt = y + k - 5
        nb = y + 5 - k
        g = _gw(k)
        if nt >= 0 and nt < H:
            it = col + nt * W
            a0 = a0 + g * h0[it]
            a1 = a1 + g * h1[it]
            a2 = a2 + g * h2[it]
            a3 = a3 + g * h3[it]
        if nb >= 0 and nb < H:
            ib = col + nb * W
            a0 = a0 + g * h0[ib]
            a1 = a1 + g * h1[ib]
            a2 = a2 + g * h2[ib]
            a3 = a3 + g * h3[ib]

    # Center row (k=5)
    ic = col + y * W
    a0 = a0 + G05 * h0[ic]
    a1 = a1 + G05 * h1[ic]
    a2 = a2 + G05 * h2[ic]
    a3 = a3 + G05 * h3[ic]

    mu1 = a0
    mu2 = a1
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1mu2 = mu1 * mu2

    Av = mu1_sq + mu2_sq + C1
    Bv = a2 - mu1_sq - mu2_sq + C2
    Cv = wp.float32(2.0) * mu1mu2 + C1
    Dv = wp.float32(2.0) * (a3 - mu1mu2) + C2

    rA = wp.float32(1.0) / Av
    rB = wp.float32(1.0) / Bv
    rAB = rA * rB

    m = Cv * Dv * rAB
    ssim_out[tid] = m

    dm0[tid] = wp.float32(2.0) * (mu2 * (Dv - Cv) * rAB + mu1 * m * (rB - rA))
    dm1[tid] = -m * rB
    dm2[tid] = wp.float32(2.0) * Cv * rAB


@wp.kernel
def _fwd_v_infer(
    h0: wp.array(dtype=wp.float32),
    h1: wp.array(dtype=wp.float32),
    h2: wp.array(dtype=wp.float32),
    h3: wp.array(dtype=wp.float32),
    ssim_out: wp.array(dtype=wp.float32),
    H: wp.int32,
    W: wp.int32,
    C1: wp.float32,
    C2: wp.float32,
):
    tid = wp.tid()
    y = (tid / W) % H
    col = tid - y * W

    a0 = wp.float32(0.0)
    a1 = wp.float32(0.0)
    a2 = wp.float32(0.0)
    a3 = wp.float32(0.0)

    for k in range(5):
        nt = y + k - 5
        nb = y + 5 - k
        g = _gw(k)
        if nt >= 0 and nt < H:
            it = col + nt * W
            a0 = a0 + g * h0[it]
            a1 = a1 + g * h1[it]
            a2 = a2 + g * h2[it]
            a3 = a3 + g * h3[it]
        if nb >= 0 and nb < H:
            ib = col + nb * W
            a0 = a0 + g * h0[ib]
            a1 = a1 + g * h1[ib]
            a2 = a2 + g * h2[ib]
            a3 = a3 + g * h3[ib]

    ic = col + y * W
    a0 = a0 + G05 * h0[ic]
    a1 = a1 + G05 * h1[ic]
    a2 = a2 + G05 * h2[ic]
    a3 = a3 + G05 * h3[ic]

    mu1 = a0
    mu2 = a1
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1mu2 = mu1 * mu2

    Av = mu1_sq + mu2_sq + C1
    Bv = a2 - mu1_sq - mu2_sq + C2
    Cv = wp.float32(2.0) * mu1mu2 + C1
    Dv = wp.float32(2.0) * (a3 - mu1mu2) + C2

    ssim_out[tid] = (Cv * Dv) / (Av * Bv)


@wp.kernel
def _fwd_v_infer_valid(
    h0: wp.array(dtype=wp.float32),
    h1: wp.array(dtype=wp.float32),
    h2: wp.array(dtype=wp.float32),
    h3: wp.array(dtype=wp.float32),
    ssim_out: wp.array(dtype=wp.float32),
    H: wp.int32,
    W: wp.int32,
    Hv: wp.int32,
    Wv: wp.int32,
    C1: wp.float32,
    C2: wp.float32,
):
    tid = wp.tid()
    xv = tid % Wv
    yv = (tid / Wv) % Hv
    bc = tid / (Hv * Wv)
    y = yv + 5
    base_col = bc * H * W + xv + 5

    a0 = wp.float32(0.0)
    a1 = wp.float32(0.0)
    a2 = wp.float32(0.0)
    a3 = wp.float32(0.0)

    # All vertical neighbors guaranteed in [0, H-1] — no bounds check
    for k in range(5):
        nt = y + k - 5
        nb = y + 5 - k
        g = _gw(k)
        it = base_col + nt * W
        ib = base_col + nb * W
        a0 = a0 + g * (h0[it] + h0[ib])
        a1 = a1 + g * (h1[it] + h1[ib])
        a2 = a2 + g * (h2[it] + h2[ib])
        a3 = a3 + g * (h3[it] + h3[ib])

    ic = base_col + y * W
    a0 = a0 + G05 * h0[ic]
    a1 = a1 + G05 * h1[ic]
    a2 = a2 + G05 * h2[ic]
    a3 = a3 + G05 * h3[ic]

    mu1 = a0
    mu2 = a1
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1mu2 = mu1 * mu2

    Av = mu1_sq + mu2_sq + C1
    Bv = a2 - mu1_sq - mu2_sq + C2
    Cv = wp.float32(2.0) * mu1mu2 + C1
    Dv = wp.float32(2.0) * (a3 - mu1mu2) + C2

    ssim_out[tid] = (Cv * Dv) / (Av * Bv)


@wp.kernel
def _fwd_v_train_valid(
    h0: wp.array(dtype=wp.float32),
    h1: wp.array(dtype=wp.float32),
    h2: wp.array(dtype=wp.float32),
    h3: wp.array(dtype=wp.float32),
    ssim_out: wp.array(dtype=wp.float32),
    dm0: wp.array(dtype=wp.float32),
    dm1: wp.array(dtype=wp.float32),
    dm2: wp.array(dtype=wp.float32),
    H: wp.int32,
    W: wp.int32,
    Hv: wp.int32,
    Wv: wp.int32,
    C1: wp.float32,
    C2: wp.float32,
):
    tid = wp.tid()
    xv = tid % Wv
    yv = (tid / Wv) % Hv
    bc = tid / (Hv * Wv)
    y = yv + 5
    x = xv + 5
    base_col = bc * H * W + x

    a0 = wp.float32(0.0)
    a1 = wp.float32(0.0)
    a2 = wp.float32(0.0)
    a3 = wp.float32(0.0)

    # All vertical neighbors guaranteed in [0, H-1] — no bounds check
    for k in range(5):
        nt = y + k - 5
        nb = y + 5 - k
        g = _gw(k)
        it = base_col + nt * W
        ib = base_col + nb * W
        a0 = a0 + g * (h0[it] + h0[ib])
        a1 = a1 + g * (h1[it] + h1[ib])
        a2 = a2 + g * (h2[it] + h2[ib])
        a3 = a3 + g * (h3[it] + h3[ib])

    ic = base_col + y * W
    a0 = a0 + G05 * h0[ic]
    a1 = a1 + G05 * h1[ic]
    a2 = a2 + G05 * h2[ic]
    a3 = a3 + G05 * h3[ic]

    mu1 = a0
    mu2 = a1
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1mu2 = mu1 * mu2

    Av = mu1_sq + mu2_sq + C1
    Bv = a2 - mu1_sq - mu2_sq + C2
    Cv = wp.float32(2.0) * mu1mu2 + C1
    Dv = wp.float32(2.0) * (a3 - mu1mu2) + C2

    rA = wp.float32(1.0) / Av
    rB = wp.float32(1.0) / Bv
    rAB = rA * rB

    m = Cv * Dv * rAB
    ssim_out[tid] = m

    # Write dm* at full-image offset (backward reads at full-image positions)
    full_idx = bc * H * W + y * W + x
    dm0[full_idx] = wp.float32(2.0) * (mu2 * (Dv - Cv) * rAB + mu1 * m * (rB - rA))
    dm1[full_idx] = -m * rB
    dm2[full_idx] = wp.float32(2.0) * Cv * rAB


@wp.kernel
def _bwd_h_scalar(
    dm0: wp.array(dtype=wp.float32),
    dm1: wp.array(dtype=wp.float32),
    dm2: wp.array(dtype=wp.float32),
    t0: wp.array(dtype=wp.float32),
    t1: wp.array(dtype=wp.float32),
    t2: wp.array(dtype=wp.float32),
    W: wp.int32,
    inv_N: wp.float32,
    upstream: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    x = tid % W
    row = tid - x
    scale = inv_N * upstream[0]

    s0 = wp.float32(0.0)
    s1 = wp.float32(0.0)
    s2 = wp.float32(0.0)

    for k in range(5):
        nl = x + k - 5
        nr = x + 5 - k
        g_s = _gw(k) * scale
        if nl >= 0 and nl < W:
            il = row + nl
            s0 = s0 + g_s * dm0[il]
            s1 = s1 + g_s * dm1[il]
            s2 = s2 + g_s * dm2[il]
        if nr >= 0 and nr < W:
            ir = row + nr
            s0 = s0 + g_s * dm0[ir]
            s1 = s1 + g_s * dm1[ir]
            s2 = s2 + g_s * dm2[ir]

    g_s = G05 * scale
    ic = row + x
    s0 = s0 + g_s * dm0[ic]
    s1 = s1 + g_s * dm1[ic]
    s2 = s2 + g_s * dm2[ic]

    t0[tid] = s0
    t1[tid] = s1
    t2[tid] = s2


@wp.kernel
def _bwd_h_scalar_valid(
    dm0: wp.array(dtype=wp.float32),
    dm1: wp.array(dtype=wp.float32),
    dm2: wp.array(dtype=wp.float32),
    t0: wp.array(dtype=wp.float32),
    t1: wp.array(dtype=wp.float32),
    t2: wp.array(dtype=wp.float32),
    H: wp.int32,
    W: wp.int32,
    inv_N: wp.float32,
    upstream: wp.array(dtype=wp.float32),
):
    tid = wp.tid()
    x = tid % W
    y = (tid / W) % H
    row = tid - x

    # Valid region: y in [5, H-6], x in [5, W-6]
    if y < 5 or y >= H - 5:
        t0[tid] = wp.float32(0.0)
        t1[tid] = wp.float32(0.0)
        t2[tid] = wp.float32(0.0)
        return

    scale = inv_N * upstream[0]

    s0 = wp.float32(0.0)
    s1 = wp.float32(0.0)
    s2 = wp.float32(0.0)

    for k in range(5):
        nl = x + k - 5
        nr = x + 5 - k
        g = _gw(k)
        if nl >= 5 and nl < W - 5:
            il = row + nl
            g_s = g * scale
            s0 = s0 + g_s * dm0[il]
            s1 = s1 + g_s * dm1[il]
            s2 = s2 + g_s * dm2[il]
        if nr >= 5 and nr < W - 5:
            ir = row + nr
            g_s = g * scale
            s0 = s0 + g_s * dm0[ir]
            s1 = s1 + g_s * dm1[ir]
            s2 = s2 + g_s * dm2[ir]

    if x >= 5 and x < W - 5:
        ic = row + x
        g_s = G05 * scale
        s0 = s0 + g_s * dm0[ic]
        s1 = s1 + g_s * dm1[ic]
        s2 = s2 + g_s * dm2[ic]

    t0[tid] = s0
    t1[tid] = s1
    t2[tid] = s2


@wp.kernel
def _bwd_v(
    img1: wp.array(dtype=wp.float32),
    img2: wp.array(dtype=wp.float32),
    t0: wp.array(dtype=wp.float32),
    t1: wp.array(dtype=wp.float32),
    t2: wp.array(dtype=wp.float32),
    dL_dimg1: wp.array(dtype=wp.float32),
    H: wp.int32,
    W: wp.int32,
):
    tid = wp.tid()
    y = (tid / W) % H
    col = tid - y * W

    pix1 = img1[tid]
    pix2 = img2[tid]

    a0 = wp.float32(0.0)
    a1 = wp.float32(0.0)
    a2 = wp.float32(0.0)

    # Gaussian symmetry: w[k] == w[10-k] for k=0..4
    for k in range(5):
        nt = y + k - 5
        nb = y + 5 - k
        g = _gw(k)
        if nt >= 0 and nt < H:
            it = col + nt * W
            a0 = a0 + g * t0[it]
            a1 = a1 + g * t1[it]
            a2 = a2 + g * t2[it]
        if nb >= 0 and nb < H:
            ib = col + nb * W
            a0 = a0 + g * t0[ib]
            a1 = a1 + g * t1[ib]
            a2 = a2 + g * t2[ib]

    # Center row (k=5)
    ic = col + y * W
    a0 = a0 + G05 * t0[ic]
    a1 = a1 + G05 * t1[ic]
    a2 = a2 + G05 * t2[ic]

    dL_dimg1[tid] = a0 + wp.float32(2.0) * pix1 * a1 + pix2 * a2


# ---------------------------------------------------------------------------
# PyTorch autograd Function
# ---------------------------------------------------------------------------

class FusedSSIMMap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C1, C2, img1, img2, padding, train):
        B, CH, H, W = img1.shape
        device = img1.device
        dev = str(device)
        N = B * CH * H * W

        img1 = img1.contiguous()
        img2 = img2.contiguous()

        if train:
            # --- Fast plan-based path: only update img1/img2 pointers ---
            key = (dev, B, CH, H, W, padding)
            plan = _PLAN_CACHE.get(key)
            if plan is None:
                plan = _SSIMPlan(B, CH, H, W, C1, C2, padding, device)
                _PLAN_CACHE[key] = plan

            loss, w_img1, w_img2 = plan.forward(img1, img2)

            ctx.plan = plan
            ctx.w_img1 = w_img1
            ctx.w_img2 = w_img2
            ctx.save_for_backward(img1.detach(), img2)  # prevent data GC
            return loss

        # --- Inference path (rare) ---
        w_img1 = wp.from_torch(img1.view(-1))
        w_img2 = wp.from_torch(img2.view(-1))

        h_buf = torch.empty(4 * N, dtype=torch.float32, device=device)
        w_h = [wp.from_torch(h_buf[i * N:(i + 1) * N]) for i in range(4)]

        _launch_cached(_C4_FWD_H_INF, _fwd_h, N,
                       [w_img1, w_img2,
                        w_h[0], w_h[1], w_h[2], w_h[3], W], dev,
                       block_dim=get_tuned_block_dim("ssim_fwd_h", dev))

        if padding == "valid":
            Hv, Wv = H - 10, W - 10
            Nv = B * CH * Hv * Wv
            ssim_flat = torch.empty(Nv, dtype=torch.float32, device=device)
            w_ssim = wp.from_torch(ssim_flat)
            _launch_cached(_C4_FWD_VIV, _fwd_v_infer_valid, Nv,
                           [w_h[0], w_h[1], w_h[2], w_h[3],
                            w_ssim, H, W, Hv, Wv,
                            float(C1), float(C2)], dev,
                           block_dim=get_tuned_block_dim("ssim_fwd_v", dev))
            return ssim_flat.mean()
        else:
            ssim_map = torch.empty_like(img1)
            w_ssim = wp.from_torch(ssim_map.view(-1))
            _launch_cached(_C4_FWD_VI, _fwd_v_infer, N,
                           [w_h[0], w_h[1], w_h[2], w_h[3],
                            w_ssim, H, W, float(C1), float(C2)], dev,
                           block_dim=get_tuned_block_dim("ssim_fwd_v", dev))
            return ssim_map.mean()

    @staticmethod
    def backward(ctx, opt_grad):
        plan = ctx.plan
        dL_dimg1 = plan.backward(ctx.w_img1, ctx.w_img2, opt_grad)
        return None, None, dL_dimg1, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ALLOWED_PADDING = ("same", "valid")


def fused_ssim(img1, img2, padding="same", train=True):
    """Compute mean SSIM between *img1* and *img2*.

    Drop-in replacement for ``fused_ssim.fused_ssim`` (CUDA version).

    Parameters
    ----------
    img1, img2 : torch.Tensor
        Shape ``(B, C, H, W)``, dtype ``float32``, on a CUDA device.
    padding : str
        ``"same"`` (default) or ``"valid"``.
    train : bool
        If True, saves intermediate tensors needed for backward.

    Returns
    -------
    torch.Tensor
        Scalar - mean SSIM across all pixels.
    """
    ensure_aligned()
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    assert padding in _ALLOWED_PADDING
    return FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)
