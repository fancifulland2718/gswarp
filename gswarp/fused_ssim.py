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

from collections import OrderedDict
from threading import RLock
import weakref

import torch
import warp as wp

from ._stream import (
    TransientLaunchArrayScope,
    current_execution_context,
    execution_context,
    submission_guard,
    torch_launch_array,
)
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
# Per-autograd-node training workspace.
#
# A forward and its backward must own the same intermediate tensors. A global
# plan keyed only by tensor shape lets a second forward overwrite the first
# forward's workspace before autograd reaches it, so training plans are not
# shared across live forward calls.
# ---------------------------------------------------------------------------

# Inference launch caches (rarely used, kept lightweight)
_C4_FWD_H_INF: dict = {}
_C4_FWD_VI: dict = {}
_C4_FWD_VIV: dict = {}
_TRAIN_PLAN_POOL: OrderedDict[tuple, list] = OrderedDict()
_TRAIN_PLAN_POOL_LOCK = RLock()
_MAX_FREE_TRAIN_PLANS_PER_KEY = 2
_MAX_FREE_TRAIN_PLANS_PER_DEVICE = 4
_TRAIN_PLAN_CREATIONS = 0
_TRAIN_PLAN_REUSES = 0
_TRAIN_PLAN_ACTIVE_LEASES = 0


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
    """Private training workspace and launch plan for one SSIM forward.

    The owning autograd context keeps this object alive until its backward
    call, preventing another forward from replacing its saved intermediates.
    """

    def __init__(
        self,
        B,
        CH,
        H,
        W,
        C1,
        C2,
        padding,
        device,
        *,
        block_dims,
        warp_stream,
    ):
        dev = str(device)
        N = B * CH * H * W
        valid = (padding == "valid")
        self.warp_stream = warp_stream

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
        w_h = [torch_launch_array(shared_buf[i * N:(i + 1) * N]) for i in range(4)]
        w_t = [torch_launch_array(shared_buf[i * N:(i + 1) * N]) for i in range(3)]

        # dm workspace (3 x N) - written by fwd_v, read by bwd_h
        dm_buf = torch.empty(3 * N, dtype=torch.float32, device=device)
        w_dm = [torch_launch_array(dm_buf[i * N:(i + 1) * N]) for i in range(3)]
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
        w_ssim = torch_launch_array(self.t_ssim)

        # Placeholder for upstream grad
        ph_up = torch_launch_array(torch.ones(1, dtype=torch.float32, device=device))

        # --- Record all 4 launch commands ---
        # Use a workspace array as placeholder for img1/img2 (same dtype/size)
        ph = w_h[0]

        bd_fwd_h, bd_fwd_v, bd_bwd_h, bd_bwd_v = block_dims

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
            inputs=[ph, ph, w_t[0], w_t[1], w_t[2], w_h[3], H, W],
            device=dev, record_cmd=True, block_dim=bd_bwd_v)

        # Keep references to prevent GC of backing torch buffers
        # (Warp arrays hold raw CUDA pointers, not Python refs)
        self._refs = (shared_buf, dm_buf, ph_up)
        self._output_shape = (B, CH, H, W)
        self._device = device

    def forward(self, img1, img2):
        """Run forward. Returns (loss_scalar, w_img1, w_img2)."""
        dynamic_arrays = TransientLaunchArrayScope()
        w_img1 = dynamic_arrays.array(img1.view(-1))
        w_img2 = dynamic_arrays.array(img2.view(-1))

        self._cmd_fwd_h.set_param_at_index_from_ctype(0, w_img1)
        self._cmd_fwd_h.set_param_at_index_from_ctype(1, w_img2)
        self._cmd_fwd_h.launch()

        self._cmd_fwd_v.launch()

        return self.t_ssim.mean(), w_img1, w_img2

    def backward(self, w_img1, w_img2, opt_grad):
        """Run backward. Returns dL/dimg1 tensor."""
        dynamic_arrays = TransientLaunchArrayScope()
        w_up = dynamic_arrays.array(opt_grad.reshape(1))
        t_dL = torch.empty(
            self._output_shape, dtype=torch.float32, device=self._device
        )
        w_dL = dynamic_arrays.array(t_dL.view(-1))

        self._cmd_bwd_h.set_param_at_index_from_ctype(
            self._bwd_h_up_idx, w_up
        )
        self._cmd_bwd_h.launch()

        self._cmd_bwd_v.set_param_at_index_from_ctype(0, w_img1)
        self._cmd_bwd_v.set_param_at_index_from_ctype(1, w_img2)
        self._cmd_bwd_v.set_param_at_index_from_ctype(5, w_dL)
        self._cmd_bwd_v.launch()

        return t_dL


class _SSIMPlanLease:
    """Keep one internal plan private for the lifetime of an autograd graph."""

    __slots__ = ("plan", "_finalizer", "__weakref__")

    def __init__(self, key, plan):
        self.plan = plan
        self._finalizer = weakref.finalize(
            self, _release_training_plan, key, plan
        )
        self._finalizer.atexit = False


def _training_plan_key(B, CH, H, W, C1, C2, padding, device):
    context = current_execution_context(device)
    if context is None:
        raise RuntimeError(
            "SSIM training plans require an active CUDA execution context"
        )
    block_dims = (
        get_tuned_block_dim("ssim_fwd_h", device),
        get_tuned_block_dim("ssim_fwd_v", device),
        get_tuned_block_dim("ssim_bwd_h", device),
        get_tuned_block_dim("ssim_bwd_v", device),
    )
    key = (
        str(context.device),
        context.stream_handle,
        B,
        CH,
        H,
        W,
        float(C1),
        float(C2),
        padding,
        *block_dims,
    )
    return key, block_dims, context.warp_stream


def _acquire_training_plan(B, CH, H, W, C1, C2, padding, device):
    global _TRAIN_PLAN_ACTIVE_LEASES
    global _TRAIN_PLAN_CREATIONS
    global _TRAIN_PLAN_REUSES

    key, block_dims, warp_stream = _training_plan_key(
        B, CH, H, W, C1, C2, padding, device
    )
    with _TRAIN_PLAN_POOL_LOCK:
        bucket = _TRAIN_PLAN_POOL.get(key)
        if bucket:
            plan = bucket.pop()
            if not bucket:
                del _TRAIN_PLAN_POOL[key]
            _TRAIN_PLAN_REUSES += 1
        else:
            plan = None
        _TRAIN_PLAN_ACTIVE_LEASES += 1

    if plan is None:
        try:
            plan = _SSIMPlan(
                B,
                CH,
                H,
                W,
                C1,
                C2,
                padding,
                device,
                block_dims=block_dims,
                warp_stream=warp_stream,
            )
        except Exception:
            with _TRAIN_PLAN_POOL_LOCK:
                _TRAIN_PLAN_ACTIVE_LEASES = max(
                    0, _TRAIN_PLAN_ACTIVE_LEASES - 1
                )
            raise
        with _TRAIN_PLAN_POOL_LOCK:
            _TRAIN_PLAN_CREATIONS += 1
    return _SSIMPlanLease(key, plan)


def _release_training_plan(key, plan):
    global _TRAIN_PLAN_ACTIVE_LEASES

    evicted = []
    with _TRAIN_PLAN_POOL_LOCK:
        _TRAIN_PLAN_ACTIVE_LEASES = max(0, _TRAIN_PLAN_ACTIVE_LEASES - 1)
        bucket = _TRAIN_PLAN_POOL.setdefault(key, [])
        if len(bucket) < _MAX_FREE_TRAIN_PLANS_PER_KEY:
            bucket.append(plan)
            _TRAIN_PLAN_POOL.move_to_end(key)
        else:
            evicted.append(plan)

        device_key = key[0]
        while sum(
            len(plans)
            for pool_key, plans in _TRAIN_PLAN_POOL.items()
            if pool_key[0] == device_key
        ) > _MAX_FREE_TRAIN_PLANS_PER_DEVICE:
            for pool_key in tuple(_TRAIN_PLAN_POOL):
                if pool_key[0] != device_key:
                    continue
                plans = _TRAIN_PLAN_POOL[pool_key]
                evicted.append(plans.pop(0))
                if not plans:
                    del _TRAIN_PLAN_POOL[pool_key]
                break

    if evicted:
        with submission_guard(key[0]):
            for retired in evicted:
                if retired.warp_stream is not None:
                    wp.synchronize_stream(retired.warp_stream)


def _clear_training_plan_pool(device_key=None):
    retired = []
    with _TRAIN_PLAN_POOL_LOCK:
        for key in tuple(_TRAIN_PLAN_POOL):
            if device_key is None or key[0] == device_key:
                retired.extend(_TRAIN_PLAN_POOL.pop(key))
    synchronized = set()
    for plan in retired:
        stream = plan.warp_stream
        if stream is not None and id(stream) not in synchronized:
            wp.synchronize_stream(stream)
            synchronized.add(id(stream))


def _get_fused_ssim_cache_report():
    with _TRAIN_PLAN_POOL_LOCK:
        free_by_stream = {
            f"{key[0]}@{key[1]}": sum(
                len(plans)
                for pool_key, plans in _TRAIN_PLAN_POOL.items()
                if pool_key[:2] == key[:2]
            )
            for key in _TRAIN_PLAN_POOL
        }
        return {
            "active_training_leases": _TRAIN_PLAN_ACTIVE_LEASES,
            "free_training_plans": sum(
                len(plans) for plans in _TRAIN_PLAN_POOL.values()
            ),
            "training_plan_creations": _TRAIN_PLAN_CREATIONS,
            "training_plan_reuses": _TRAIN_PLAN_REUSES,
            "free_training_plans_by_stream": free_by_stream,
            "free_plan_limit_per_key": _MAX_FREE_TRAIN_PLANS_PER_KEY,
            "free_plan_limit_per_device": _MAX_FREE_TRAIN_PLANS_PER_DEVICE,
        }


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
        with execution_context(img1.device):
            return FusedSSIMMap._forward_impl(
                ctx, C1, C2, img1, img2, padding, train
            )

    @staticmethod
    def _forward_impl(ctx, C1, C2, img1, img2, padding, train):
        B, CH, H, W = img1.shape
        device = img1.device
        dev = str(device)
        N = B * CH * H * W

        img1 = img1.contiguous()
        img2 = img2.contiguous()
        ctx.train = train

        if train:
            lease = _acquire_training_plan(
                B, CH, H, W, C1, C2, padding, device
            )
            plan = lease.plan
            loss, w_img1, w_img2 = plan.forward(img1, img2)

            ctx.plan_lease = lease
            ctx.w_img1 = w_img1
            ctx.w_img2 = w_img2
            ctx.save_for_backward(img1.detach(), img2)  # prevent data GC
            return loss

        # --- Inference path (rare) ---
        w_img1 = torch_launch_array(img1.view(-1))
        w_img2 = torch_launch_array(img2.view(-1))

        h_buf = torch.empty(4 * N, dtype=torch.float32, device=device)
        w_h = [torch_launch_array(h_buf[i * N:(i + 1) * N]) for i in range(4)]

        _launch_cached(_C4_FWD_H_INF, _fwd_h, N,
                       [w_img1, w_img2,
                        w_h[0], w_h[1], w_h[2], w_h[3], W], dev,
                       block_dim=get_tuned_block_dim("ssim_fwd_h", dev))

        if padding == "valid":
            Hv, Wv = H - 10, W - 10
            Nv = B * CH * Hv * Wv
            ssim_flat = torch.empty(Nv, dtype=torch.float32, device=device)
            w_ssim = torch_launch_array(ssim_flat)
            _launch_cached(_C4_FWD_VIV, _fwd_v_infer_valid, Nv,
                           [w_h[0], w_h[1], w_h[2], w_h[3],
                            w_ssim, H, W, Hv, Wv,
                            float(C1), float(C2)], dev,
                           block_dim=get_tuned_block_dim("ssim_fwd_v", dev))
            return ssim_flat.mean()
        else:
            ssim_map = torch.empty_like(img1)
            w_ssim = torch_launch_array(ssim_map.view(-1))
            _launch_cached(_C4_FWD_VI, _fwd_v_infer, N,
                           [w_h[0], w_h[1], w_h[2], w_h[3],
                            w_ssim, H, W, float(C1), float(C2)], dev,
                           block_dim=get_tuned_block_dim("ssim_fwd_v", dev))
            return ssim_map.mean()

    @staticmethod
    def backward(ctx, opt_grad):
        device = ctx.saved_tensors[0].device if ctx.train else opt_grad.device
        with execution_context(device):
            return FusedSSIMMap._backward_impl(ctx, opt_grad)

    @staticmethod
    def _backward_impl(ctx, opt_grad):
        if not ctx.train:
            return None, None, None, None, None, None
        plan = ctx.plan_lease.plan
        dL_dimg1 = plan.backward(ctx.w_img1, ctx.w_img2, opt_grad)
        return None, None, dL_dimg1, None, None, None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_ALLOWED_PADDING = ("same", "valid")


def _validate_fused_ssim_inputs(img1, img2, padding, train) -> None:
    if not isinstance(img1, torch.Tensor) or not isinstance(img2, torch.Tensor):
        raise ValueError("img1 and img2 must be torch.Tensor instances")
    if img1.ndim != 4 or img2.ndim != 4:
        raise ValueError("img1 and img2 must have shape (B, C, H, W)")
    if img1.shape != img2.shape:
        raise ValueError(f"img1 and img2 must have the same shape, got {tuple(img1.shape)} and {tuple(img2.shape)}")
    if not img1.is_cuda or not img2.is_cuda or img1.device != img2.device:
        raise ValueError("img1 and img2 must be on the same CUDA device")
    if img1.dtype != torch.float32 or img2.dtype != torch.float32:
        raise ValueError("img1 and img2 must have dtype torch.float32")
    if padding not in _ALLOWED_PADDING:
        raise ValueError(f"padding must be one of {_ALLOWED_PADDING}, got {padding!r}")
    if padding == "valid" and (img1.shape[2] < 11 or img1.shape[3] < 11):
        raise ValueError("padding='valid' requires image height and width of at least 11")
    if not isinstance(train, bool):
        raise ValueError("train must be a bool")


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
    _validate_fused_ssim_inputs(img1, img2, padding, train)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    return FusedSSIMMap.apply(C1, C2, img1, img2, padding, train)


def clear_fused_ssim_caches(device=None) -> None:
    """Clear reusable SSIM plans and inference launches."""

    caches = (_C4_FWD_H_INF, _C4_FWD_VI, _C4_FWD_VIV)
    resolved_device = None if device is None else torch.device(device)
    if (
        resolved_device is not None
        and resolved_device.type == "cuda"
        and resolved_device.index is None
        and torch.cuda.is_available()
    ):
        resolved_device = torch.device("cuda", torch.cuda.current_device())
    device_key = None if resolved_device is None else str(resolved_device)
    known_devices = {
        key[0]
        for cache in caches
        for key in cache
        if isinstance(key, tuple) and isinstance(key[0], str)
    }
    with _TRAIN_PLAN_POOL_LOCK:
        known_devices.update(key[0] for key in _TRAIN_PLAN_POOL)
    targets = known_devices if device_key is None else {device_key}
    for target in sorted(targets):
        with submission_guard(target):
            _clear_training_plan_pool(target)
            for cache in caches:
                for key in tuple(cache):
                    if key[0] == target:
                        del cache[key]
