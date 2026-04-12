"""Shared GPU auto-tuning: device queries, occupancy estimation, block_dim selection.

Extracted from ``_rasterizer.py`` so that SSIM, KNN, and rasterizer kernels can
all share the same occupancy-based block_dim plan, without redundant device queries.

Multi-GPU design
----------------
Results are keyed by concrete device string (e.g. ``"cuda:0"``, ``"cuda:1"``).
Each device gets an independent tuning plan.  ``get_tuned_block_dim`` accepts an
optional *device* argument to pick the right plan; when omitted it uses the
PyTorch current CUDA device.

Usage::

    from ._tuning import get_tuned_block_dim, initialize_tuning

    # Once at startup (called by rasterizer; SSIM and KNN inherit for free):
    initialize_tuning(device)

    # Per-kernel launch (pass device explicitly for multi-GPU safety):
    block = get_tuned_block_dim("ssim_fwd_h", device)
    wp.launch(kernel, dim=N, ..., block_dim=block)
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = [
    "SM_ARCHITECTURE_PROPS",
    "FAMILY_COMPUTE",
    "FAMILY_MEMORY",
    "FAMILY_ATOMIC",
    "FAMILY_LATENCY",
    "FAMILY_WARP_SPECIALIZED",
    "normalize_device",
    "query_device_info",
    "query_sm_properties",
    "estimate_occupancy",
    "recommend_block_dim",
    "register_kernel_class",
    "get_tuned_block_dim",
    "initialize_tuning",
    "get_canonical_device",
]

# ---------------------------------------------------------------------------
# SM architecture property table — keyed by (major, minor) CC.
# Sources: NVIDIA CUDA Programming Guide Table 15, architecture whitepapers.
# ---------------------------------------------------------------------------

SM_ARCHITECTURE_PROPS: dict[tuple[int, int], dict[str, int]] = {
    # Volta
    (7, 0): {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32,
             "shared_mem_per_sm_bytes": 98304, "l2_cache_bytes": 6291456, "reg_alloc_unit": 256},
    # Turing
    (7, 5): {"regs_per_sm": 65536, "max_warps_per_sm": 32, "max_blocks_per_sm": 16,
             "shared_mem_per_sm_bytes": 65536, "l2_cache_bytes": 4194304, "reg_alloc_unit": 256},
    # Ampere GA100
    (8, 0): {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32,
             "shared_mem_per_sm_bytes": 167936, "l2_cache_bytes": 41943040, "reg_alloc_unit": 256},
    # Ampere GA10x
    (8, 6): {"regs_per_sm": 65536, "max_warps_per_sm": 48, "max_blocks_per_sm": 16,
             "shared_mem_per_sm_bytes": 102400, "l2_cache_bytes": 4194304, "reg_alloc_unit": 256},
    # Ada Lovelace
    (8, 9): {"regs_per_sm": 65536, "max_warps_per_sm": 48, "max_blocks_per_sm": 24,
             "shared_mem_per_sm_bytes": 102400, "l2_cache_bytes": 33554432, "reg_alloc_unit": 256},
    # Hopper
    (9, 0): {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32,
             "shared_mem_per_sm_bytes": 233472, "l2_cache_bytes": 52428800, "reg_alloc_unit": 256},
    # Blackwell
    (10, 0): {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32,
              "shared_mem_per_sm_bytes": 233472, "l2_cache_bytes": 67108864, "reg_alloc_unit": 256},
}

# ---------------------------------------------------------------------------
# Kernel family constants — characterise each kernel's performance profile.
# ---------------------------------------------------------------------------

FAMILY_COMPUTE: str = "compute_bound"
FAMILY_MEMORY: str = "memory_bound"
FAMILY_ATOMIC: str = "atomic_bound"
FAMILY_LATENCY: str = "latency_bound"
FAMILY_WARP_SPECIALIZED: str = "warp_specialized"

# ---------------------------------------------------------------------------
# Kernel register estimates (empirical).  Modules register their own classes
# via ``register_kernel_class()``.
# ---------------------------------------------------------------------------

_KERNEL_REG_ESTIMATES: dict[str, int] = {
    "default": 96,
}
_KERNEL_FAMILY: dict[str, str] = {
    "default": FAMILY_COMPUTE,
}
_KERNEL_FIXED_BLOCK_DIM: dict[str, int] = {}

# ---------------------------------------------------------------------------
# Per-device plan storage.
#   _TUNED_BLOCK_DIM_BY_DEVICE[dev_key][kernel_class] = block_dim
#   _TUNING_CACHE[dev_key] = full report dict
# ---------------------------------------------------------------------------

_TUNED_BLOCK_DIM_BY_DEVICE: dict[str, dict[str, int]] = {}
_TUNING_CACHE: dict[str, dict[str, Any]] = {}
# Set of devices for which verbose output has already been printed.
_TUNING_LOGGED_DEVICES: set[str] = set()
# The first device explicitly passed to initialize_tuning().  Used as the
# fallback when callers do not specify a device, so that all gswarp modules
# default to the same GPU as the external training framework.
_CANONICAL_DEVICE: torch.device | None = None


# ---------------------------------------------------------------------------
# Device queries
# ---------------------------------------------------------------------------

def normalize_device(device: torch.device | str | None = None) -> torch.device:
    """Resolve *device* to a concrete ``torch.device``.

    Resolution order when *device* is ``None``:
    1. The canonical device set by the first ``initialize_tuning(device)`` call.
    2. ``torch.cuda.current_device()`` (PyTorch default).
    """
    if device is None:
        if _CANONICAL_DEVICE is not None:
            return _CANONICAL_DEVICE
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def query_device_info(device: torch.device) -> tuple[Any | None, int | None, int | None, torch.device]:
    """Return (device_props, free_bytes, total_bytes, normalized_device)."""
    device_props = None
    free_memory = None
    total_memory = None
    normalized = device
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        normalized = torch.device("cuda", idx)
        device_props = torch.cuda.get_device_properties(idx)
        free_memory, total_memory = torch.cuda.mem_get_info(idx)
    return device_props, free_memory, total_memory, normalized


def _get_sm_arch_props(major: int, minor: int) -> dict[str, int] | None:
    props = SM_ARCHITECTURE_PROPS.get((major, minor))
    if props is not None:
        return dict(props)
    for (m, n), p in sorted(SM_ARCHITECTURE_PROPS.items(), reverse=True):
        if m <= major:
            return dict(p)
    return None


def query_sm_properties(device_props: Any | None) -> dict[str, Any]:
    """Build a comprehensive SM properties dict from *device_props*."""
    if device_props is None:
        return {}
    major, minor = int(device_props.major), int(device_props.minor)
    arch = _get_sm_arch_props(major, minor)
    max_threads_per_sm = int(getattr(device_props, "max_threads_per_multi_processor", 0))
    warp_size = int(getattr(device_props, "warp_size", 32))
    sm_count = int(device_props.multi_processor_count)
    sm_props: dict[str, Any] = {
        "compute_capability": f"{major}.{minor}",
        "sm_count": sm_count,
        "warp_size": warp_size,
        "max_threads_per_sm": max_threads_per_sm,
    }
    if arch is not None:
        sm_props.update(arch)
    else:
        sm_props["regs_per_sm"] = 65536
        sm_props["max_warps_per_sm"] = max_threads_per_sm // warp_size if max_threads_per_sm > 0 else 48
        sm_props["max_blocks_per_sm"] = 16
        sm_props["shared_mem_per_sm_bytes"] = 0
        sm_props["l2_cache_bytes"] = 0
        sm_props["reg_alloc_unit"] = 256
    return sm_props


# ---------------------------------------------------------------------------
# Occupancy estimation
# ---------------------------------------------------------------------------

def estimate_occupancy(regs_per_thread: int, block_dim: int, sm_props: dict[str, Any]) -> dict[str, Any]:
    """Estimate theoretical SM occupancy for given register usage and block size."""
    warp_size = sm_props.get("warp_size", 32)
    warps_per_block = block_dim // warp_size
    if warps_per_block <= 0:
        return {"block_dim": block_dim, "occupancy": 0.0, "active_warps_per_sm": 0, "active_blocks_per_sm": 0}

    reg_alloc_unit = sm_props.get("reg_alloc_unit", 256)
    regs_per_sm = sm_props.get("regs_per_sm", 65536)
    max_warps_per_sm = sm_props.get("max_warps_per_sm", 48)
    max_blocks_per_sm = sm_props.get("max_blocks_per_sm", 16)

    regs_per_warp = ((regs_per_thread * warp_size + reg_alloc_unit - 1) // reg_alloc_unit) * reg_alloc_unit
    max_warps_by_regs = regs_per_sm // regs_per_warp if regs_per_warp > 0 else max_warps_per_sm
    max_blocks_by_regs = max_warps_by_regs // warps_per_block
    max_blocks_by_warps = max_warps_per_sm // warps_per_block
    active_blocks = min(max_blocks_by_regs, max_blocks_by_warps, max_blocks_per_sm)
    active_warps = active_blocks * warps_per_block
    occupancy = active_warps / max_warps_per_sm if max_warps_per_sm > 0 else 0.0
    return {
        "block_dim": block_dim,
        "warps_per_block": warps_per_block,
        "regs_per_warp_alloc": regs_per_warp,
        "active_blocks_per_sm": active_blocks,
        "active_warps_per_sm": active_warps,
        "occupancy": occupancy,
    }


def recommend_block_dim(sm_props: dict[str, Any], kernel_class: str = "default") -> int:
    """Choose block_dim using a per-kernel-family heuristic.

    Family-specific tie-breaking when multiple candidates yield the
    same warps-per-SM occupancy:

    * **compute_bound** / **latency_bound**: prefer *smaller* blocks
      (more blocks per SM → better independent scheduling / stall hiding).
    * **memory_bound**: prefer *larger* blocks (better spatial locality
      for streaming access patterns in L2 cache).
    * **atomic_bound**: among candidates reaching ≥50 % occupancy,
      prefer the *smallest* block (fewer threads per block → less
      intra-block contention on shared atomic locations).
    * **warp_specialized**: return the architecturally fixed block_dim
      (e.g. 256 for cooperative tile-loading, 32 for warp-level reduction).
    """
    # Warp-specialized kernels have a fixed block_dim set at registration.
    fixed = _KERNEL_FIXED_BLOCK_DIM.get(kernel_class)
    if fixed is not None:
        return fixed

    family = _KERNEL_FAMILY.get(kernel_class, FAMILY_COMPUTE)
    regs = _KERNEL_REG_ESTIMATES.get(kernel_class, _KERNEL_REG_ESTIMATES["default"])
    max_warps_per_sm = sm_props.get("max_warps_per_sm", 48)

    if family == FAMILY_ATOMIC:
        return _recommend_atomic(regs, sm_props, max_warps_per_sm)

    # Compute occupancy for every candidate.
    candidates: list[tuple[int, int]] = []  # (warps, block_dim)
    for bd in (64, 128, 192, 256):
        occ = estimate_occupancy(regs, bd, sm_props)
        candidates.append((occ["active_warps_per_sm"], bd))

    # memory_bound: maximise occupancy, tie-break *larger* block.
    if family == FAMILY_MEMORY:
        candidates.sort(key=lambda t: (t[0], t[1]))
        return candidates[-1][1]

    # compute_bound / latency_bound: maximise occupancy, tie-break *smaller* block.
    candidates.sort(key=lambda t: (t[0], -t[1]))
    return candidates[-1][1]


def _recommend_atomic(regs: int, sm_props: dict[str, Any], max_warps_per_sm: int) -> int:
    """Heuristic for atomic-bound kernels.

    Among candidates achieving at least 50 % peak occupancy, pick the
    smallest block_dim to minimise intra-block atomic contention.
    Falls back to the smallest block that achieves the *maximum*
    occupancy if no candidate reaches the threshold.
    """
    threshold_warps = max_warps_per_sm // 2  # 50 % occupancy
    best_dim = 256
    best_warps = 0
    above: list[tuple[int, int]] = []  # (block_dim, warps)
    for bd in (64, 128, 192, 256):
        occ = estimate_occupancy(regs, bd, sm_props)
        warps = occ["active_warps_per_sm"]
        if warps > best_warps or (warps == best_warps and bd < best_dim):
            best_warps = warps
            best_dim = bd
        if warps >= threshold_warps:
            above.append((bd, warps))
    if above:
        # Smallest block that still meets the 50 % threshold.
        return min(above, key=lambda t: t[0])[0]
    return best_dim


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_kernel_class(
    name: str,
    estimated_regs: int,
    family: str = FAMILY_COMPUTE,
    fixed_block_dim: int | None = None,
) -> None:
    """Register a kernel class with its estimated register usage and family.

    Parameters
    ----------
    name:
        Unique kernel class identifier (e.g. ``"preprocess"``).
    estimated_regs:
        Empirical register count per thread (from NCU or estimation).
    family:
        Performance family — one of ``FAMILY_COMPUTE``, ``FAMILY_MEMORY``,
        ``FAMILY_ATOMIC``, ``FAMILY_LATENCY``, or ``FAMILY_WARP_SPECIALIZED``.
        Controls the tie-breaking heuristic in ``recommend_block_dim()``.
    fixed_block_dim:
        Required when *family* is ``FAMILY_WARP_SPECIALIZED``.  The
        architecturally mandated block size (e.g. 256 for cooperative
        tile-loading, 32 for warp-level reduction).  Ignored for other
        families.

    Call this before ``initialize_tuning()`` to include the kernel class
    in the block_dim plan.  Can also be called after; the next
    ``initialize_tuning()`` or ``get_tuned_block_dim()`` call will pick it up.

    Registering the same name twice updates the estimate and invalidates any
    cached plan that used it (so the next call to ``initialize_tuning`` or
    ``get_tuned_block_dim`` recomputes correctly).
    """
    old = _KERNEL_REG_ESTIMATES.get(name)
    _KERNEL_REG_ESTIMATES[name] = estimated_regs
    _KERNEL_FAMILY[name] = family
    if fixed_block_dim is not None:
        _KERNEL_FIXED_BLOCK_DIM[name] = fixed_block_dim
    else:
        _KERNEL_FIXED_BLOCK_DIM.pop(name, None)
    if old != estimated_regs:
        # Invalidate per-device cached plans so they get recomputed.
        for dev_plan in _TUNED_BLOCK_DIM_BY_DEVICE.values():
            dev_plan.pop(name, None)


def get_tuned_block_dim(kernel_class: str,
                         device: torch.device | str | None = None) -> int:
    """Return the tuned block_dim for *kernel_class* on *device*.

    Parameters
    ----------
    kernel_class:
        Name used when registering the kernel via ``register_kernel_class()``.
    device:
        Target CUDA device.  When ``None``, uses the current PyTorch CUDA
        device (``torch.cuda.current_device()``).  Pass the actual training
        device explicitly in multi-GPU settings.

    Returns
    -------
    int
        Optimal block_dim, or 256 if tuning information is unavailable.
    """
    dev = normalize_device(device)
    dev_key = str(dev)

    dev_plan = _TUNED_BLOCK_DIM_BY_DEVICE.get(dev_key)
    if dev_plan is None:
        # First call for this device — run tuning lazily (silent).
        initialize_tuning(dev, verbose=False)
        dev_plan = _TUNED_BLOCK_DIM_BY_DEVICE.get(dev_key, {})

    bd = dev_plan.get(kernel_class)
    if bd is not None:
        return bd

    # Late-registered class — compute on-the-fly and cache.
    device_props, _, _, _ = query_device_info(dev)
    sm_props = query_sm_properties(device_props)
    if sm_props:
        bd = recommend_block_dim(sm_props, kernel_class)
        dev_plan[kernel_class] = bd
        return bd
    return 256


def get_canonical_device() -> torch.device | None:
    """Return the canonical training device set by the first ``initialize_tuning`` call.

    Returns ``None`` if ``initialize_tuning`` has not been called yet.
    """
    return _CANONICAL_DEVICE


def initialize_tuning(device: torch.device | str | None = None,
                       verbose: bool = True) -> dict[str, Any]:
    """Query *device* and build block_dim recommendations for all registered kernel classes.

    Safe to call multiple times; results are cached per device.  Calling this
    once from the rasterizer makes the plan immediately available to SSIM and
    KNN kernels without any extra device queries.

    The first call with an explicit *device* establishes the **canonical
    device**: all subsequent ``normalize_device(None)`` lookups (e.g. from
    inference helpers) will resolve to this same GPU, matching the external
    training framework's device choice.

    Parameters
    ----------
    device:
        GPU to query.  Defaults to the current PyTorch CUDA device.
    verbose:
        Print a one-time summary to stdout.  Subsequent calls for the same
        device are always silent regardless of this flag.
    """
    global _CANONICAL_DEVICE
    dev = normalize_device(device)
    # Lock in the canonical device on the first explicit call so every
    # gswarp module defaults to the same GPU as the training framework.
    if _CANONICAL_DEVICE is None and device is not None:
        _CANONICAL_DEVICE = dev
    dev_key = str(dev)

    cached = _TUNING_CACHE.get(dev_key)
    if cached is not None:
        # Re-check for late-registered kernel classes not in the cached plan.
        dev_plan = _TUNED_BLOCK_DIM_BY_DEVICE.setdefault(dev_key, {})
        missing = [kc for kc in _KERNEL_REG_ESTIMATES if kc not in dev_plan]
        if missing:
            device_props, _, _, _ = query_device_info(dev)
            sm_props = query_sm_properties(device_props)
            if sm_props:
                for kc in missing:
                    dev_plan[kc] = recommend_block_dim(sm_props, kc)
                    cached["block_dim_plan"][kc] = dev_plan[kc]
                    cached["occupancy_snapshot"][kc] = estimate_occupancy(
                        _KERNEL_REG_ESTIMATES[kc], dev_plan[kc], sm_props)
        return cached

    device_props, free_memory, total_memory, dev = query_device_info(dev)
    sm_props = query_sm_properties(device_props)

    plan: dict[str, int] = {}
    occ_snap: dict[str, dict[str, Any]] = {}
    if sm_props:
        for kc in _KERNEL_REG_ESTIMATES:
            plan[kc] = recommend_block_dim(sm_props, kc)
            occ_snap[kc] = estimate_occupancy(_KERNEL_REG_ESTIMATES[kc], plan[kc], sm_props)

    # Store in the per-device plan (merge, don't overwrite, in case other
    # modules pre-populated the dict via late calls to get_tuned_block_dim).
    dev_plan = _TUNED_BLOCK_DIM_BY_DEVICE.setdefault(dev_key, {})
    dev_plan.update(plan)

    report = {
        "device": dev_key,
        "device_name": getattr(device_props, "name", "unknown"),
        "compute_capability": f"{device_props.major}.{device_props.minor}" if device_props else None,
        "sm_count": int(device_props.multi_processor_count) if device_props else None,
        "sm_properties": sm_props,
        "block_dim_plan": plan,
        "occupancy_snapshot": occ_snap,
        "free_memory_bytes": free_memory,
        "total_memory_bytes": total_memory,
    }
    _TUNING_CACHE[dev_key] = report

    if verbose and dev_key not in _TUNING_LOGGED_DEVICES:
        _print_tuning_report(report)
        _TUNING_LOGGED_DEVICES.add(dev_key)

    return report


def _print_tuning_report(report: dict[str, Any]) -> None:
    """Compact tuning summary to stdout."""
    _H = "\033[96m"
    _R = "\033[0m"
    _G = "\033[92m"
    _Y = "\033[93m"

    print(f"{_H}[gswarp-tuning] Auto-tuning initialized{_R}")
    print(f"{_G}  device: {report['device']} ({report['device_name']}){_R}")

    sm = report.get("sm_properties", {})
    if sm:
        print(f"{_Y}  sm_count={sm.get('sm_count','?')} "
              f"max_warps/sm={sm.get('max_warps_per_sm','?')} "
              f"max_blocks/sm={sm.get('max_blocks_per_sm','?')} "
              f"regs/sm={sm.get('regs_per_sm','?')}{_R}")

    plan = report.get("block_dim_plan", {})
    if plan:
        parts = [f"{k}={v}[{_KERNEL_FAMILY.get(k, '?')}]" for k, v in sorted(plan.items())]
        print(f"{_Y}  block_dim: {', '.join(parts)}{_R}")

    occ = report.get("occupancy_snapshot", {})
    if occ:
        occ_parts = [f"{k}={o['occupancy']:.0%}({o['active_warps_per_sm']}w)"
                     for k, o in sorted(occ.items())]
        print(f"{_Y}  occupancy: {', '.join(occ_parts)}{_R}")
