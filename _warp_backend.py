import os
from typing import Any

import torch
import warp as wp

__all__ = [
    "clear_warp_caches",
    "get_default_parameter_info",
    "is_available",
    "get_backward_mode",
    "get_binning_sort_mode",
    "get_runtime_auto_tuning_config",
    "get_runtime_tuning_report",
    "initialize_runtime_tuning",
    "preprocess_gaussians",
    "rasterize_gaussians",
    "mark_visible",
    "rasterize_gaussians_backward",
    "set_runtime_auto_tuning",
    "get_compute_depth",
    "set_compute_depth",
]


NUM_CHANNELS = 3
BLOCK_X = 16
BLOCK_Y = 16
sh_c0 = wp.constant(wp.float32(0.28209479177387814))
sh_c1 = wp.constant(wp.float32(0.4886025119029199))
sh_c2_0 = wp.constant(wp.float32(1.0925484305920792))
sh_c2_1 = wp.constant(wp.float32(-1.0925484305920792))
sh_c2_2 = wp.constant(wp.float32(0.31539156525252005))
sh_c2_3 = wp.constant(wp.float32(-1.0925484305920792))
sh_c2_4 = wp.constant(wp.float32(0.5462742152960396))

sh_c3_0 = wp.constant(wp.float32(-0.5900435899266435))
sh_c3_1 = wp.constant(wp.float32(2.890611442640554))
sh_c3_2 = wp.constant(wp.float32(-0.4570457994644658))
sh_c3_3 = wp.constant(wp.float32(0.3731763325901154))
sh_c3_4 = wp.constant(wp.float32(-0.4570457994644658))
sh_c3_5 = wp.constant(wp.float32(1.445305721320277))
sh_c3_6 = wp.constant(wp.float32(-0.5900435899266435))

DEFAULT_BACKWARD_MODE = "manual"
_BACKWARD_MODE = DEFAULT_BACKWARD_MODE
BINNING_SORT_MODES = ("warp_radix", "torch", "warp_depth_stable_tile")
DEFAULT_BINNING_SORT_MODE = "warp_depth_stable_tile"
_BINNING_SORT_MODE = DEFAULT_BINNING_SORT_MODE
WARP_RADIX_DETERMINISTIC_TIEBREAK = False
TORCH_SINGLE_SORT_THRESHOLD = 1000000
FORWARD_GEOM_FLOAT_WIDTH = 16
# L2: clamped stored as int32 in geom_buffer (eliminates backward bool→int32 conversion)
FORWARD_GEOM_CLAMP_WIDTH = NUM_CHANNELS * 4
BINNING_AUTO_TUNE_GROWTH_CAP = 4.0
BINNING_AUTO_TUNE_SWITCH_RATIO = 2.5
BINNING_AUTO_TUNE_KEEP_RATIO = 2.0
BINNING_AUTO_TUNE_MIN_SWITCH_POINTS = 16384
BINNING_AUTO_TUNE_MIN_KEEP_POINTS = 12288
VISIBILITY_NEAR_PLANE = 0.2
PREPROCESS_CULL_SIGMA = 3.0
PREPROCESS_CULL_FOV_SCALE = 1.3

# D1: Skip per-pixel depth accumulation (forward) and depth gradient (backward)
# when depth output is not used in the loss.  Controlled globally or via
# GSWARP_COMPUTE_DEPTH=0 environment variable.
_COMPUTE_DEPTH = os.environ.get("GSWARP_COMPUTE_DEPTH", "1") != "0"
DET_EPSILON = 1.0e-7
ONE_MINUS_ALPHA_MIN = 1.0e-5
_RADIX_SORT_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor, Any | None, torch.Tensor]] = {}
_RADIX_SORT_I32_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor, Any | None, torch.Tensor]] = {}
_INDEX_GATHER_I32_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor]] = {}
_INDEX_GATHER_I64_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor]] = {}
_SCAN_I32_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor, int]] = {}
_PROJECT_VISIBLE_BUFFER_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
_SEQUENCE_BUFFER_CACHE: dict[str, torch.Tensor] = {}
_DEPTH_SORT_ORDER_CACHE: dict[str, tuple[torch.Tensor, int]] = {}
_DEPTH_SORT_CHECK_FLAG: dict[str, torch.Tensor] = {}
_RUNTIME_TUNING_CACHE: dict[str, dict[str, Any]] = {}
_RUNTIME_TUNING_LOGGED_DEVICES: set[str] = set()
_RUNTIME_BINNING_POLICY_STATE: dict[str, dict[str, Any]] = {}
_AUTO_TUNE_ENABLED = True
_AUTO_TUNE_VERBOSE = True
FORWARD_GEOM_STRIDE_BYTES = FORWARD_GEOM_FLOAT_WIDTH * 4 + FORWARD_GEOM_CLAMP_WIDTH
# B5: track point count for auto-clear after pruning
_PREV_POINT_COUNT: int | None = None

# C4: per-kernel Launch object caches — keyed by (device_str, dim)
_C4_LAUNCH_CACHE_SH: dict[tuple[str, int], Any] = {}
_C4_LAUNCH_CACHE_COV3D: dict[tuple[str, int], Any] = {}
_C4_LAUNCH_CACHE_RENDER_BWD: dict[tuple[str, int], Any] = {}
_C4_LAUNCH_CACHE_PROJ_MEANS: dict[tuple[str, int], Any] = {}
_C4_LAUNCH_CACHE_COV2D: dict[tuple[str, int], Any] = {}
_C4_LAUNCH_CACHE_ACCUM: dict[tuple[str, int, int], Any] = {}
# C12: split SH backward caches
_C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS: dict[tuple[str, int, int], Any] = {}
# F1: merged vec3 SH backward cache (deg 0-3 single kernel)
_C4_LAUNCH_CACHE_SH_V3: dict[tuple[str, int], Any] = {}
# P1: forward hot-path launch caches
_C4_LAUNCH_CACHE_FWD_SH: dict[tuple[str, int], Any] = {}
_C4_LAUNCH_CACHE_FWD_PREPROCESS: dict[tuple[str, int], Any] = {}
_C4_LAUNCH_CACHE_FWD_RENDER: dict[tuple[str, int], Any] = {}
# G1: binning duplicate launch cache (C4-cached)
_C4_LAUNCH_CACHE_BINNING_DUPLICATE: dict[tuple[str, int], Any] = {}
# T1: tiled256 forward render launch cache
_C4_LAUNCH_CACHE_FWD_RENDER_TILED256: dict[tuple[str, int], Any] = {}


# T1: Cooperative tile batch size for render kernels.
# Each screen-tile block loads RENDER_TILE_BATCH Gaussians into shared memory
# per iteration. Must be a compile-time constant (wp.constant).
RENDER_TILE_BATCH = 32

# One-shot initialization flag — after first _require_warp() succeeds, all
# subsequent calls short-circuit immediately.
_WARP_INITIALIZED = False

# Tuned kernel block dimensions — set once during initialize_runtime_tuning(),
# used by all subsequent wp.launch() calls for the corresponding kernel class.
_TUNED_BLOCK_DIM: dict[str, int] = {}  # kernel_class -> block_dim

# SM architecture property table — keyed by (major, minor) compute capability.
# Used for occupancy estimation when properties are not available via PyTorch.
# Sources: NVIDIA CUDA Programming Guide Table 15 (Thread Block, Register, SM)
#          Architecture whitepapers for L2 cache / shared memory.
_SM_ARCHITECTURE_PROPS: dict[tuple[int, int], dict[str, int]] = {
    # Volta
    (7, 0): {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32, "shared_mem_per_sm_bytes": 98304, "l2_cache_bytes": 6291456, "reg_alloc_unit": 256},
    # Turing
    (7, 5): {"regs_per_sm": 65536, "max_warps_per_sm": 32, "max_blocks_per_sm": 16, "shared_mem_per_sm_bytes": 65536, "l2_cache_bytes": 4194304, "reg_alloc_unit": 256},
    # Ampere GA100
    (8, 0): {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32, "shared_mem_per_sm_bytes": 167936, "l2_cache_bytes": 41943040, "reg_alloc_unit": 256},
    # Ampere GA10x
    (8, 6): {"regs_per_sm": 65536, "max_warps_per_sm": 48, "max_blocks_per_sm": 16, "shared_mem_per_sm_bytes": 102400, "l2_cache_bytes": 4194304, "reg_alloc_unit": 256},
    # Ada Lovelace
    (8, 9): {"regs_per_sm": 65536, "max_warps_per_sm": 48, "max_blocks_per_sm": 24, "shared_mem_per_sm_bytes": 102400, "l2_cache_bytes": 33554432, "reg_alloc_unit": 256},
    # Hopper
    (9, 0): {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32, "shared_mem_per_sm_bytes": 233472, "l2_cache_bytes": 52428800, "reg_alloc_unit": 256},
    # Blackwell
    (10, 0): {"regs_per_sm": 65536, "max_warps_per_sm": 64, "max_blocks_per_sm": 32, "shared_mem_per_sm_bytes": 233472, "l2_cache_bytes": 67108864, "reg_alloc_unit": 256},
}

_ANSI_RESET = "\033[0m"
_ANSI_HEADER = "\033[96m"
_ANSI_DEVICE = "\033[92m"
_ANSI_MEMORY = "\033[93m"
_ANSI_TILE = "\033[95m"
_ANSI_STRATEGY = "\033[94m"
_ANSI_HW = "\033[33m"

# -----------------------------------------------------------------------------
# Runtime tuning and diagnostics
# -----------------------------------------------------------------------------


def _runtime_color(text: str, color: str) -> str:
    return f"{color}{text}{_ANSI_RESET}"


def get_default_parameter_info() -> dict[str, dict[str, Any]]:
    return {
        "BLOCK_X": {
            "value": BLOCK_X,
            "category": "kernel_tile",
            "auto_tunable": False,
            "description": "Compile-time tile width used by the current kernels. Reported for reference only.",
        },
        "BLOCK_Y": {
            "value": BLOCK_Y,
            "category": "kernel_tile",
            "auto_tunable": False,
            "description": "Compile-time tile height used by the current kernels. Reported for reference only.",
        },
        "DEFAULT_BACKWARD_MODE": {
            "value": DEFAULT_BACKWARD_MODE,
            "category": "experimental",
            "auto_tunable": False,
            "description": "Backward implementation mode. The current backend only supports 'manual'.",
        },
        "DEFAULT_BINNING_SORT_MODE": {
            "value": DEFAULT_BINNING_SORT_MODE,
            "category": "experimental",
            "auto_tunable": True,
            "description": "Default Warp binning strategy. Unless the user explicitly overrides it, the backend uses 'torch' as the mainline path (benchmarked as best overall).",
        },
        "WARP_RADIX_DETERMINISTIC_TIEBREAK": {
            "value": WARP_RADIX_DETERMINISTIC_TIEBREAK,
            "category": "experimental",
            "auto_tunable": False,
            "description": "Deterministic tie-break for radix sort. Disabled by default because it adds overhead.",
        },
        "TORCH_SINGLE_SORT_THRESHOLD": {
            "value": TORCH_SINGLE_SORT_THRESHOLD,
            "category": "experimental",
            "auto_tunable": False,
            "description": "Threshold used only by the torch sort fallback path.",
        },
    }


def get_runtime_auto_tuning_config() -> dict[str, bool]:
    return {
        "enabled": _AUTO_TUNE_ENABLED,
        "verbose": _AUTO_TUNE_VERBOSE,
    }


def set_runtime_auto_tuning(enabled: bool | None = None, verbose: bool | None = None) -> None:
    global _AUTO_TUNE_ENABLED, _AUTO_TUNE_VERBOSE
    if enabled is not None:
        _AUTO_TUNE_ENABLED = bool(enabled)
    if verbose is not None:
        _AUTO_TUNE_VERBOSE = bool(verbose)





def get_compute_depth() -> bool:
    """Return whether per-pixel depth is computed in the render kernels."""
    return _COMPUTE_DEPTH


def set_compute_depth(enabled: bool) -> None:
    """Enable or disable per-pixel depth computation.

    When disabled, forward render kernels skip depth accumulation and backward
    kernels skip depth gradient computation.  This saves ~5-10% of total
    iteration time for scenes that do not use depth in the loss (the common
    case for truck, lego, etc.).  Can also be set via GSWARP_COMPUTE_DEPTH=0.
    """
    global _COMPUTE_DEPTH
    _COMPUTE_DEPTH = bool(enabled)


def _normalize_runtime_device(device: torch.device | str | None = None) -> torch.device:
    if device is None:
        if torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")
    if isinstance(device, str):
        return torch.device(device)
    return device


def _query_runtime_device_info(runtime_device: torch.device) -> tuple[Any | None, int | None, int | None, torch.device]:
    device_props = None
    free_memory = None
    total_memory = None
    normalized_device = runtime_device
    if runtime_device.type == "cuda" and torch.cuda.is_available():
        device_index = runtime_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
            normalized_device = torch.device("cuda", device_index)
        device_props = torch.cuda.get_device_properties(device_index)
        free_memory, total_memory = torch.cuda.mem_get_info(device_index)
    return device_props, free_memory, total_memory, normalized_device

def _get_sm_architecture_props(major: int, minor: int) -> dict[str, int] | None:
    """Look up SM architecture constants by compute capability.

    Falls back to the nearest lower architecture if the exact (major, minor) is
    not in the table."""
    props = _SM_ARCHITECTURE_PROPS.get((major, minor))
    if props is not None:
        return dict(props)
    for (m, n), p in sorted(_SM_ARCHITECTURE_PROPS.items(), reverse=True):
        if m <= major:
            return dict(p)
    return None


def _query_sm_properties(device_props: Any | None) -> dict[str, Any]:
    """Build a comprehensive SM properties dict from *device_props* + the
    architecture lookup table."""
    if device_props is None:
        return {}
    major, minor = int(device_props.major), int(device_props.minor)
    arch = _get_sm_architecture_props(major, minor)
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
        sm_props["regs_per_sm"] = arch["regs_per_sm"]
        sm_props["max_warps_per_sm"] = arch["max_warps_per_sm"]
        sm_props["max_blocks_per_sm"] = arch["max_blocks_per_sm"]
        sm_props["shared_mem_per_sm_bytes"] = arch["shared_mem_per_sm_bytes"]
        sm_props["l2_cache_bytes"] = arch["l2_cache_bytes"]
        sm_props["reg_alloc_unit"] = arch["reg_alloc_unit"]
    else:
        sm_props["regs_per_sm"] = 65536
        sm_props["max_warps_per_sm"] = max_threads_per_sm // warp_size if max_threads_per_sm > 0 else 48
        sm_props["max_blocks_per_sm"] = 16
        sm_props["shared_mem_per_sm_bytes"] = 0
        sm_props["l2_cache_bytes"] = 0
        sm_props["reg_alloc_unit"] = 256
    return sm_props


def _estimate_occupancy(regs_per_thread: int, block_dim: int, sm_props: dict[str, Any]) -> dict[str, Any]:
    """Estimate theoretical SM occupancy for a kernel with *regs_per_thread*
    registers launched at *block_dim* threads per block."""
    warp_size = sm_props.get("warp_size", 32)
    warps_per_block = block_dim // warp_size
    if warps_per_block <= 0:
        return {"block_dim": block_dim, "occupancy": 0.0, "active_warps_per_sm": 0, "active_blocks_per_sm": 0}
    reg_alloc_unit = sm_props.get("reg_alloc_unit", 256)
    regs_per_sm = sm_props.get("regs_per_sm", 65536)
    max_warps_per_sm = sm_props.get("max_warps_per_sm", 48)
    max_blocks_per_sm = sm_props.get("max_blocks_per_sm", 16)
    # Register allocation: rounded up to allocation-unit boundary per warp.
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


# Expected register usage per kernel class (empirical, measured via NCU on sm_89
# for the fused kernels).  Used as representative values when actual register
# counts are unavailable.
_KERNEL_REG_ESTIMATES: dict[str, int] = {
    "render": 96,
    "preprocess": 110,
    "backward_preprocess": 127,
    "backward_render": 96,
    "light": 48,
    "default": 96,
}


def _recommend_block_dim(sm_props: dict[str, Any], kernel_class: str = "default") -> int:
    """Choose the block_dim that maximises warps-per-SM for *kernel_class*.

    Among candidates with equal occupancy, prefers the largest block_dim to
    benefit from intra-block scheduling."""
    regs = _KERNEL_REG_ESTIMATES.get(kernel_class, _KERNEL_REG_ESTIMATES["default"])
    best_dim = 256
    best_warps = 0
    for candidate in (64, 128, 192, 256):
        occ = _estimate_occupancy(regs, candidate, sm_props)
        warps = occ["active_warps_per_sm"]
        if warps > best_warps or (warps == best_warps and candidate > best_dim):
            best_warps = warps
            best_dim = candidate
    return best_dim


def _build_block_dim_recommendations(sm_props: dict[str, Any]) -> dict[str, int]:
    """Return per-kernel-class block_dim recommendations."""
    recommendations: dict[str, int] = {}
    for kc in _KERNEL_REG_ESTIMATES:
        recommendations[kc] = _recommend_block_dim(sm_props, kc)
    return recommendations


def _get_block_dim(kernel_class: str) -> int:
    """Return the tuned block_dim for *kernel_class*, or 256 as default."""
    return _TUNED_BLOCK_DIM.get(kernel_class, _TUNED_BLOCK_DIM.get("default", 256))


def _recommend_tile_shape(device_props: Any | None, free_memory_bytes: int | None) -> tuple[int, int]:
    if device_props is None:
        return BLOCK_X, BLOCK_Y
    free_memory_gib = 0.0 if free_memory_bytes is None else float(free_memory_bytes) / float(1024**3)
    sm_count = int(getattr(device_props, "multi_processor_count", 0))
    if free_memory_gib <= 2.0:
        return 8, 16
    if free_memory_gib >= 8.0 and sm_count >= 48:
        return 16, 16
    return 16, 16


def _get_runtime_binning_policy_state(device: torch.device) -> dict[str, Any]:
    device_key = str(_normalize_runtime_device(device))
    state = _RUNTIME_BINNING_POLICY_STATE.get(device_key)
    if state is None:
        state = {
            "initial_point_count": None,
            "max_point_count_seen": 0,
            "last_point_count": 0,
            "last_selected_mode": DEFAULT_BINNING_SORT_MODE,
            "last_reason": "default_warp_mainline",
        }
        _RUNTIME_BINNING_POLICY_STATE[device_key] = state
    return state


def _build_binning_policy_snapshot(device: torch.device) -> dict[str, Any]:
    state = _get_runtime_binning_policy_state(device)
    last_point_count = int(state["last_point_count"])
    return {
        "policy": "default_mainline",
        "default_mode": DEFAULT_BINNING_SORT_MODE,
        "experimental_mode": None,
        "selected_mode": state["last_selected_mode"],
        "reason": state["last_reason"],
        "current_point_count": last_point_count,
        "growth_ratio": None,
        "switch_ratio": None,
        "keep_ratio": None,
        "max_experimental_point_count": None,
        "thresholds": {
            "initial_point_count": state["initial_point_count"],
            "expected_peak_point_count": None,
            "switch_point_count": None,
            "keep_point_count": None,
            "experimental_max_point_count": None,
        },
        "max_point_count_seen": int(state["max_point_count_seen"]),
        "ran": False,
    }


def _recommend_binning_sort_mode(device: torch.device, device_props: Any | None, free_memory_bytes: int | None) -> tuple[str, dict[str, Any]]:
    del free_memory_bytes
    policy = _build_binning_policy_snapshot(device)
    if device.type != "cuda" or device_props is None:
        policy["selected_mode"] = "torch"
        policy["reason"] = "cuda_or_warp_unavailable"
        return "torch", policy
    policy["selected_mode"] = DEFAULT_BINNING_SORT_MODE
    policy["reason"] = "default_warp_mainline"
    return policy["selected_mode"], policy


def _select_auto_binning_sort_mode(device: torch.device, point_count: int) -> tuple[str, dict[str, Any]]:
    runtime_device = _normalize_runtime_device(device)
    if runtime_device.type != "cuda":
        return DEFAULT_BINNING_SORT_MODE, {
            "policy": "default_mainline",
            "selected_mode": DEFAULT_BINNING_SORT_MODE,
            "reason": "cuda_or_warp_unavailable",
            "current_point_count": int(point_count),
        }

    state = _get_runtime_binning_policy_state(runtime_device)
    if point_count > 0 and state["initial_point_count"] is None:
        state["initial_point_count"] = int(point_count)
    selected_mode = DEFAULT_BINNING_SORT_MODE
    reason = "default_warp_mainline"
    if point_count <= 0:
        reason = "empty_workload"

    state["last_point_count"] = int(point_count)
    state["max_point_count_seen"] = max(int(state["max_point_count_seen"]), int(point_count))
    state["last_selected_mode"] = selected_mode
    state["last_reason"] = reason
    return selected_mode, _build_binning_policy_snapshot(runtime_device)


def _build_runtime_tuning_report(device: torch.device | str | None = None) -> dict[str, Any]:
    runtime_device = _normalize_runtime_device(device)
    device_props, free_memory, total_memory, runtime_device = _query_runtime_device_info(runtime_device)
    device_key = str(runtime_device)

    recommended_tile_x, recommended_tile_y = _recommend_tile_shape(device_props, free_memory)
    recommended_binning_sort_mode, binning_policy = _recommend_binning_sort_mode(runtime_device, device_props, free_memory)

    # Build comprehensive SM properties and occupancy-based block_dim plan.
    sm_props = _query_sm_properties(device_props)
    block_dim_plan = _build_block_dim_recommendations(sm_props) if sm_props else {}
    # Populate the global _TUNED_BLOCK_DIM once.
    if block_dim_plan and not _TUNED_BLOCK_DIM:
        _TUNED_BLOCK_DIM.update(block_dim_plan)

    # Build representative occupancy snapshot per kernel class.
    occupancy_snapshot: dict[str, dict[str, Any]] = {}
    if sm_props:
        for kc, est_regs in _KERNEL_REG_ESTIMATES.items():
            chosen_bd = block_dim_plan.get(kc, 256)
            occupancy_snapshot[kc] = _estimate_occupancy(est_regs, chosen_bd, sm_props)

    report = {
        "device": device_key,
        "device_type": runtime_device.type,
        "device_name": getattr(device_props, "name", runtime_device.type.upper()),
        "compute_capability": None if device_props is None else f"{device_props.major}.{device_props.minor}",
        "sm_count": None if device_props is None else int(device_props.multi_processor_count),
        "warp_size": None if device_props is None else int(device_props.warp_size),
        "free_memory_bytes": None if free_memory is None else int(free_memory),
        "total_memory_bytes": None if total_memory is None else int(total_memory),
        "sm_properties": sm_props,
        "current_tile": (BLOCK_X, BLOCK_Y),
        "recommended_tile": (recommended_tile_x, recommended_tile_y),
        "tile_runtime_mutable": False,
        "use_tile_kernels": True,
        "compute_depth": _COMPUTE_DEPTH,
        "applied_binning_sort_mode": _BINNING_SORT_MODE,
        "recommended_binning_sort_mode": recommended_binning_sort_mode,
        "block_dim_plan": block_dim_plan,
        "occupancy_snapshot": occupancy_snapshot,
        "benchmark": {
            "ran": False,
            "selected_mode": recommended_binning_sort_mode,
            "reason": "default_mainline",
        },
        "binning_policy": binning_policy,
        "applied_auto_tune": True,
    }
    return report


def _refresh_runtime_tuning_report_memory(report: dict[str, Any]) -> dict[str, Any]:
    runtime_device = _normalize_runtime_device(report["device"])
    device_props, free_memory, total_memory, runtime_device = _query_runtime_device_info(runtime_device)
    updated = dict(report)
    updated["device"] = str(runtime_device)
    updated["device_name"] = getattr(device_props, "name", updated["device_name"])
    updated["compute_capability"] = None if device_props is None else f"{device_props.major}.{device_props.minor}"
    updated["sm_count"] = None if device_props is None else int(device_props.multi_processor_count)
    updated["warp_size"] = None if device_props is None else int(device_props.warp_size)
    updated["free_memory_bytes"] = None if free_memory is None else int(free_memory)
    updated["total_memory_bytes"] = None if total_memory is None else int(total_memory)
    updated["recommended_tile"] = _recommend_tile_shape(device_props, free_memory)
    updated["applied_binning_sort_mode"] = _BINNING_SORT_MODE
    updated["binning_policy"] = _build_binning_policy_snapshot(runtime_device)
    updated["recommended_binning_sort_mode"] = updated["binning_policy"]["selected_mode"]
    updated["benchmark"] = {
        "ran": False,
        "selected_mode": updated["recommended_binning_sort_mode"],
        "reason": "default_mainline",
    }
    return updated


def _print_runtime_tuning_report(report: dict[str, Any]) -> None:
    print(_runtime_color("[warp-backend] Runtime auto-tuning initialized", _ANSI_HEADER))
    print(_runtime_color(f"device: {report['device']} ({report['device_name']})", _ANSI_DEVICE))
    if report["free_memory_bytes"] is not None and report["total_memory_bytes"] is not None:
        free_gib = report["free_memory_bytes"] / float(1024**3)
        total_gib = report["total_memory_bytes"] / float(1024**3)
        print(_runtime_color(f"memory: {free_gib:.2f} / {total_gib:.2f} GiB free", _ANSI_MEMORY))
    else:
        print(_runtime_color("memory: unavailable", _ANSI_MEMORY))
    # SM hardware details
    sm = report.get("sm_properties", {})
    if sm:
        l2_kb = sm.get("l2_cache_bytes", 0) // 1024
        smem_kb = sm.get("shared_mem_per_sm_bytes", 0) // 1024
        print(_runtime_color(
            "hw: sm_count={}, max_warps/sm={}, max_blocks/sm={}, regs/sm={}, shared/sm={}KB, L2={}KB".format(
                sm.get("sm_count", "?"),
                sm.get("max_warps_per_sm", "?"),
                sm.get("max_blocks_per_sm", "?"),
                sm.get("regs_per_sm", "?"),
                smem_kb,
                l2_kb,
            ),
            _ANSI_HW,
        ))
    # Block-dim plan and occupancy
    bdp = report.get("block_dim_plan", {})
    if bdp:
        parts = [f"{k}={v}" for k, v in sorted(bdp.items())]
        print(_runtime_color("block_dim: " + ", ".join(parts), _ANSI_HW))
    occ = report.get("occupancy_snapshot", {})
    if occ:
        occ_parts = []
        for k in sorted(occ):
            o = occ[k]
            occ_parts.append(f"{k}={o['occupancy']:.0%}({o['active_warps_per_sm']}w)")
        print(_runtime_color("occupancy: " + ", ".join(occ_parts), _ANSI_HW))
    print(
        _runtime_color(
            "tile: current {}x{}, recommended {}x{} (compile-time constant)".format(
                report["current_tile"][0],
                report["current_tile"][1],
                report["recommended_tile"][0],
                report["recommended_tile"][1],
            ),
            _ANSI_TILE,
        )
    )
    print(
        _runtime_color(
            "strategy: binning_sort={}".format(
                report["applied_binning_sort_mode"],
            ),
            _ANSI_STRATEGY,
        )
    )


def initialize_runtime_tuning(device: torch.device | str | None = None, verbose: bool = True) -> dict[str, Any]:
    if wp is None:
        raise ImportError("warp-lang is required for the Warp backend.")
    wp.init()
    runtime_device = _normalize_runtime_device(device)
    device_key = str(runtime_device)
    report = _RUNTIME_TUNING_CACHE.get(device_key)
    if report is None:
        report = _build_runtime_tuning_report(runtime_device)
        _RUNTIME_TUNING_CACHE[device_key] = report
    else:
        report = _refresh_runtime_tuning_report_memory(report)
        _RUNTIME_TUNING_CACHE[device_key] = report
    if verbose and device_key not in _RUNTIME_TUNING_LOGGED_DEVICES:
        _print_runtime_tuning_report(report)
        _RUNTIME_TUNING_LOGGED_DEVICES.add(device_key)
    return dict(report)


def get_runtime_tuning_report(device: torch.device | str | None = None) -> dict[str, Any]:
    runtime_device = _normalize_runtime_device(device)
    device_key = str(runtime_device)
    cached = _RUNTIME_TUNING_CACHE.get(device_key)
    if cached is None:
        return initialize_runtime_tuning(runtime_device, verbose=False)
    refreshed = _refresh_runtime_tuning_report_memory(cached)
    _RUNTIME_TUNING_CACHE[device_key] = refreshed
    return dict(refreshed)


# -----------------------------------------------------------------------------
# Exported runtime configuration
# -----------------------------------------------------------------------------


def is_available() -> bool:
    return wp is not None


def clear_warp_caches() -> None:
    """Release all grow-only buffer caches.

    Call after ``densify_and_prune`` or when switching scenes to reclaim GPU
    memory held by stale high-water-mark allocations.  The caches will be
    lazily re-populated on the next forward pass.
    """
    # Synchronize to ensure no in-flight kernels reference cached buffers.
    torch.cuda.synchronize()

    # A: grow-only GPU buffer caches (~128 MB at 2M Gaussians)
    _RADIX_SORT_BUFFER_CACHE.clear()
    _RADIX_SORT_I32_BUFFER_CACHE.clear()
    _INDEX_GATHER_I32_BUFFER_CACHE.clear()
    _INDEX_GATHER_I64_BUFFER_CACHE.clear()
    _SCAN_I32_BUFFER_CACHE.clear()
    _PROJECT_VISIBLE_BUFFER_CACHE.clear()
    _SEQUENCE_BUFFER_CACHE.clear()
    _DEPTH_SORT_ORDER_CACHE.clear()
    _DEPTH_SORT_CHECK_FLAG.clear()

    # B: C4 launch object caches (no GPU memory, just Python objects)
    _C4_LAUNCH_CACHE_SH.clear()
    _C4_LAUNCH_CACHE_COV3D.clear()
    _C4_LAUNCH_CACHE_RENDER_BWD.clear()
    _C4_LAUNCH_CACHE_PROJ_MEANS.clear()
    _C4_LAUNCH_CACHE_COV2D.clear()
    _C4_LAUNCH_CACHE_ACCUM.clear()
    _C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS.clear()
    _C4_LAUNCH_CACHE_SH_V3.clear()
    _C4_LAUNCH_CACHE_FWD_SH.clear()
    _C4_LAUNCH_CACHE_FWD_PREPROCESS.clear()
    _C4_LAUNCH_CACHE_FWD_RENDER.clear()
    _C4_LAUNCH_CACHE_BINNING_DUPLICATE.clear()
    _C4_LAUNCH_CACHE_FWD_RENDER_TILED256.clear()

    # B5: reset point-count tracker so next forward doesn't false-trigger
    global _PREV_POINT_COUNT
    _PREV_POINT_COUNT = None


def get_backward_mode() -> str:
    return _BACKWARD_MODE


def set_backward_mode(mode: str) -> None:
    global _BACKWARD_MODE
    if mode != "manual":
        raise ValueError("mode must be 'manual'")
    _BACKWARD_MODE = mode


def get_binning_sort_mode() -> str:
    return _BINNING_SORT_MODE


def set_binning_sort_mode(mode: str) -> None:
    global _BINNING_SORT_MODE
    if mode not in BINNING_SORT_MODES:
        raise ValueError("mode must be one of 'torch', 'warp_radix', or 'warp_depth_stable_tile'")
    _BINNING_SORT_MODE = mode


# -----------------------------------------------------------------------------
# Tensor allocation and forward-state packing
# -----------------------------------------------------------------------------


def _can_use_warp_scalar_alloc(device: torch.device | str) -> bool:
    return wp is not None and _normalize_runtime_device(device).type == "cuda"


def _get_runtime_warp_device(device: torch.device | str) -> str:
    runtime_device = _normalize_runtime_device(device)
    if not _AUTO_TUNE_ENABLED or not _WARP_INITIALIZED:
        return str(runtime_device)
    return str(get_runtime_tuning_report(runtime_device)["device"])


def _get_warp_dtype(torch_dtype: torch.dtype):

    mapping = {
        torch.bool: wp.bool,
        torch.uint8: wp.uint8,
        torch.int32: wp.int32,
        torch.int64: wp.int64,
        torch.float32: wp.float32,
    }
    warp_dtype = mapping.get(torch_dtype)
    if warp_dtype is None:
        raise TypeError(f"Unsupported Warp allocation dtype: {torch_dtype}")
    return warp_dtype


def _allocate_warp_scalar_array(shape, dtype: torch.dtype, device: torch.device | str, fill_value: Any | None = None) -> tuple[Any, torch.Tensor]:
    warp_dtype = _get_warp_dtype(dtype)
    warp_device = _get_runtime_warp_device(device)
    if fill_value is None:
        warp_array = wp.empty(shape=shape, dtype=warp_dtype, device=warp_device)
    elif fill_value is False or fill_value == 0:
        warp_array = wp.zeros(shape=shape, dtype=warp_dtype, device=warp_device)
    elif fill_value is True or fill_value == 1:
        warp_array = wp.ones(shape=shape, dtype=warp_dtype, device=warp_device)
    else:
        warp_array = wp.full(shape=shape, value=fill_value, dtype=warp_dtype, device=warp_device)
    return warp_array, wp.to_torch(warp_array)


def _allocate_scalar_tensor(shape, dtype: torch.dtype, device: torch.device | str, fill_value: Any | None = None) -> torch.Tensor:
    if _can_use_warp_scalar_alloc(device):
        _warp_array, tensor = _allocate_warp_scalar_array(shape, dtype, device, fill_value=fill_value)
        return tensor
    if fill_value is None:
        return torch.empty(shape, dtype=dtype, device=device)
    if fill_value is False or fill_value == 0:
        return torch.zeros(shape, dtype=dtype, device=device)
    if fill_value is True or fill_value == 1:
        return torch.ones(shape, dtype=dtype, device=device)
    return torch.full(shape, fill_value, dtype=dtype, device=device)


def _as_detached_contiguous_dtype(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.requires_grad:
        tensor = tensor.detach()
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


def _pack_forward_aux_buffers(preprocess_outputs, binning_state, n_contrib):
    points_xy_image_tensor = _as_detached_contiguous_dtype(preprocess_outputs["points_xy_image"], torch.float32)
    depths_tensor = _as_detached_contiguous_dtype(preprocess_outputs["depths"], torch.float32)
    conic_opacity_tensor = _as_detached_contiguous_dtype(preprocess_outputs["conic_opacity"], torch.float32)
    rgb_tensor = _as_detached_contiguous_dtype(preprocess_outputs["rgb"], torch.float32)
    cov3d_tensor = _as_detached_contiguous_dtype(preprocess_outputs["cov3d_all"], torch.float32)
    # L2: store clamped as int32 — avoids backward int32 conversion allocation
    clamped_tensor = _as_detached_contiguous_dtype(preprocess_outputs["clamped"], torch.int32)
    point_list_tensor = _as_detached_contiguous_dtype(binning_state["point_list"], torch.int32)
    ranges_tensor = _as_detached_contiguous_dtype(binning_state["ranges"].reshape(-1), torch.int32)
    img_tensor = _as_detached_contiguous_dtype(n_contrib.reshape(-1), torch.int32)

    # F4: torch.cat already returns contiguous tensor; skip redundant .contiguous()
    geom_buffer = torch.cat([
        points_xy_image_tensor.reshape(-1).view(torch.uint8),
        depths_tensor.reshape(-1).view(torch.uint8),
        conic_opacity_tensor.reshape(-1).view(torch.uint8),
        rgb_tensor.reshape(-1).view(torch.uint8),
        cov3d_tensor.reshape(-1).view(torch.uint8),
        clamped_tensor.view(torch.uint8).reshape(-1),
    ])
    binning_buffer = torch.cat([point_list_tensor, ranges_tensor]).view(torch.uint8)
    img_buffer = img_tensor.view(torch.uint8)
    return geom_buffer, binning_buffer, img_buffer


def _unpack_forward_aux_buffers(geom_buffer, binning_buffer, img_buffer, num_rendered, image_height, image_width):
    if geom_buffer.numel() == 0 or binning_buffer.numel() == 0 or img_buffer.numel() == 0:
        return None

    grid_x = (image_width + BLOCK_X - 1) // BLOCK_X
    grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
    tile_count = grid_x * grid_y
    point_count = geom_buffer.numel() // FORWARD_GEOM_STRIDE_BYTES
    points_xy_bytes = point_count * 2 * 4
    depths_bytes = point_count * 4
    conic_opacity_bytes = point_count * 4 * 4
    rgb_bytes = point_count * 3 * 4
    cov_bytes = point_count * 6 * 4
    geom_clamp_bytes = point_count * FORWARD_GEOM_CLAMP_WIDTH
    # point_list in binning_buffer includes RENDER_TILE_BATCH padding appended
    # by _build_binning_state, so account for it when validating buffer size.
    _i32_size = 4  # torch.int32 element size
    expected_binning_bytes = (num_rendered + RENDER_TILE_BATCH + tile_count * 2) * _i32_size
    expected_img_bytes = image_height * image_width * _i32_size
    if geom_buffer.numel() != points_xy_bytes + depths_bytes + conic_opacity_bytes + rgb_bytes + cov_bytes + geom_clamp_bytes or binning_buffer.numel() != expected_binning_bytes or img_buffer.numel() != expected_img_bytes:
        return None

    offset = 0
    points_xy_image_tensor = geom_buffer[offset : offset + points_xy_bytes].view(torch.float32).reshape(point_count, 2)
    offset = offset + points_xy_bytes
    depths_tensor = geom_buffer[offset : offset + depths_bytes].view(torch.float32)
    offset = offset + depths_bytes
    conic_opacity_tensor = geom_buffer[offset : offset + conic_opacity_bytes].view(torch.float32).reshape(point_count, 4)
    offset = offset + conic_opacity_bytes
    rgb_tensor = geom_buffer[offset : offset + rgb_bytes].view(torch.float32).reshape(point_count, 3)
    offset = offset + rgb_bytes
    cov3d_tensor = geom_buffer[offset : offset + cov_bytes].view(torch.float32).reshape(point_count, 6)
    offset = offset + cov_bytes
    # L2: clamped stored as int32 — direct view, no backward conversion needed
    clamped_tensor = geom_buffer[offset:].view(torch.int32).reshape(point_count, NUM_CHANNELS)
    binning_tensor = binning_buffer.view(torch.int32)
    img_tensor = img_buffer.view(torch.int32)

    preprocess_outputs = {
        "points_xy_image": points_xy_image_tensor,
        "depths": depths_tensor,
        "conic_opacity": conic_opacity_tensor,
        "rgb": rgb_tensor,
        "clamped": clamped_tensor,
        "cov3d_all": cov3d_tensor,
        "radii": torch.empty((point_count,), dtype=torch.int32, device=geom_buffer.device),
    }
    binning_state = {
        "grid_x": grid_x,
        "grid_y": grid_y,
        "point_offsets": torch.empty((0,), dtype=torch.int32, device=geom_buffer.device),
        "point_list": binning_tensor[:num_rendered + RENDER_TILE_BATCH],
        "point_list_keys": torch.empty((0,), dtype=torch.int64, device=geom_buffer.device),
        "ranges": binning_tensor[num_rendered + RENDER_TILE_BATCH:].reshape(tile_count, 2),
        "num_rendered": num_rendered,
    }
    n_contrib = img_tensor.reshape(image_height, image_width)
    return preprocess_outputs, binning_state, n_contrib


# -----------------------------------------------------------------------------
# Warp integration, buffer caches, and sorting helpers
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# C4: Warp launch cache — eliminates wp.from_torch + wp.launch Python overhead
# -----------------------------------------------------------------------------

def _require_warp() -> None:
    """Ensure Warp is imported AND runtime tuning has been executed.

    Designed to be called at public entry points only.  After the first
    successful invocation the function short-circuits via *_WARP_INITIALIZED*
    so that repeated calls in a training loop have near-zero overhead (one
    global bool check)."""
    global _WARP_INITIALIZED
    if _WARP_INITIALIZED:
        return
    if wp is None:
        raise ImportError("warp-lang is required for the Warp backend.")
    if _AUTO_TUNE_ENABLED:
        initialize_runtime_tuning(verbose=_AUTO_TUNE_VERBOSE)
    else:
        wp.init()
    _WARP_INITIALIZED = True


def _pack_binning_sort_keys(tile_ids: torch.Tensor, point_list: torch.Tensor, depths: torch.Tensor):
        tile_ids_i32 = _as_detached_contiguous_dtype(tile_ids, torch.int32)
        point_list_i32 = _as_detached_contiguous_dtype(point_list, torch.int32)
        depths_f32 = _as_detached_contiguous_dtype(depths, torch.float32)
        packed_keys, packed_keys_warp = _get_index_gather_i64_buffer(tile_ids.device, point_list.shape[0])
        wp.launch(
            kernel=_pack_binning_keys_warp_kernel,
            dim=point_list.shape[0],
            inputs=[
                wp.from_torch(tile_ids_i32, dtype=wp.int32),
                wp.from_torch(point_list_i32, dtype=wp.int32),
                wp.from_torch(depths_f32, dtype=wp.float32),
            ],
            outputs=[
                packed_keys_warp if packed_keys_warp is not None else wp.from_torch(packed_keys, dtype=wp.int64),
            ],
            device=str(tile_ids.device),
        )
        return packed_keys


def _get_radix_sort_buffers(device: torch.device, required_count: int):
    device_key = str(device)
    cached = _RADIX_SORT_BUFFER_CACHE.get(device_key)
    if cached is not None:
        key_warp, key_buffer, value_warp, value_buffer = cached
        if key_buffer.numel() >= required_count and value_buffer.numel() >= required_count:
            return key_buffer, value_buffer, key_warp, value_warp

    if _can_use_warp_scalar_alloc(device):
        key_warp, key_buffer = _allocate_warp_scalar_array(required_count, torch.int64, device)
        value_warp, value_buffer = _allocate_warp_scalar_array(required_count, torch.int32, device)
    else:
        key_warp = None
        value_warp = None
        key_buffer = torch.empty((required_count,), dtype=torch.int64, device=device)
        value_buffer = torch.empty((required_count,), dtype=torch.int32, device=device)
    _RADIX_SORT_BUFFER_CACHE[device_key] = (key_warp, key_buffer, value_warp, value_buffer)
    return key_buffer, value_buffer, key_warp, value_warp


def _get_radix_sort_i32_buffers(device: torch.device, required_count: int):
    device_key = str(device)
    cached = _RADIX_SORT_I32_BUFFER_CACHE.get(device_key)
    if cached is not None:
        key_warp, key_buffer, value_warp, value_buffer = cached
        if key_buffer.numel() >= required_count and value_buffer.numel() >= required_count:
            return key_buffer, value_buffer, key_warp, value_warp

    if _can_use_warp_scalar_alloc(device):
        key_warp, key_buffer = _allocate_warp_scalar_array(required_count, torch.int32, device)
        value_warp, value_buffer = _allocate_warp_scalar_array(required_count, torch.int32, device)
    else:
        key_warp = None
        value_warp = None
        key_buffer = torch.empty((required_count,), dtype=torch.int32, device=device)
        value_buffer = torch.empty((required_count,), dtype=torch.int32, device=device)
    _RADIX_SORT_I32_BUFFER_CACHE[device_key] = (key_warp, key_buffer, value_warp, value_buffer)
    return key_buffer, value_buffer, key_warp, value_warp


def _get_index_gather_i32_buffer(device: torch.device, required_count: int):
    device_key = str(device)
    cached = _INDEX_GATHER_I32_BUFFER_CACHE.get(device_key)
    if cached is not None:
        warp_buffer, tensor_buffer = cached
        if tensor_buffer.numel() >= required_count:
            return tensor_buffer[:required_count], warp_buffer

    if _can_use_warp_scalar_alloc(device):
        warp_buffer, buffer = _allocate_warp_scalar_array(required_count, torch.int32, device)
    else:
        warp_buffer = None
        buffer = torch.empty((required_count,), dtype=torch.int32, device=device)
    _INDEX_GATHER_I32_BUFFER_CACHE[device_key] = (warp_buffer, buffer)
    return buffer, warp_buffer


def _get_index_gather_i64_buffer(device: torch.device, required_count: int):
    device_key = str(device)
    cached = _INDEX_GATHER_I64_BUFFER_CACHE.get(device_key)
    if cached is not None:
        warp_buffer, tensor_buffer = cached
        if tensor_buffer.numel() >= required_count:
            return tensor_buffer[:required_count], warp_buffer

    if _can_use_warp_scalar_alloc(device):
        warp_buffer, buffer = _allocate_warp_scalar_array(required_count, torch.int64, device)
    else:
        warp_buffer = None
        buffer = torch.empty((required_count,), dtype=torch.int64, device=device)
    _INDEX_GATHER_I64_BUFFER_CACHE[device_key] = (warp_buffer, buffer)
    return buffer, warp_buffer


def _get_scan_i32_buffer(device: torch.device, required_count: int):
    device_key = str(device)
    cached = _SCAN_I32_BUFFER_CACHE.get(device_key)
    if cached is not None:
        warp_buffer, tensor_buffer, warp_size = cached
        if tensor_buffer.numel() >= required_count:
            sliced = tensor_buffer[:required_count]
            if warp_buffer is not None and warp_size == required_count:
                return sliced, warp_buffer
            new_warp = wp.from_torch(sliced, dtype=wp.int32) if _can_use_warp_scalar_alloc(device) else None
            _SCAN_I32_BUFFER_CACHE[device_key] = (new_warp, tensor_buffer, required_count)
            return sliced, new_warp

    if _can_use_warp_scalar_alloc(device):
        warp_buffer, buffer = _allocate_warp_scalar_array(required_count, torch.int32, device)
    else:
        warp_buffer = None
        buffer = torch.empty((required_count,), dtype=torch.int32, device=device)
    _SCAN_I32_BUFFER_CACHE[device_key] = (warp_buffer, buffer, required_count)
    return buffer, warp_buffer


def _get_project_visible_buffers(device: torch.device, point_count: int):
    device_key = str(device)
    cached = _PROJECT_VISIBLE_BUFFER_CACHE.get(device_key)
    if cached is not None:
        visible_mask, p_proj, p_view_z = cached
        if visible_mask.shape[0] >= point_count and p_proj.shape[0] >= point_count and p_view_z.shape[0] >= point_count:
            return visible_mask[:point_count], p_proj[:point_count], p_view_z[:point_count]

    visible_mask = torch.empty((point_count,), dtype=torch.int32, device=device)
    p_proj = torch.empty((point_count, 3), dtype=torch.float32, device=device)
    p_view_z = torch.empty((point_count,), dtype=torch.float32, device=device)
    _PROJECT_VISIBLE_BUFFER_CACHE[device_key] = (visible_mask, p_proj, p_view_z)
    return visible_mask, p_proj, p_view_z


def _get_sequence_buffer(device: torch.device, required_count: int):
    device_key = str(device)
    cached = _SEQUENCE_BUFFER_CACHE.get(device_key)
    if cached is not None and cached.numel() >= required_count:
        return cached[:required_count]

    sequence = torch.arange(required_count, dtype=torch.int32, device=device)
    _SEQUENCE_BUFFER_CACHE[device_key] = sequence
    return sequence


def _inclusive_scan_i32(src: torch.Tensor):
    if src.numel() == 0:
        return _allocate_scalar_tensor((0,), torch.int32, src.device)

    src_i32 = _as_detached_contiguous_dtype(src, torch.int32)

    scanned, scanned_warp = _get_scan_i32_buffer(src_i32.device, src_i32.shape[0])
    wp.utils.array_scan(
        wp.from_torch(src_i32, dtype=wp.int32),
        scanned_warp if scanned_warp is not None else wp.from_torch(scanned, dtype=wp.int32),
        inclusive=True,
    )
    return scanned


def _gather_i32_by_index(src: torch.Tensor, indices: torch.Tensor):
    src_i32 = _as_detached_contiguous_dtype(src, torch.int32)
    indices_i32 = _as_detached_contiguous_dtype(indices, torch.int32)
    gathered, gathered_warp = _get_index_gather_i32_buffer(src.device, indices.shape[0])
    wp.launch(
        kernel=_gather_i32_by_index_warp_kernel,
        dim=indices.shape[0],
        inputs=[
            wp.from_torch(src_i32, dtype=wp.int32),
            wp.from_torch(indices_i32, dtype=wp.int32),
        ],
        outputs=[gathered_warp if gathered_warp is not None else wp.from_torch(gathered, dtype=wp.int32)],
        device=str(src.device),
    )
    return gathered


def _warp_radix_sort_pairs_in_place(key_buffer: torch.Tensor, value_buffer: torch.Tensor, count: int):
    key_warp = None
    value_warp = None
    cached = _RADIX_SORT_BUFFER_CACHE.get(str(key_buffer.device))
    if cached is not None:
        cached_key_warp, cached_key_buffer, cached_value_warp, cached_value_buffer = cached
        if cached_key_buffer.data_ptr() == key_buffer.data_ptr() and cached_value_buffer.data_ptr() == value_buffer.data_ptr():
            key_warp = cached_key_warp
            value_warp = cached_value_warp
    wp.utils.radix_sort_pairs(
        key_warp if key_warp is not None else wp.from_torch(key_buffer, dtype=wp.int64),
        value_warp if value_warp is not None else wp.from_torch(value_buffer, dtype=wp.int32),
        count,
    )
    return key_buffer[:count], value_buffer[:count]


def _warp_radix_sort_i32_pairs_in_place(key_buffer: torch.Tensor, value_buffer: torch.Tensor, count: int):
    key_warp = None
    value_warp = None
    cached = _RADIX_SORT_I32_BUFFER_CACHE.get(str(key_buffer.device))
    if cached is not None:
        cached_key_warp, cached_key_buffer, cached_value_warp, cached_value_buffer = cached
        if cached_key_buffer.data_ptr() == key_buffer.data_ptr() and cached_value_buffer.data_ptr() == value_buffer.data_ptr():
            key_warp = cached_key_warp
            value_warp = cached_value_warp
    _key_arr = key_warp if key_warp is not None else wp.from_torch(key_buffer, dtype=wp.int32)
    _val_arr = value_warp if value_warp is not None else wp.from_torch(value_buffer, dtype=wp.int32)
    wp.utils.radix_sort_pairs(
        _key_arr,
        _val_arr,
        count,
    )
    return key_buffer[:count], value_buffer[:count]

# -----------------------------------------------------------------------------
# Warp kernel definitions
# -----------------------------------------------------------------------------


if wp is not None:


    @wp.kernel
    def _gather_i32_by_index_warp_kernel(
        src: wp.array(dtype=wp.int32),
        indices: wp.array(dtype=wp.int32),
        out: wp.array(dtype=wp.int32),
    ):
        idx = wp.tid()
        out[idx] = src[indices[idx]]

    @wp.kernel
    def _cov3d_from_scale_rotation_warp_kernel(
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        out_cov3d_flat: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        s = scales[tid] * scale_modifier
        q = rotations[tid]
        r = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y + r * z)
        r02 = 2.0 * (x * z - r * y)
        r10 = 2.0 * (x * y - r * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z + r * x)
        r20 = 2.0 * (x * z + r * y)
        r21 = 2.0 * (y * z - r * x)
        r22 = 1.0 - 2.0 * (x * x + y * y)

        m00 = s[0] * r00
        m01 = s[0] * r01
        m02 = s[0] * r02
        m10 = s[1] * r10
        m11 = s[1] * r11
        m12 = s[1] * r12
        m20 = s[2] * r20
        m21 = s[2] * r21
        m22 = s[2] * r22

        base = tid * 6
        out_cov3d_flat[base + 0] = m00 * m00 + m10 * m10 + m20 * m20
        out_cov3d_flat[base + 1] = m00 * m01 + m10 * m11 + m20 * m21
        out_cov3d_flat[base + 2] = m00 * m02 + m10 * m12 + m20 * m22
        out_cov3d_flat[base + 3] = m01 * m01 + m11 * m11 + m21 * m21
        out_cov3d_flat[base + 4] = m01 * m02 + m11 * m12 + m21 * m22
        out_cov3d_flat[base + 5] = m02 * m02 + m12 * m12 + m22 * m22


    @wp.func
    def _cov2d_from_scale_rotation_gram_wp(
        scale: wp.vec3,
        rotation: wp.vec4,
        scale_modifier: wp.float32,
        t00: wp.float32,
        t10: wp.float32,
        t20: wp.float32,
        t01: wp.float32,
        t11: wp.float32,
        t21: wp.float32,
    ):
        s = scale * scale_modifier
        r = rotation[0]
        xq = rotation[1]
        yq = rotation[2]
        zq = rotation[3]

        r00 = 1.0 - 2.0 * (yq * yq + zq * zq)
        r01 = 2.0 * (xq * yq + r * zq)
        r02 = 2.0 * (xq * zq - r * yq)
        r10 = 2.0 * (xq * yq - r * zq)
        r11 = 1.0 - 2.0 * (xq * xq + zq * zq)
        r12 = 2.0 * (yq * zq + r * xq)
        r20 = 2.0 * (xq * zq + r * yq)
        r21 = 2.0 * (yq * zq - r * xq)
        r22 = 1.0 - 2.0 * (xq * xq + yq * yq)

        m00 = s[0] * r00
        m01 = s[0] * r01
        m02 = s[0] * r02
        m10 = s[1] * r10
        m11 = s[1] * r11
        m12 = s[1] * r12
        m20 = s[2] * r20
        m21 = s[2] * r21
        m22 = s[2] * r22

        # A = M * T  (3x2)
        a00 = m00 * t00 + m01 * t10 + m02 * t20
        a10 = m10 * t00 + m11 * t10 + m12 * t20
        a20 = m20 * t00 + m21 * t10 + m22 * t20
        a01 = m00 * t01 + m01 * t11 + m02 * t21
        a11 = m10 * t01 + m11 * t11 + m12 * t21
        a21 = m20 * t01 + m21 * t11 + m22 * t21

        # cov2d = A^T * A + 0.3*I  (2x2 symmetric, guaranteed PSD)
        return wp.vec3(
            a00 * a00 + a10 * a10 + a20 * a20 + 0.3,
            a00 * a01 + a10 * a11 + a20 * a21,
            a01 * a01 + a11 * a11 + a21 * a21 + 0.3,
        )


    # W1: constant needed by warp32 backward kernel
    _BLOCK_PIXELS_C = wp.constant(BLOCK_X * BLOCK_Y)

    # ---- Tiled-256 forward render kernel (cooperative Gaussian loading) ----
    # Each block = 256 threads = one 16×16 pixel tile.
    # Gaussians are loaded in batches of 256 via wp.tile() + wp.tile_extract(),
    # eliminating 256× redundant global memory reads compared to per-pixel SIMT.
    @wp.kernel
    def _render_tiles_tiled256_warp_kernel(
        ranges_flat: wp.array(dtype=wp.int32),
        point_list: wp.array(dtype=wp.int32),
        points_xy_image: wp.array(dtype=wp.vec2),
        features_flat: wp.array(dtype=wp.float32),
        depths: wp.array(dtype=wp.float32),
        conic_opacity: wp.array(dtype=wp.vec4),
        background: wp.array(dtype=wp.float32),
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        num_tiles: wp.int32,
        compute_depth: wp.int32,
        out_color_flat: wp.array(dtype=wp.float32),
        out_depth_flat: wp.array(dtype=wp.float32),
        out_alpha_flat: wp.array(dtype=wp.float32),
        n_contrib: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        tile_id = tid // _BLOCK_PIXELS_C
        local_id = tid % _BLOCK_PIXELS_C

        if tile_id >= num_tiles:
            return

        # Map local_id → pixel within the 16×16 tile
        tile_x = tile_id % grid_x
        tile_y = tile_id // grid_x
        pix_x = tile_x * BLOCK_X + (local_id % BLOCK_X)
        pix_y = tile_y * BLOCK_Y + (local_id // BLOCK_X)
        total_pixels = image_width * image_height

        inside = int(0)
        if pix_x < image_width and pix_y < image_height:
            inside = 1
        pixel_id = pix_y * image_width + pix_x

        start = ranges_flat[tile_id * 2]
        end = ranges_flat[tile_id * 2 + 1]
        n_gs = end - start

        T = float(1.0)
        contributor = int(0)
        last_contributor = int(0)
        color0 = float(0.0)
        color1 = float(0.0)
        color2 = float(0.0)
        weight = float(0.0)
        depth_acc = float(0.0)
        pixf_x = float(pix_x)
        pixf_y = float(pix_y)
        done = int(0)
        if inside == 0:
            done = 1

        rounds = (n_gs + 255) // 256
        for i in range(rounds):
            # ---- Block-level early-exit vote ----
            t_done = wp.tile(done)
            total_done = wp.tile_sum(t_done)
            all_done = wp.tile_extract(total_done, 0)
            if all_done == _BLOCK_PIXELS_C:
                break

            # ---- Cooperative load: each thread loads 1 Gaussian ----
            progress = i * 256 + local_id
            # 2C: simplified safe_idx — end > start guaranteed by n_gs > 0
            safe_idx = wp.min(start + progress, end - 1)
            my_id = point_list[safe_idx]

            my_xy = points_xy_image[my_id]
            my_co = conic_opacity[my_id]

            # 2A: vec2/vec4 tiles — fewer tile_extract calls (3 vs 7)
            t_id = wp.tile(my_id)
            t_xy = wp.tile(my_xy, preserve_type=True)
            t_co = wp.tile(my_co, preserve_type=True)

            # ---- Inner loop: all 256 pixels process the batch ----
            batch_count = wp.min(256, n_gs - i * 256)
            for j in range(batch_count):
                contributor = contributor + 1

                if done == 0:
                    coll_id = wp.tile_extract(t_id, j)
                    # 2A: extract vec2/vec4 directly
                    xy_j = wp.tile_extract(t_xy, j)
                    co_j = wp.tile_extract(t_co, j)

                    d_x = xy_j[0] - pixf_x
                    d_y = xy_j[1] - pixf_y
                    power = -0.5 * (co_j[0] * d_x * d_x + co_j[2] * d_y * d_y) - co_j[1] * d_x * d_y

                    if power <= 0.0:
                        alpha = wp.min(float(0.99), co_j[3] * wp.exp(power))
                        if alpha >= (1.0 / 255.0):
                            test_T = T * (1.0 - alpha)
                            if test_T < 0.0001:
                                done = 1
                            else:
                                contribution = alpha * T
                                feat_base = coll_id * NUM_CHANNELS
                                color0 = color0 + features_flat[feat_base + 0] * contribution
                                color1 = color1 + features_flat[feat_base + 1] * contribution
                                color2 = color2 + features_flat[feat_base + 2] * contribution
                                weight = weight + contribution
                                if compute_depth != 0:
                                    depth_acc = depth_acc + depths[coll_id] * contribution
                                T = test_T
                                last_contributor = contributor

        if inside != 0:
            n_contrib[pixel_id] = last_contributor
            out_color_flat[pixel_id] = color0 + T * background[0]
            out_color_flat[total_pixels + pixel_id] = color1 + T * background[1]
            out_color_flat[2 * total_pixels + pixel_id] = color2 + T * background[2]
            out_alpha_flat[pixel_id] = weight
            out_depth_flat[pixel_id] = depth_acc

    # ---- C3: Fused backward accumulation (replaces 3 torch.add + 1 slice-assign + 2 torch.zeros) ----
    @wp.kernel
    def _fused_backward_accumulate_warp_kernel(
        grad_projected_means: wp.array(dtype=wp.vec3),
        grad_cov_means: wp.array(dtype=wp.vec3),
        grad_sh_means: wp.array(dtype=wp.vec3),
        has_sh: wp.int32,
        render_grad_points: wp.array(dtype=wp.vec2),
        grad_means3D: wp.array(dtype=wp.vec3),
        grad_means2D: wp.array(dtype=wp.vec3),
    ):
        i = wp.tid()
        acc = grad_projected_means[i] + grad_cov_means[i]
        if has_sh != 0:
            acc = acc + grad_sh_means[i]
        grad_means3D[i] = acc
        rp = render_grad_points[i]
        grad_means2D[i] = wp.vec3(rp[0], rp[1], 0.0)

        # ---- C12: split SH backward into degree 0-1 and degree 2-3 kernels ----


    # ---- E1: Fused forward preprocess (project + cov3d + cov2d) ----
    # Merges _project_visible_points, _cov3d_from_scale_rotation, and
    # _cov2d_preprocess_masked_pack_scale_rotation into one kernel.
    # Benefits: rotation matrix M computed once; intermediate p_proj / p_view_z
    # / visible_mask stay in registers; 2 fewer kernel launches.
    @wp.kernel
    def _fused_project_cov3d_cov2d_preprocess_sr_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        opacities: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        proj_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        # outputs
        out_cov3d_flat: wp.array(dtype=wp.float32),
        visible_mask_out: wp.array(dtype=wp.int32),
        depths: wp.array(dtype=wp.float32),
        radii: wp.array(dtype=wp.int32),
        proj_2d: wp.array(dtype=wp.vec2),
        conic_2d: wp.array(dtype=wp.vec3),
        conic_2d_inv: wp.array(dtype=wp.vec3),
        points_xy_image: wp.array(dtype=wp.vec2),
        tiles_touched: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
    ):
        idx = wp.tid()
        cov_base = idx * 6

        # zero all outputs (covers invisible / invalid cases)
        out_cov3d_flat[cov_base + 0] = float(0.0)
        out_cov3d_flat[cov_base + 1] = float(0.0)
        out_cov3d_flat[cov_base + 2] = float(0.0)
        out_cov3d_flat[cov_base + 3] = float(0.0)
        out_cov3d_flat[cov_base + 4] = float(0.0)
        out_cov3d_flat[cov_base + 5] = float(0.0)
        depths[idx] = float(0.0)
        radii[idx] = int(0)
        proj_2d[idx] = wp.vec2(0.0, 0.0)
        conic_2d[idx] = wp.vec3(0.0, 0.0, 0.0)
        conic_2d_inv[idx] = wp.vec3(0.0, 0.0, 0.0)
        points_xy_image[idx] = wp.vec2(0.0, 0.0)
        tiles_touched[idx] = int(0)
        conic_opacity[idx] = wp.vec4(0.0, 0.0, 0.0, 0.0)

        p = means3d[idx]

        # ---- projection & visibility (from _project_visible_points) ----
        p_view_z = view_flat[2] * p[0] + view_flat[6] * p[1] + view_flat[10] * p[2] + view_flat[14]
        if p_view_z <= 0.2:
            visible_mask_out[idx] = int(0)
            return
        visible_mask_out[idx] = int(1)

        # full homogeneous projection
        hom_x = proj_flat[0] * p[0] + proj_flat[4] * p[1] + proj_flat[8] * p[2] + proj_flat[12]
        hom_y = proj_flat[1] * p[0] + proj_flat[5] * p[1] + proj_flat[9] * p[2] + proj_flat[13]
        hom_w = proj_flat[3] * p[0] + proj_flat[7] * p[1] + proj_flat[11] * p[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        p_proj_x = hom_x * inv_w
        p_proj_y = hom_y * inv_w

        # ---- rotation matrix M = diag(s)*R  (shared by cov3d & cov2d) ----
        s = scales[idx] * scale_modifier
        q = rotations[idx]
        rr = q[0]
        xq = q[1]
        yq = q[2]
        zq = q[3]

        r00 = 1.0 - 2.0 * (yq * yq + zq * zq)
        r01 = 2.0 * (xq * yq + rr * zq)
        r02 = 2.0 * (xq * zq - rr * yq)
        r10 = 2.0 * (xq * yq - rr * zq)
        r11 = 1.0 - 2.0 * (xq * xq + zq * zq)
        r12 = 2.0 * (yq * zq + rr * xq)
        r20 = 2.0 * (xq * zq + rr * yq)
        r21 = 2.0 * (yq * zq - rr * xq)
        r22 = 1.0 - 2.0 * (xq * xq + yq * yq)

        m00 = s[0] * r00
        m01 = s[0] * r01
        m02 = s[0] * r02
        m10 = s[1] * r10
        m11 = s[1] * r11
        m12 = s[1] * r12
        m20 = s[2] * r20
        m21 = s[2] * r21
        m22 = s[2] * r22

        # ---- cov3d = M^T M  (6 upper-triangle values, for backward) ----
        out_cov3d_flat[cov_base + 0] = m00 * m00 + m10 * m10 + m20 * m20
        out_cov3d_flat[cov_base + 1] = m00 * m01 + m10 * m11 + m20 * m21
        out_cov3d_flat[cov_base + 2] = m00 * m02 + m10 * m12 + m20 * m22
        out_cov3d_flat[cov_base + 3] = m01 * m01 + m11 * m11 + m21 * m21
        out_cov3d_flat[cov_base + 4] = m01 * m02 + m11 * m12 + m21 * m22
        out_cov3d_flat[cov_base + 5] = m02 * m02 + m12 * m12 + m22 * m22

        # ---- cov2d via Gram factorisation (reusing M from above) ----
        vx = view_flat[0] * p[0] + view_flat[4] * p[1] + view_flat[8] * p[2] + view_flat[12]
        vy = view_flat[1] * p[0] + view_flat[5] * p[1] + view_flat[9] * p[2] + view_flat[13]
        vz = p_view_z

        limx = 1.3 * tanfovx
        limy = 1.3 * tanfovy
        txtz = vx / vz
        tytz = vy / vz
        vx = wp.clamp(txtz, -limx, limx) * vz
        vy = wp.clamp(tytz, -limy, limy) * vz

        a_j = focal_x / vz
        b_j = focal_y / vz
        c_j = -(focal_x * vx) / (vz * vz)
        d_j = -(focal_y * vy) / (vz * vz)

        t00 = view_flat[0] * a_j + view_flat[2] * c_j
        t10 = view_flat[4] * a_j + view_flat[6] * c_j
        t20 = view_flat[8] * a_j + view_flat[10] * c_j
        t01 = view_flat[1] * b_j + view_flat[2] * d_j
        t11 = view_flat[5] * b_j + view_flat[6] * d_j
        t21 = view_flat[9] * b_j + view_flat[10] * d_j

        # A = M * T  (3x2)
        a00 = m00 * t00 + m01 * t10 + m02 * t20
        a10 = m10 * t00 + m11 * t10 + m12 * t20
        a20 = m20 * t00 + m21 * t10 + m22 * t20
        a01 = m00 * t01 + m01 * t11 + m02 * t21
        a11 = m10 * t01 + m11 * t11 + m12 * t21
        a21 = m20 * t01 + m21 * t11 + m22 * t21

        # cov2d = A^T A + 0.3*I  (2x2 symmetric PSD)
        cov_a = a00 * a00 + a10 * a10 + a20 * a20 + 0.3
        cov_b = a00 * a01 + a10 * a11 + a20 * a21
        cov_c = a01 * a01 + a11 * a11 + a21 * a21 + 0.3

        det = cov_a * cov_c - cov_b * cov_b
        if wp.abs(det) <= DET_EPSILON:
            return

        det_inv = 1.0 / det
        conic = wp.vec3(cov_c * det_inv, -cov_b * det_inv, cov_a * det_inv)

        mid = 0.5 * (cov_a + cov_c)
        root = wp.sqrt(wp.max(0.1, mid * mid - det))
        lambda1 = mid + root
        lambda2 = mid - root
        radius = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(lambda1, lambda2))))

        point_x = _ndc_to_pix_wp(p_proj_x, image_width)
        point_y = _ndc_to_pix_wp(p_proj_y, image_height)

        rect = _compute_tile_rect_snugbox_cov2d_wp(point_x, point_y, cov_a, cov_c, opacities[idx], grid_x, grid_y)
        rect_area = (rect[2] - rect[0]) * (rect[3] - rect[1])
        if rect_area == 0:
            return

        point_image = wp.vec2(point_x, point_y)
        depths[idx] = p_view_z
        radii[idx] = radius
        proj_2d[idx] = point_image
        conic_2d[idx] = conic
        conic_2d_inv[idx] = wp.vec3(cov_a, cov_b, cov_c)
        points_xy_image[idx] = point_image
        tiles_touched[idx] = rect_area
        conic_opacity[idx] = wp.vec4(conic[0], conic[1], conic[2], opacities[idx])


    # ---- E2: Fused backward preprocess + accumulate ----
    # Merges _backward_projected_means, _backward_cov2d_cov3d_fused, and
    # _fused_backward_accumulate into one kernel.  Benefits: means3D /
    # viewmatrix / radii read once; intermediate grad_projected / grad_cov_means
    # eliminated; 2 fewer kernel launches.
    @wp.kernel
    def _fused_backward_preprocess_accumulate_warp_kernel(
        # --- inputs (projected means part) ---
        means3d: wp.array(dtype=wp.vec3),
        radii: wp.array(dtype=wp.int32),
        proj_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        grad_mean2d: wp.array(dtype=wp.vec2),
        grad_proj_2d: wp.array(dtype=wp.vec2),
        grad_depths: wp.array(dtype=wp.float32),
        # --- inputs (cov2d + cov3d fused part) ---
        cov3d_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        grad_conic_opacity: wp.array(dtype=wp.vec4),
        grad_conic_2d_flat: wp.array(dtype=wp.float32),
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        # --- inputs (accumulate part) ---
        grad_sh_means: wp.array(dtype=wp.vec3),
        has_sh: wp.int32,
        render_grad_points: wp.array(dtype=wp.vec2),
        # --- outputs ---
        grad_means3D: wp.array(dtype=wp.vec3),
        grad_means2D: wp.array(dtype=wp.vec3),
        grad_scales_out: wp.array(dtype=wp.vec3),
        grad_rotations_out: wp.array(dtype=wp.vec4),
    ):
        tid = wp.tid()
        mean = means3d[tid]
        rad = radii[tid]

        # ======== Part A: backward_projected_means ========
        hom_x = proj_flat[0] * mean[0] + proj_flat[4] * mean[1] + proj_flat[8] * mean[2] + proj_flat[12]
        hom_y = proj_flat[1] * mean[0] + proj_flat[5] * mean[1] + proj_flat[9] * mean[2] + proj_flat[13]
        hom_w = proj_flat[3] * mean[0] + proj_flat[7] * mean[1] + proj_flat[11] * mean[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)

        grad_xy = grad_mean2d[tid]
        inv_w2 = inv_w * inv_w
        if rad > 0:
            grad_xy = grad_xy + grad_proj_2d[tid]
        gx = grad_xy[0]
        gy = grad_xy[1]
        gh = hom_x * gx + hom_y * gy
        pm_x = inv_w * (proj_flat[0] * gx + proj_flat[1] * gy) - proj_flat[3] * inv_w2 * gh
        pm_y = inv_w * (proj_flat[4] * gx + proj_flat[5] * gy) - proj_flat[7] * inv_w2 * gh
        pm_z = inv_w * (proj_flat[8] * gx + proj_flat[9] * gy) - proj_flat[11] * inv_w2 * gh
        
        # z should be defined in Part B, but for fused kernel, it can be here 
        z = view_flat[2] * mean[0] + view_flat[6] * mean[1] + view_flat[10] * mean[2] + view_flat[14]
        depth_grad = grad_depths[tid]
        pm_x = pm_x + (view_flat[2] - view_flat[3] * z) * depth_grad
        pm_y = pm_y + (view_flat[6] - view_flat[7] * z) * depth_grad
        pm_z = pm_z + (view_flat[10] - view_flat[11] * z) * depth_grad
        grad_projected = wp.vec3(pm_x, pm_y, pm_z)

        # ======== Part B: backward_cov2d_cov3d_fused ========
        grad_cov_means = wp.vec3(0.0, 0.0, 0.0)
        if rad <= 0:
            grad_scales_out[tid] = wp.vec3(0.0, 0.0, 0.0)
            grad_rotations_out[tid] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        else:
            base = tid * 6
            _gc_v4 = grad_conic_opacity[tid]
            grad_conic2_base = tid * 3

            x = view_flat[0] * mean[0] + view_flat[4] * mean[1] + view_flat[8] * mean[2] + view_flat[12]
            y = view_flat[1] * mean[0] + view_flat[5] * mean[1] + view_flat[9] * mean[2] + view_flat[13]
            z = view_flat[2] * mean[0] + view_flat[6] * mean[1] + view_flat[10] * mean[2] + view_flat[14]
            tz_inv = 1.0 / (z + 1.0e-7)
            limx = 1.3 * tanfovx
            limy = 1.3 * tanfovy
            txtz = x * tz_inv
            tytz = y * tz_inv
            x_grad_mul = float(1.0)
            y_grad_mul = float(1.0)
            if txtz < -limx or txtz > limx:
                x_grad_mul = float(0.0)
            if tytz < -limy or tytz > limy:
                y_grad_mul = float(0.0)
            tx = wp.clamp(txtz, -limx, limx) 
            ty = wp.clamp(tytz, -limy, limy)
            j00 = focal_x * tz_inv
            j02 = - j00 * tx
            j11 = focal_y * tz_inv
            j12 = - j11 * ty

            w00 = view_flat[0]
            w01 = view_flat[1]
            w02 = view_flat[2]
            w10 = view_flat[4]
            w11 = view_flat[5]
            w12 = view_flat[6]
            w20 = view_flat[8]
            w21 = view_flat[9]
            w22 = view_flat[10]

            t00 = w00 * j00 + w02 * j02
            t10 = w10 * j00 + w12 * j02
            t20 = w20 * j00 + w22 * j02
            t01 = w01 * j11 + w02 * j12
            t11 = w11 * j11 + w12 * j12
            t21 = w21 * j11 + w22 * j12

            v00 = cov3d_flat[base + 0]
            v01 = cov3d_flat[base + 1]
            v02 = cov3d_flat[base + 2]
            v11 = cov3d_flat[base + 3]
            v12 = cov3d_flat[base + 4]
            v22 = cov3d_flat[base + 5]

            vt0x = v00 * t00 + v01 * t10 + v02 * t20
            vt0y = v01 * t00 + v11 * t10 + v12 * t20
            vt0z = v02 * t00 + v12 * t10 + v22 * t20
            vt1x = v00 * t01 + v01 * t11 + v02 * t21
            vt1y = v01 * t01 + v11 * t11 + v12 * t21
            vt1z = v02 * t01 + v12 * t11 + v22 * t21

            aa = t00 * vt0x + t10 * vt0y + t20 * vt0z + 0.3
            bb = t00 * vt1x + t10 * vt1y + t20 * vt1z
            cc = t01 * vt1x + t11 * vt1y + t21 * vt1z + 0.3
            denom = aa * cc - bb * bb
            denom2inv = _conic_denom2inv_wp(denom)

            total_conic0 = _gc_v4[0] + grad_conic_2d_flat[grad_conic2_base + 0]
            total_conic1 = _gc_v4[1] + grad_conic_2d_flat[grad_conic2_base + 1]
            total_conic2 = _gc_v4[2] + grad_conic_2d_flat[grad_conic2_base + 2]

            dL_da = denom2inv * (-cc * cc * total_conic0 + 2.0 * bb * cc * total_conic1 - bb * bb * total_conic2)
            dL_dc = denom2inv * (-aa * aa * total_conic2 + 2.0 * aa * bb * total_conic1 - bb * bb  * total_conic0)
            dL_db = denom2inv * 2.0 * (bb * cc * total_conic0 - (aa * cc + bb * bb) * total_conic1 + aa * bb * total_conic2)

            gc0 = t00 * t00 * dL_da + t00 * t01 * dL_db + t01 * t01 * dL_dc
            gc3 = t10 * t10 * dL_da + t10 * t11 * dL_db + t11 * t11 * dL_dc
            gc5 = t20 * t20 * dL_da + t20 * t21 * dL_db + t21 * t21 * dL_dc
            gc1 = 2.0 * t00 * t10 * dL_da + (t00 * t11 + t01 * t10) * dL_db + 2.0 * t01 * t11 * dL_dc
            gc2 = 2.0 * t00 * t20 * dL_da + (t00 * t21 + t01 * t20) * dL_db + 2.0 * t01 * t21 * dL_dc
            gc4 = 2.0 * t10 * t20 * dL_da + (t10 * t21 + t11 * t20) * dL_db + 2.0 * t11 * t21 * dL_dc

            dL_dT00 = 2.0 * vt0x * dL_da + vt1x * dL_db
            dL_dT01 = 2.0 * vt0y * dL_da + vt1y * dL_db
            dL_dT02 = 2.0 * vt0z * dL_da + vt1z * dL_db
            dL_dT10 = vt0x * dL_db + 2.0 * vt1x * dL_dc
            dL_dT11 = vt0y * dL_db + 2.0 * vt1y * dL_dc
            dL_dT12 = vt0z * dL_db + 2.0 * vt1z * dL_dc

            dL_dJ00 = w00 * dL_dT00 + w10 * dL_dT01 + w20 * dL_dT02
            dL_dJ02 = w02 * dL_dT00 + w12 * dL_dT01 + w22 * dL_dT02
            dL_dJ11 = w01 * dL_dT10 + w11 * dL_dT11 + w21 * dL_dT12
            dL_dJ12 = w02 * dL_dT10 + w12 * dL_dT11 + w22 * dL_dT12

            dL_dtx = x_grad_mul * - j00 * tz_inv * dL_dJ02
            dL_dty = y_grad_mul * - j11 * tz_inv * dL_dJ12
            dL_dtz = - ( j00 * dL_dJ00 + j11 * dL_dJ11+ 2.0 *  j02 * dL_dJ02 + 2.0 * j12 *dL_dJ12 ) * tz_inv 
            grad_cov_means = wp.vec3(
                w00 * dL_dtx + w01 * dL_dty + w02 * dL_dtz,
                w10 * dL_dtx + w11 * dL_dty + w12 * dL_dtz,
                w20 * dL_dtx + w21 * dL_dty + w22 * dL_dtz,
            )

            # --- cov3d backward (uses local gc0..gc5) ---
            ss = scales[tid] * scale_modifier
            qq = rotations[tid]
            rr = qq[0]
            xq = qq[1]
            yq = qq[2]
            zq = qq[3]

            r00 = 1.0 - 2.0 * (yq * yq + zq * zq)
            r01 = 2.0 * (xq * yq + rr * zq)
            r02 = 2.0 * (xq * zq - rr * yq)
            r10 = 2.0 * (xq * yq - rr * zq)
            r11 = 1.0 - 2.0 * (xq * xq + zq * zq)
            r12 = 2.0 * (yq * zq + rr * xq)
            r20 = 2.0 * (xq * zq + rr * yq)
            r21 = 2.0 * (yq * zq - rr * xq)
            r22 = 1.0 - 2.0 * (xq * xq + yq * yq)

            mm00 = ss[0] * r00
            mm01 = ss[0] * r01
            mm02 = ss[0] * r02
            mm10 = ss[1] * r10
            mm11 = ss[1] * r11
            mm12 = ss[1] * r12
            mm20 = ss[2] * r20
            mm21 = ss[2] * r21
            mm22 = ss[2] * r22

            sigma00 = gc0
            sigma01 = 0.5 * gc1
            sigma02 = 0.5 * gc2
            sigma11 = gc3
            sigma12 = 0.5 * gc4
            sigma22 = gc5

            dM00 = 2.0 * (mm00 * sigma00 + mm01 * sigma01 + mm02 * sigma02)
            dM01 = 2.0 * (mm00 * sigma01 + mm01 * sigma11 + mm02 * sigma12)
            dM02 = 2.0 * (mm00 * sigma02 + mm01 * sigma12 + mm02 * sigma22)
            dM10 = 2.0 * (mm10 * sigma00 + mm11 * sigma01 + mm12 * sigma02)
            dM11 = 2.0 * (mm10 * sigma01 + mm11 * sigma11 + mm12 * sigma12)
            dM12 = 2.0 * (mm10 * sigma02 + mm11 * sigma12 + mm12 * sigma22)
            dM20 = 2.0 * (mm20 * sigma00 + mm21 * sigma01 + mm22 * sigma02)
            dM21 = 2.0 * (mm20 * sigma01 + mm21 * sigma11 + mm22 * sigma12)
            dM22 = 2.0 * (mm20 * sigma02 + mm21 * sigma12 + mm22 * sigma22)

            grad_scales_out[tid] = wp.vec3(
                scale_modifier * (dM00 * r00 + dM01 * r01 + dM02 * r02),
                scale_modifier * (dM10 * r10 + dM11 * r11 + dM12 * r12),
                scale_modifier * (dM20 * r20 + dM21 * r21 + dM22 * r22),
            )

            dR00 = dM00 * ss[0]
            dR01 = dM01 * ss[0]
            dR02 = dM02 * ss[0]
            dR10 = dM10 * ss[1]
            dR11 = dM11 * ss[1]
            dR12 = dM12 * ss[1]
            dR20 = dM20 * ss[2]
            dR21 = dM21 * ss[2]
            dR22 = dM22 * ss[2]

            grad_rotations_out[tid] = wp.vec4(
                2.0 * zq * (dR01 - dR10) + 2.0 * yq * (dR20 - dR02) + 2.0 * xq * (dR12 - dR21),
                2.0 * yq * (dR10 + dR01) + 2.0 * zq * (dR20 + dR02) + 2.0 * rr * (dR12 - dR21) - 4.0 * xq * (dR22 + dR11),
                2.0 * xq * (dR10 + dR01) + 2.0 * rr * (dR20 - dR02) + 2.0 * zq * (dR12 + dR21) - 4.0 * yq * (dR22 + dR00),
                2.0 * rr * (dR01 - dR10) + 2.0 * xq * (dR20 + dR02) + 2.0 * yq * (dR12 + dR21) - 4.0 * zq * (dR11 + dR00),
            )

        # ======== Part C: accumulate ========
        acc = grad_projected + grad_cov_means
        if has_sh != 0:
            acc = acc + grad_sh_means[tid]
        grad_means3D[tid] = acc
        rp = render_grad_points[tid]
        grad_means2D[tid] = wp.vec3(rp[0], rp[1], 0.0)

    # ---- F1: vec3-typed SH backward kernels for improved memory coalescing ----
    # M1+SoA: SH backward kernel with SoA-transposed layout for coalesced memory access.
    # Data arrives as (K, P, 3) → shs_v3[k * point_count + tid] gives coalesced vec3 reads.
    # Eliminates 86% excessive sectors from AoS stride (192 bytes between threads).
    @wp.kernel
    def _backward_rgb_from_sh_v3_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        campos_flat: wp.array(dtype=wp.float32),
        shs_v3: wp.array(dtype=wp.vec3),
        degree: wp.int32,
        coeff_count: wp.int32,
        point_count: wp.int32,
        grad_color_v3: wp.array(dtype=wp.vec3),
        clamped_flat: wp.array(dtype=wp.int32),
        grad_means: wp.array(dtype=wp.vec3),
        grad_sh_v3: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()

        dir_orig = means3d[tid] - wp.vec3(campos_flat[0], campos_flat[1], campos_flat[2])
        dir_len_sq = wp.dot(dir_orig, dir_orig)
        inv_dir_len = 1.0 / wp.sqrt(wp.max(dir_len_sq, 1.0e-20))
        direction = dir_orig * inv_dir_len
        x = direction[0]
        y = direction[1]
        z = direction[2]

        # M2: inline clamp masking (avoids _masked_grad allocation)
        _gc = grad_color_v3[tid]
        grad_rgb = wp.vec3(
            _gc[0] * (1.0 - wp.float32(clamped_flat[tid * 3])),
            _gc[1] * (1.0 - wp.float32(clamped_flat[tid * 3 + 1])),
            _gc[2] * (1.0 - wp.float32(clamped_flat[tid * 3 + 2])),
        )

        acc_dx = wp.vec3(0.0, 0.0, 0.0)
        acc_dy = wp.vec3(0.0, 0.0, 0.0)
        acc_dz = wp.vec3(0.0, 0.0, 0.0)

        # SoA stride: shs_v3 and grad_sh_v3 are (K, P, 3) layout
        # shs_v3[k * _P + tid] gives coalesced access (adjacent threads → adjacent vec3s)
        _P = point_count

        if coeff_count > 0:
            grad_sh_v3[tid] = sh_c0 * grad_rgb

        if degree > 0 and coeff_count > 3:
            grad_sh_v3[_P + tid] = (-sh_c1 * y) * grad_rgb
            grad_sh_v3[2 * _P + tid] = (sh_c1 * z) * grad_rgb
            grad_sh_v3[3 * _P + tid] = (-sh_c1 * x) * grad_rgb

            acc_dx = (-sh_c1) * shs_v3[3 * _P + tid]
            acc_dy = (-sh_c1) * shs_v3[_P + tid]
            acc_dz = sh_c1 * shs_v3[2 * _P + tid]

        if degree > 1 and coeff_count > 8:
            xx = x * x
            yy = y * y
            zz = z * z
            xy = x * y
            yz = y * z
            xz = x * z

            grad_sh_v3[4 * _P + tid] = (sh_c2_0 * xy) * grad_rgb
            grad_sh_v3[5 * _P + tid] = (sh_c2_1 * yz) * grad_rgb
            grad_sh_v3[6 * _P + tid] = (sh_c2_2 * (2.0 * zz - xx - yy)) * grad_rgb
            grad_sh_v3[7 * _P + tid] = (sh_c2_3 * xz) * grad_rgb
            grad_sh_v3[8 * _P + tid] = (sh_c2_4 * (xx - yy)) * grad_rgb

            s4 = shs_v3[4 * _P + tid]
            s5 = shs_v3[5 * _P + tid]
            s6 = shs_v3[6 * _P + tid]
            s7 = shs_v3[7 * _P + tid]
            s8 = shs_v3[8 * _P + tid]

            acc_dx = acc_dx + (sh_c2_0 * y) * s4 + (sh_c2_2 * (-2.0 * x)) * s6 + (sh_c2_3 * z) * s7 + (sh_c2_4 * (2.0 * x)) * s8
            acc_dy = acc_dy + (sh_c2_0 * x) * s4 + (sh_c2_1 * z) * s5 + (sh_c2_2 * (-2.0 * y)) * s6 + (sh_c2_4 * (-2.0 * y)) * s8
            acc_dz = acc_dz + (sh_c2_1 * y) * s5 + (sh_c2_2 * (4.0 * z)) * s6 + (sh_c2_3 * x) * s7

            if degree > 2 and coeff_count > 15:
                grad_sh_v3[9 * _P + tid] = (sh_c3_0 * y * (3.0 * xx - yy)) * grad_rgb
                grad_sh_v3[10 * _P + tid] = (sh_c3_1 * xy * z) * grad_rgb
                grad_sh_v3[11 * _P + tid] = (sh_c3_2 * y * (4.0 * zz - xx - yy)) * grad_rgb
                grad_sh_v3[12 * _P + tid] = (sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * grad_rgb
                grad_sh_v3[13 * _P + tid] = (sh_c3_4 * x * (4.0 * zz - xx - yy)) * grad_rgb
                grad_sh_v3[14 * _P + tid] = (sh_c3_5 * z * (xx - yy)) * grad_rgb
                grad_sh_v3[15 * _P + tid] = (sh_c3_6 * x * (xx - 3.0 * yy)) * grad_rgb

                s9 = shs_v3[9 * _P + tid]
                s10 = shs_v3[10 * _P + tid]
                s11 = shs_v3[11 * _P + tid]
                s12 = shs_v3[12 * _P + tid]
                s13 = shs_v3[13 * _P + tid]
                s14 = shs_v3[14 * _P + tid]
                s15 = shs_v3[15 * _P + tid]

                acc_dx = acc_dx + (sh_c3_0 * (6.0 * xy)) * s9 + (sh_c3_1 * yz) * s10 + (sh_c3_2 * (-2.0 * xy)) * s11 + (sh_c3_3 * (-6.0 * xz)) * s12 + (sh_c3_4 * (-3.0 * xx + 4.0 * zz - yy)) * s13 + (sh_c3_5 * (2.0 * xz)) * s14 + (sh_c3_6 * (3.0 * (xx - yy))) * s15
                acc_dy = acc_dy + (sh_c3_0 * (3.0 * (xx - yy))) * s9 + (sh_c3_1 * xz) * s10 + (sh_c3_2 * (-3.0 * yy + 4.0 * zz - xx)) * s11 + (sh_c3_3 * (-6.0 * yz)) * s12 + (sh_c3_4 * (-2.0 * xy)) * s13 + (sh_c3_5 * (-2.0 * yz)) * s14 + (sh_c3_6 * (-6.0 * xy)) * s15
                acc_dz = acc_dz + (sh_c3_1 * xy) * s10 + (sh_c3_2 * (8.0 * yz)) * s11 + (sh_c3_3 * (3.0 * (2.0 * zz - xx - yy))) * s12 + (sh_c3_4 * (8.0 * xz)) * s13 + (sh_c3_5 * (xx - yy)) * s14

        dL_ddir = wp.vec3(
            wp.dot(acc_dx, grad_rgb),
            wp.dot(acc_dy, grad_rgb),
            wp.dot(acc_dz, grad_rgb),
        )
        grad_means[tid] = _dnormvdv_wp(dir_orig, dL_ddir)


    @wp.func
    def _compute_power(con_o: wp.vec4, d_x: float, d_y: float):
        return -0.5 * (con_o[0] * d_x * d_x + con_o[2] * d_y * d_y) - con_o[1] * d_x * d_y


    @wp.func
    def _compute_alpha(con_o: wp.vec4, power: float):
        return wp.min(float(0.99), con_o[3] * wp.exp(power))


    @wp.func
    def _ndc_to_pix_wp(value: float, size: int):
        return ((value + 1.0) * float(size) - 1.0) * 0.5

    @wp.func
    def _dnormvdv_wp(vector: wp.vec3, grad_vector: wp.vec3):
        sum2 = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]
        invsum32 = 1.0 / wp.sqrt(wp.max(sum2 * sum2 * sum2, 1.0e-20))
        return wp.vec3(
            ((sum2 - vector[0] * vector[0]) * grad_vector[0] - vector[1] * vector[0] * grad_vector[1] - vector[2] * vector[0] * grad_vector[2]) * invsum32,
            (-vector[0] * vector[1] * grad_vector[0] + (sum2 - vector[1] * vector[1]) * grad_vector[1] - vector[2] * vector[1] * grad_vector[2]) * invsum32,
            (-vector[0] * vector[2] * grad_vector[0] - vector[1] * vector[2] * grad_vector[1] + (sum2 - vector[2] * vector[2]) * grad_vector[2]) * invsum32,
        )


    @wp.func
    def _conic_denom2inv_wp(denom: wp.float32):
        return 1.0 / (denom * denom + 1.0e-7)


    @wp.kernel
    def _backward_rgb_from_sh_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        campos_flat: wp.array(dtype=wp.float32),
        shs_flat: wp.array(dtype=wp.float32),
        degree: wp.int32,
        coeff_count: wp.int32,
        clamped_flat: wp.array(dtype=wp.int32),
        grad_color_flat: wp.array(dtype=wp.float32),
        grad_means: wp.array(dtype=wp.vec3),
        grad_sh_flat: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        sh_base = tid * coeff_count * NUM_CHANNELS
        color_base = tid * NUM_CHANNELS

        dir_orig = means3d[tid] - wp.vec3(campos_flat[0], campos_flat[1], campos_flat[2])
        dir_len_sq = dir_orig[0] * dir_orig[0] + dir_orig[1] * dir_orig[1] + dir_orig[2] * dir_orig[2]
        inv_dir_len = 1.0 / wp.sqrt(wp.max(dir_len_sq, 1.0e-20))
        direction = dir_orig * inv_dir_len
        x = direction[0]
        y = direction[1]
        z = direction[2]

        grad_rgb0 = grad_color_flat[color_base + 0] * float(1 - clamped_flat[color_base + 0])
        grad_rgb1 = grad_color_flat[color_base + 1] * float(1 - clamped_flat[color_base + 1])
        grad_rgb2 = grad_color_flat[color_base + 2] * float(1 - clamped_flat[color_base + 2])

        dRGBdx0 = float(0.0)
        dRGBdx1 = float(0.0)
        dRGBdx2 = float(0.0)
        dRGBdy0 = float(0.0)
        dRGBdy1 = float(0.0)
        dRGBdy2 = float(0.0)
        dRGBdz0 = float(0.0)
        dRGBdz1 = float(0.0)
        dRGBdz2 = float(0.0)

        if coeff_count > 0:
            grad_sh_flat[sh_base + 0] = sh_c0 * grad_rgb0
            grad_sh_flat[sh_base + 1] = sh_c0 * grad_rgb1
            grad_sh_flat[sh_base + 2] = sh_c0 * grad_rgb2

        if degree > 0 and coeff_count > 3:
            grad_sh_flat[sh_base + 3] = -sh_c1 * y * grad_rgb0
            grad_sh_flat[sh_base + 4] = -sh_c1 * y * grad_rgb1
            grad_sh_flat[sh_base + 5] = -sh_c1 * y * grad_rgb2
            grad_sh_flat[sh_base + 6] = sh_c1 * z * grad_rgb0
            grad_sh_flat[sh_base + 7] = sh_c1 * z * grad_rgb1
            grad_sh_flat[sh_base + 8] = sh_c1 * z * grad_rgb2
            grad_sh_flat[sh_base + 9] = -sh_c1 * x * grad_rgb0
            grad_sh_flat[sh_base + 10] = -sh_c1 * x * grad_rgb1
            grad_sh_flat[sh_base + 11] = -sh_c1 * x * grad_rgb2

            dRGBdx0 = -sh_c1 * shs_flat[sh_base + 9]
            dRGBdx1 = -sh_c1 * shs_flat[sh_base + 10]
            dRGBdx2 = -sh_c1 * shs_flat[sh_base + 11]
            dRGBdy0 = -sh_c1 * shs_flat[sh_base + 3]
            dRGBdy1 = -sh_c1 * shs_flat[sh_base + 4]
            dRGBdy2 = -sh_c1 * shs_flat[sh_base + 5]
            dRGBdz0 = sh_c1 * shs_flat[sh_base + 6]
            dRGBdz1 = sh_c1 * shs_flat[sh_base + 7]
            dRGBdz2 = sh_c1 * shs_flat[sh_base + 8]

            if degree > 1 and coeff_count > 8:
                xx = x * x
                yy = y * y
                zz = z * z
                xy = x * y
                yz = y * z
                xz = x * z

                grad_sh_flat[sh_base + 12] = sh_c2_0 * xy * grad_rgb0
                grad_sh_flat[sh_base + 13] = sh_c2_0 * xy * grad_rgb1
                grad_sh_flat[sh_base + 14] = sh_c2_0 * xy * grad_rgb2
                grad_sh_flat[sh_base + 15] = sh_c2_1 * yz * grad_rgb0
                grad_sh_flat[sh_base + 16] = sh_c2_1 * yz * grad_rgb1
                grad_sh_flat[sh_base + 17] = sh_c2_1 * yz * grad_rgb2
                grad_sh_flat[sh_base + 18] = sh_c2_2 * (2.0 * zz - xx - yy) * grad_rgb0
                grad_sh_flat[sh_base + 19] = sh_c2_2 * (2.0 * zz - xx - yy) * grad_rgb1
                grad_sh_flat[sh_base + 20] = sh_c2_2 * (2.0 * zz - xx - yy) * grad_rgb2
                grad_sh_flat[sh_base + 21] = sh_c2_3 * xz * grad_rgb0
                grad_sh_flat[sh_base + 22] = sh_c2_3 * xz * grad_rgb1
                grad_sh_flat[sh_base + 23] = sh_c2_3 * xz * grad_rgb2
                grad_sh_flat[sh_base + 24] = sh_c2_4 * (xx - yy) * grad_rgb0
                grad_sh_flat[sh_base + 25] = sh_c2_4 * (xx - yy) * grad_rgb1
                grad_sh_flat[sh_base + 26] = sh_c2_4 * (xx - yy) * grad_rgb2

                dRGBdx0 = dRGBdx0 + sh_c2_0 * y * shs_flat[sh_base + 12] + sh_c2_2 * (-2.0 * x) * shs_flat[sh_base + 18] + sh_c2_3 * z * shs_flat[sh_base + 21] + sh_c2_4 * (2.0 * x) * shs_flat[sh_base + 24]
                dRGBdx1 = dRGBdx1 + sh_c2_0 * y * shs_flat[sh_base + 13] + sh_c2_2 * (-2.0 * x) * shs_flat[sh_base + 19] + sh_c2_3 * z * shs_flat[sh_base + 22] + sh_c2_4 * (2.0 * x) * shs_flat[sh_base + 25]
                dRGBdx2 = dRGBdx2 + sh_c2_0 * y * shs_flat[sh_base + 14] + sh_c2_2 * (-2.0 * x) * shs_flat[sh_base + 20] + sh_c2_3 * z * shs_flat[sh_base + 23] + sh_c2_4 * (2.0 * x) * shs_flat[sh_base + 26]
                dRGBdy0 = dRGBdy0 + sh_c2_0 * x * shs_flat[sh_base + 12] + sh_c2_1 * z * shs_flat[sh_base + 15] + sh_c2_2 * (-2.0 * y) * shs_flat[sh_base + 18] + sh_c2_4 * (-2.0 * y) * shs_flat[sh_base + 24]
                dRGBdy1 = dRGBdy1 + sh_c2_0 * x * shs_flat[sh_base + 13] + sh_c2_1 * z * shs_flat[sh_base + 16] + sh_c2_2 * (-2.0 * y) * shs_flat[sh_base + 19] + sh_c2_4 * (-2.0 * y) * shs_flat[sh_base + 25]
                dRGBdy2 = dRGBdy2 + sh_c2_0 * x * shs_flat[sh_base + 14] + sh_c2_1 * z * shs_flat[sh_base + 17] + sh_c2_2 * (-2.0 * y) * shs_flat[sh_base + 20] + sh_c2_4 * (-2.0 * y) * shs_flat[sh_base + 26]
                dRGBdz0 = dRGBdz0 + sh_c2_1 * y * shs_flat[sh_base + 15] + sh_c2_2 * (4.0 * z) * shs_flat[sh_base + 18] + sh_c2_3 * x * shs_flat[sh_base + 21]
                dRGBdz1 = dRGBdz1 + sh_c2_1 * y * shs_flat[sh_base + 16] + sh_c2_2 * (4.0 * z) * shs_flat[sh_base + 19] + sh_c2_3 * x * shs_flat[sh_base + 22]
                dRGBdz2 = dRGBdz2 + sh_c2_1 * y * shs_flat[sh_base + 17] + sh_c2_2 * (4.0 * z) * shs_flat[sh_base + 20] + sh_c2_3 * x * shs_flat[sh_base + 23]

                if degree > 2 and coeff_count > 15:
                    grad_sh_flat[sh_base + 27] = sh_c3_0 * y * (3.0 * xx - yy) * grad_rgb0
                    grad_sh_flat[sh_base + 28] = sh_c3_0 * y * (3.0 * xx - yy) * grad_rgb1
                    grad_sh_flat[sh_base + 29] = sh_c3_0 * y * (3.0 * xx - yy) * grad_rgb2
                    grad_sh_flat[sh_base + 30] = sh_c3_1 * xy * z * grad_rgb0
                    grad_sh_flat[sh_base + 31] = sh_c3_1 * xy * z * grad_rgb1
                    grad_sh_flat[sh_base + 32] = sh_c3_1 * xy * z * grad_rgb2
                    grad_sh_flat[sh_base + 33] = sh_c3_2 * y * (4.0 * zz - xx - yy) * grad_rgb0
                    grad_sh_flat[sh_base + 34] = sh_c3_2 * y * (4.0 * zz - xx - yy) * grad_rgb1
                    grad_sh_flat[sh_base + 35] = sh_c3_2 * y * (4.0 * zz - xx - yy) * grad_rgb2
                    grad_sh_flat[sh_base + 36] = sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * grad_rgb0
                    grad_sh_flat[sh_base + 37] = sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * grad_rgb1
                    grad_sh_flat[sh_base + 38] = sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * grad_rgb2
                    grad_sh_flat[sh_base + 39] = sh_c3_4 * x * (4.0 * zz - xx - yy) * grad_rgb0
                    grad_sh_flat[sh_base + 40] = sh_c3_4 * x * (4.0 * zz - xx - yy) * grad_rgb1
                    grad_sh_flat[sh_base + 41] = sh_c3_4 * x * (4.0 * zz - xx - yy) * grad_rgb2
                    grad_sh_flat[sh_base + 42] = sh_c3_5 * z * (xx - yy) * grad_rgb0
                    grad_sh_flat[sh_base + 43] = sh_c3_5 * z * (xx - yy) * grad_rgb1
                    grad_sh_flat[sh_base + 44] = sh_c3_5 * z * (xx - yy) * grad_rgb2
                    grad_sh_flat[sh_base + 45] = sh_c3_6 * x * (xx - 3.0 * yy) * grad_rgb0
                    grad_sh_flat[sh_base + 46] = sh_c3_6 * x * (xx - 3.0 * yy) * grad_rgb1
                    grad_sh_flat[sh_base + 47] = sh_c3_6 * x * (xx - 3.0 * yy) * grad_rgb2

                    dRGBdx0 = dRGBdx0 + sh_c3_0 * (6.0 * xy) * shs_flat[sh_base + 27] + sh_c3_1 * yz * shs_flat[sh_base + 30] + sh_c3_2 * (-2.0 * xy) * shs_flat[sh_base + 33] + sh_c3_3 * (-6.0 * xz) * shs_flat[sh_base + 36] + sh_c3_4 * (-3.0 * xx + 4.0 * zz - yy) * shs_flat[sh_base + 39] + sh_c3_5 * (2.0 * xz) * shs_flat[sh_base + 42] + sh_c3_6 * (3.0 * (xx - yy)) * shs_flat[sh_base + 45]
                    dRGBdx1 = dRGBdx1 + sh_c3_0 * (6.0 * xy) * shs_flat[sh_base + 28] + sh_c3_1 * yz * shs_flat[sh_base + 31] + sh_c3_2 * (-2.0 * xy) * shs_flat[sh_base + 34] + sh_c3_3 * (-6.0 * xz) * shs_flat[sh_base + 37] + sh_c3_4 * (-3.0 * xx + 4.0 * zz - yy) * shs_flat[sh_base + 40] + sh_c3_5 * (2.0 * xz) * shs_flat[sh_base + 43] + sh_c3_6 * (3.0 * (xx - yy)) * shs_flat[sh_base + 46]
                    dRGBdx2 = dRGBdx2 + sh_c3_0 * (6.0 * xy) * shs_flat[sh_base + 29] + sh_c3_1 * yz * shs_flat[sh_base + 32] + sh_c3_2 * (-2.0 * xy) * shs_flat[sh_base + 35] + sh_c3_3 * (-6.0 * xz) * shs_flat[sh_base + 38] + sh_c3_4 * (-3.0 * xx + 4.0 * zz - yy) * shs_flat[sh_base + 41] + sh_c3_5 * (2.0 * xz) * shs_flat[sh_base + 44] + sh_c3_6 * (3.0 * (xx - yy)) * shs_flat[sh_base + 47]
                    dRGBdy0 = dRGBdy0 + sh_c3_0 * (3.0 * (xx - yy)) * shs_flat[sh_base + 27] + sh_c3_1 * xz * shs_flat[sh_base + 30] + sh_c3_2 * (-3.0 * yy + 4.0 * zz - xx) * shs_flat[sh_base + 33] + sh_c3_3 * (-6.0 * yz) * shs_flat[sh_base + 36] + sh_c3_4 * (-2.0 * xy) * shs_flat[sh_base + 39] + sh_c3_5 * (-2.0 * yz) * shs_flat[sh_base + 42] + sh_c3_6 * (-6.0 * xy) * shs_flat[sh_base + 45]
                    dRGBdy1 = dRGBdy1 + sh_c3_0 * (3.0 * (xx - yy)) * shs_flat[sh_base + 28] + sh_c3_1 * xz * shs_flat[sh_base + 31] + sh_c3_2 * (-3.0 * yy + 4.0 * zz - xx) * shs_flat[sh_base + 34] + sh_c3_3 * (-6.0 * yz) * shs_flat[sh_base + 37] + sh_c3_4 * (-2.0 * xy) * shs_flat[sh_base + 40] + sh_c3_5 * (-2.0 * yz) * shs_flat[sh_base + 43] + sh_c3_6 * (-6.0 * xy) * shs_flat[sh_base + 46]
                    dRGBdy2 = dRGBdy2 + sh_c3_0 * (3.0 * (xx - yy)) * shs_flat[sh_base + 29] + sh_c3_1 * xz * shs_flat[sh_base + 32] + sh_c3_2 * (-3.0 * yy + 4.0 * zz - xx) * shs_flat[sh_base + 35] + sh_c3_3 * (-6.0 * yz) * shs_flat[sh_base + 38] + sh_c3_4 * (-2.0 * xy) * shs_flat[sh_base + 41] + sh_c3_5 * (-2.0 * yz) * shs_flat[sh_base + 44] + sh_c3_6 * (-6.0 * xy) * shs_flat[sh_base + 47]
                    dRGBdz0 = dRGBdz0 + sh_c3_1 * xy * shs_flat[sh_base + 30] + sh_c3_2 * (8.0 * yz) * shs_flat[sh_base + 33] + sh_c3_3 * (3.0 * (2.0 * zz - xx - yy)) * shs_flat[sh_base + 36] + sh_c3_4 * (8.0 * xz) * shs_flat[sh_base + 39] + sh_c3_5 * (xx - yy) * shs_flat[sh_base + 42]
                    dRGBdz1 = dRGBdz1 + sh_c3_1 * xy * shs_flat[sh_base + 31] + sh_c3_2 * (8.0 * yz) * shs_flat[sh_base + 34] + sh_c3_3 * (3.0 * (2.0 * zz - xx - yy)) * shs_flat[sh_base + 37] + sh_c3_4 * (8.0 * xz) * shs_flat[sh_base + 40] + sh_c3_5 * (xx - yy) * shs_flat[sh_base + 43]
                    dRGBdz2 = dRGBdz2 + sh_c3_1 * xy * shs_flat[sh_base + 32] + sh_c3_2 * (8.0 * yz) * shs_flat[sh_base + 35] + sh_c3_3 * (3.0 * (2.0 * zz - xx - yy)) * shs_flat[sh_base + 38] + sh_c3_4 * (8.0 * xz) * shs_flat[sh_base + 41] + sh_c3_5 * (xx - yy) * shs_flat[sh_base + 44]

        dL_ddir = wp.vec3(
            dRGBdx0 * grad_rgb0 + dRGBdx1 * grad_rgb1 + dRGBdx2 * grad_rgb2,
            dRGBdy0 * grad_rgb0 + dRGBdy1 * grad_rgb1 + dRGBdy2 * grad_rgb2,
            dRGBdz0 * grad_rgb0 + dRGBdz1 * grad_rgb1 + dRGBdz2 * grad_rgb2,
        )
        grad_means[tid] = _dnormvdv_wp(dir_orig, dL_ddir)


    # P2b: vec3 forward SH — 3× fewer load/store instructions vs scalar version
    @wp.kernel
    def _forward_rgb_from_sh_v3_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        campos_flat: wp.array(dtype=wp.float32),
        shs_v3: wp.array(dtype=wp.vec3),
        degree: wp.int32,
        coeff_count: wp.int32,
        rgb_v3: wp.array(dtype=wp.vec3),
        clamped_flat: wp.array(dtype=wp.int32),
    ):
        tid = wp.tid()
        sh_base = tid * coeff_count

        dir_orig = means3d[tid] - wp.vec3(campos_flat[0], campos_flat[1], campos_flat[2])
        dir_len_sq = dir_orig[0] * dir_orig[0] + dir_orig[1] * dir_orig[1] + dir_orig[2] * dir_orig[2]
        inv_dir_len = 1.0 / wp.sqrt(wp.max(dir_len_sq, 1.0e-20))
        direction = dir_orig * inv_dir_len
        x = direction[0]
        y = direction[1]
        z = direction[2]

        color = sh_c0 * shs_v3[sh_base]

        if degree > 0 and coeff_count > 3:
            color = color + (-sh_c1 * y) * shs_v3[sh_base + 1]
            color = color + (sh_c1 * z) * shs_v3[sh_base + 2]
            color = color + (-sh_c1 * x) * shs_v3[sh_base + 3]

            if degree > 1 and coeff_count > 8:
                xx = x * x
                yy = y * y
                zz = z * z
                xy = x * y
                yz = y * z
                xz = x * z

                color = color + (sh_c2_0 * xy) * shs_v3[sh_base + 4]
                color = color + (sh_c2_1 * yz) * shs_v3[sh_base + 5]
                color = color + (sh_c2_2 * (2.0 * zz - xx - yy)) * shs_v3[sh_base + 6]
                color = color + (sh_c2_3 * xz) * shs_v3[sh_base + 7]
                color = color + (sh_c2_4 * (xx - yy)) * shs_v3[sh_base + 8]

                if degree > 2 and coeff_count > 15:
                    color = color + (sh_c3_0 * y * (3.0 * xx - yy)) * shs_v3[sh_base + 9]
                    color = color + (sh_c3_1 * xy * z) * shs_v3[sh_base + 10]
                    color = color + (sh_c3_2 * y * (4.0 * zz - xx - yy)) * shs_v3[sh_base + 11]
                    color = color + (sh_c3_3 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy)) * shs_v3[sh_base + 12]
                    color = color + (sh_c3_4 * x * (4.0 * zz - xx - yy)) * shs_v3[sh_base + 13]
                    color = color + (sh_c3_5 * z * (xx - yy)) * shs_v3[sh_base + 14]
                    color = color + (sh_c3_6 * x * (xx - 3.0 * yy)) * shs_v3[sh_base + 15]

        color = color + wp.vec3(0.5, 0.5, 0.5)
        c0 = color[0]
        c1 = color[1]
        c2 = color[2]
        clamp0 = int(0)
        clamp1 = int(0)
        clamp2 = int(0)
        if c0 < 0.0:
            c0 = 0.0
            clamp0 = int(1)
        if c1 < 0.0:
            c1 = 0.0
            clamp1 = int(1)
        if c2 < 0.0:
            c2 = 0.0
            clamp2 = int(1)

        rgb_v3[tid] = wp.vec3(c0, c1, c2)
        clamped_flat[tid * 3 + 0] = clamp0
        clamped_flat[tid * 3 + 1] = clamp1
        clamped_flat[tid * 3 + 2] = clamp2


    @wp.kernel
    def _cov2d_preprocess_masked_pack_warp_kernel(
        visible_mask: wp.array(dtype=wp.int32),
        means3d: wp.array(dtype=wp.vec3),
        cov3d_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        p_proj: wp.array(dtype=wp.vec3),
        p_view_z: wp.array(dtype=wp.float32),
        opacities: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        depths: wp.array(dtype=wp.float32),
        radii: wp.array(dtype=wp.int32),
        proj_2d: wp.array(dtype=wp.vec2),
        conic_2d: wp.array(dtype=wp.vec3),
        conic_2d_inv: wp.array(dtype=wp.vec3),
        points_xy_image: wp.array(dtype=wp.vec2),
        tiles_touched: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
    ):
        idx = wp.tid()
        depths[idx] = float(0.0)
        radii[idx] = int(0)
        proj_2d[idx] = wp.vec2(0.0, 0.0)
        conic_2d[idx] = wp.vec3(0.0, 0.0, 0.0)
        conic_2d_inv[idx] = wp.vec3(0.0, 0.0, 0.0)
        points_xy_image[idx] = wp.vec2(0.0, 0.0)
        tiles_touched[idx] = int(0)
        conic_opacity[idx] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        if visible_mask[idx] == 0:
            return

        p = means3d[idx]
        base = idx * 6

        x = view_flat[0] * p[0] + view_flat[4] * p[1] + view_flat[8] * p[2] + view_flat[12]
        y = view_flat[1] * p[0] + view_flat[5] * p[1] + view_flat[9] * p[2] + view_flat[13]
        z = view_flat[2] * p[0] + view_flat[6] * p[1] + view_flat[10] * p[2] + view_flat[14]

        limx = 1.3 * tanfovx
        limy = 1.3 * tanfovy
        txtz = x / z
        tytz = y / z
        x = wp.clamp(txtz, -limx, limx) * z
        y = wp.clamp(tytz, -limy, limy) * z

        a = focal_x / z
        b = focal_y / z
        c = -(focal_x * x) / (z * z)
        d = -(focal_y * y) / (z * z)

        t00 = view_flat[0] * a + view_flat[2] * c
        t10 = view_flat[4] * a + view_flat[6] * c
        t20 = view_flat[8] * a + view_flat[10] * c
        t01 = view_flat[1] * b + view_flat[2] * d
        t11 = view_flat[5] * b + view_flat[6] * d
        t21 = view_flat[9] * b + view_flat[10] * d

        v00 = cov3d_flat[base + 0]
        v01 = cov3d_flat[base + 1]
        v02 = cov3d_flat[base + 2]
        v11 = cov3d_flat[base + 3]
        v12 = cov3d_flat[base + 4]
        v22 = cov3d_flat[base + 5]

        vt0x = v00 * t00 + v01 * t10 + v02 * t20
        vt0y = v01 * t00 + v11 * t10 + v12 * t20
        vt0z = v02 * t00 + v12 * t10 + v22 * t20
        vt1x = v00 * t01 + v01 * t11 + v02 * t21
        vt1y = v01 * t01 + v11 * t11 + v12 * t21
        vt1z = v02 * t01 + v12 * t11 + v22 * t21

        cov_value = wp.vec3(
            t00 * vt0x + t10 * vt0y + t20 * vt0z + 0.3,
            t00 * vt1x + t10 * vt1y + t20 * vt1z,
            t01 * vt1x + t11 * vt1y + t21 * vt1z + 0.3,
        )
        det = cov_value[0] * cov_value[2] - cov_value[1] * cov_value[1]
        if wp.abs(det) < DET_EPSILON:
            return

        det_inv = 1.0 / det
        conic = wp.vec3(cov_value[2] * det_inv, -cov_value[1] * det_inv, cov_value[0] * det_inv)

        mid = 0.5 * (cov_value[0] + cov_value[2])
        root = wp.sqrt(wp.max(0.1, mid * mid - det))
        lambda1 = mid + root
        lambda2 = mid - root
        radius = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(lambda1, lambda2))))

        proj_value = p_proj[idx]
        point_x = _ndc_to_pix_wp(proj_value[0], image_width)
        point_y = _ndc_to_pix_wp(proj_value[1], image_height)

        rect = _compute_tile_rect_snugbox_cov2d_wp(point_x, point_y, cov_value[0], cov_value[2], opacities[idx], grid_x, grid_y)
        rect_area = ( rect[2] - rect[0] ) * ( rect[3] - rect[1] )
        if rect_area == 0:
            return

        

        point_image = wp.vec2(point_x, point_y)
        depths[idx] = p_view_z[idx]
        radii[idx] = radius
        proj_2d[idx] = point_image
        conic_2d[idx] = conic
        conic_2d_inv[idx] = cov_value
        points_xy_image[idx] = point_image
        tiles_touched[idx] = rect_area
        conic_opacity[idx] = wp.vec4(conic[0], conic[1], conic[2], opacities[idx])
        
        
    @wp.kernel
    def _cov2d_preprocess_masked_pack_scale_rotation_warp_kernel(
        visible_mask: wp.array(dtype=wp.int32),
        means3d: wp.array(dtype=wp.vec3),
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        view_flat: wp.array(dtype=wp.float32),
        p_proj: wp.array(dtype=wp.vec3),
        p_view_z: wp.array(dtype=wp.float32),
        opacities: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        depths: wp.array(dtype=wp.float32),
        radii: wp.array(dtype=wp.int32),
        proj_2d: wp.array(dtype=wp.vec2),
        conic_2d: wp.array(dtype=wp.vec3),
        conic_2d_inv: wp.array(dtype=wp.vec3),
        points_xy_image: wp.array(dtype=wp.vec2),
        tiles_touched: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
    ):
        idx = wp.tid()
        depths[idx] = float(0.0)
        radii[idx] = int(0)
        proj_2d[idx] = wp.vec2(0.0, 0.0)
        conic_2d[idx] = wp.vec3(0.0, 0.0, 0.0)
        conic_2d_inv[idx] = wp.vec3(0.0, 0.0, 0.0)
        points_xy_image[idx] = wp.vec2(0.0, 0.0)
        tiles_touched[idx] = int(0)
        conic_opacity[idx] = wp.vec4(0.0, 0.0, 0.0, 0.0)
        if visible_mask[idx] == 0:
            return

        p = means3d[idx]

        x = view_flat[0] * p[0] + view_flat[4] * p[1] + view_flat[8] * p[2] + view_flat[12]
        y = view_flat[1] * p[0] + view_flat[5] * p[1] + view_flat[9] * p[2] + view_flat[13]
        z = view_flat[2] * p[0] + view_flat[6] * p[1] + view_flat[10] * p[2] + view_flat[14]
        inv_z = 1.0 / (z + 1.0e-7)
        limx = 1.3 * tanfovx
        limy = 1.3 * tanfovy
        x = wp.clamp(x * inv_z, -limx, limx)
        y = wp.clamp(y * inv_z, -limy, limy)

        a = focal_x * inv_z
        b = focal_y * inv_z

        t00 = a * ( view_flat[0] - view_flat[2] * x )
        t10 = a * ( view_flat[4] - view_flat[6] * x )
        t20 = a * ( view_flat[8] - view_flat[10] * x )
        t01 = b * ( view_flat[1] - view_flat[2] * y )
        t11 = b * ( view_flat[5] - view_flat[6] * y )
        t21 = b * ( view_flat[9] - view_flat[10] * y )

        cov_value = _cov2d_from_scale_rotation_gram_wp(
            scales[idx],
            rotations[idx],
            scale_modifier,
            t00,
            t10,
            t20,
            t01,
            t11,
            t21,
        )
        det = cov_value[0] * cov_value[2] - cov_value[1] * cov_value[1]
        if wp.abs(det) < DET_EPSILON:
            return

        det_inv = 1.0 / det
        conic = wp.vec3(cov_value[2] * det_inv, -cov_value[1] * det_inv, cov_value[0] * det_inv)

        mid = 0.5 * (cov_value[0] + cov_value[2])
        root = wp.sqrt(wp.max(0.1, mid * mid - det))
        lambda1 = mid + root
        lambda2 = mid - root
        radius = wp.int32(wp.ceil(3.0 * wp.sqrt(wp.max(lambda1, lambda2))))

        proj_value = p_proj[idx]
        point_x = _ndc_to_pix_wp(proj_value[0], image_width)
        point_y = _ndc_to_pix_wp(proj_value[1], image_height)

        rect = _compute_tile_rect_snugbox_cov2d_wp(point_x, point_y, cov_value[0], cov_value[2], opacities[idx], grid_x, grid_y)
        rect_area = ( rect[2] - rect[0] ) * ( rect[3] - rect[1] )
        if rect_area == 0:
            return

        point_image = wp.vec2(point_x, point_y)
        depths[idx] = p_view_z[idx]
        radii[idx] = radius
        proj_2d[idx] = point_image
        conic_2d[idx] = conic
        conic_2d_inv[idx] = cov_value
        points_xy_image[idx] = point_image
        tiles_touched[idx] = rect_area
        conic_opacity[idx] = wp.vec4(conic[0], conic[1], conic[2], opacities[idx])
        
        
    @wp.kernel
    def _project_visible_points_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        view_flat: wp.array(dtype=wp.float32),
        proj_flat: wp.array(dtype=wp.float32),
        visible_mask: wp.array(dtype=wp.int32),
        p_proj_out: wp.array(dtype=wp.vec3),
        p_view_z_out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        point = means3d[tid]

        p_view_z = view_flat[2] * point[0] + view_flat[6] * point[1] + view_flat[10] * point[2] + view_flat[14]
        p_view_z_out[tid] = p_view_z
        visible_mask[tid] = int(p_view_z > 0.2)

        hom_x = proj_flat[0] * point[0] + proj_flat[4] * point[1] + proj_flat[8] * point[2] + proj_flat[12]
        hom_y = proj_flat[1] * point[0] + proj_flat[5] * point[1] + proj_flat[9] * point[2] + proj_flat[13]
        hom_z = proj_flat[2] * point[0] + proj_flat[6] * point[1] + proj_flat[10] * point[2] + proj_flat[14]
        hom_w = proj_flat[3] * point[0] + proj_flat[7] * point[1] + proj_flat[11] * point[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        p_proj_out[tid] = wp.vec3(hom_x * inv_w, hom_y * inv_w, hom_z * inv_w)


    @wp.func
    def _conservative_cull_radius_from_cov3d_wp(
        v00: wp.float32,
        v01: wp.float32,
        v02: wp.float32,
        v11: wp.float32,
        v12: wp.float32,
        v22: wp.float32,
    ):
        row0 = wp.abs(v00) + wp.abs(v01) + wp.abs(v02)
        row1 = wp.abs(v01) + wp.abs(v11) + wp.abs(v12)
        row2 = wp.abs(v02) + wp.abs(v12) + wp.abs(v22)
        return wp.max(row0, wp.max(row1, row2))


    @wp.func
    def _preprocess_radius_upper_wp(
        cov3d_lambda_upper: wp.float32,
        p_view_z: wp.float32,
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
    ):
        limx = PREPROCESS_CULL_FOV_SCALE * tanfovx
        limy = PREPROCESS_CULL_FOV_SCALE * tanfovy
        focal_x = float(image_width) / (2.0 * tanfovx)
        focal_y = float(image_height) / (2.0 * tanfovy)
        row0_norm = (focal_x / p_view_z) * wp.sqrt(1.0 + limx * limx)
        row1_norm = (focal_y / p_view_z) * wp.sqrt(1.0 + limy * limy)
        j_frob_sq = row0_norm * row0_norm + row1_norm * row1_norm
        cov2d_lambda_upper = cov3d_lambda_upper * j_frob_sq + 0.3
        return wp.int32(wp.ceil(PREPROCESS_CULL_SIGMA * wp.sqrt(wp.max(cov2d_lambda_upper, 0.0))))


    @wp.func
    def _compute_tile_rect_wp(
        point_x: wp.float32,
        point_y: wp.float32,
        radius: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
    ):
        rect_min_x = wp.min(grid_x, wp.max(0, wp.int32((point_x - float(radius)) / float(BLOCK_X))))
        rect_min_y = wp.min(grid_y, wp.max(0, wp.int32((point_y - float(radius)) / float(BLOCK_Y))))
        rect_max_x = wp.min(grid_x, wp.max(0, wp.int32((point_x + float(radius) + float(BLOCK_X - 1)) / float(BLOCK_X))))
        rect_max_y = wp.min(grid_y, wp.max(0, wp.int32((point_y + float(radius) + float(BLOCK_Y - 1)) / float(BLOCK_Y))))
        return wp.vec4i(rect_min_x, rect_min_y, rect_max_x, rect_max_y)


    @wp.func
    def _compute_tile_rect_snugbox_cov2d_wp(
        point_x: wp.float32,
        point_y: wp.float32,
        cov2d_aa: wp.float32,
        cov2d_cc: wp.float32,
        opacity: wp.float32,
        grid_x: wp.int32,
        grid_y: wp.int32,
    ):
        """Opacity-aware SnugBox directly from cov2d diagonal (b-insensitive).

        Mathematically equivalent to the conic version:
          r_x = ceil(sqrt(t * con_c / det_conic)) = ceil(sqrt(t * cov2d_aa))
        but avoids the det_conic = con_a*con_c - con_b^2 denominator that
        amplifies floating-point differences in the off-diagonal term b.
        """
        t = 2.0 * wp.log(wp.max(255.0 * opacity, 1.0))
        if t <= 0.0:
            return wp.vec4i(0, 0, 0, 0)
        radius_x = wp.int32(wp.ceil(wp.sqrt(wp.max(t * cov2d_aa, 0.0))))
        radius_y = wp.int32(wp.ceil(wp.sqrt(wp.max(t * cov2d_cc, 0.0))))
        rect_min_x = wp.min(grid_x, wp.max(0, wp.int32((point_x - float(radius_x)) / float(BLOCK_X))))
        rect_min_y = wp.min(grid_y, wp.max(0, wp.int32((point_y - float(radius_y)) / float(BLOCK_Y))))
        rect_max_x = wp.min(grid_x, wp.max(0, wp.int32((point_x + float(radius_x) + float(BLOCK_X - 1)) / float(BLOCK_X))))
        rect_max_y = wp.min(grid_y, wp.max(0, wp.int32((point_y + float(radius_y) + float(BLOCK_Y - 1)) / float(BLOCK_Y))))
        return wp.vec4i(rect_min_x, rect_min_y, rect_max_x, rect_max_y)

    # A2: AccuTile — per-row ellipse intersection for exact tile-x ranges.
    # Solves  con_a*dx^2 + 2*con_b*dx*dy + con_c*dy^2 = t  for dx at fixed dy
    # to narrow the tile-x loop per tile-y row in duplicate kernels.
    @wp.func
    def _accutile_row_x_range_wp(
        point_x: wp.float32,
        con_a: wp.float32,
        con_b: wp.float32,
        det_conic: wp.float32,
        t: wp.float32,
        dy: wp.float32,
        grid_x: wp.int32,
    ):
        inner = con_a * t - det_conic * dy * dy
        if inner <= 0.0:
            return wp.vec2i(0, 0)
        sqrt_inner = wp.sqrt(inner)
        dx_center = -con_b * dy / con_a
        dx_half = sqrt_inner / con_a
        x_min = point_x + dx_center - dx_half
        x_max = point_x + dx_center + dx_half
        tile_x_min = wp.min(grid_x, wp.max(0, wp.int32(x_min / float(BLOCK_X))))
        tile_x_max = wp.min(grid_x, wp.max(0, wp.int32((x_max + float(BLOCK_X - 1)) / float(BLOCK_X))))
        return wp.vec2i(tile_x_min, tile_x_max)

    @wp.func
    def _preprocess_rect_visible_wp(
        proj_value: wp.vec3,
        radius_upper: wp.int32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
    ):
        point_x = _ndc_to_pix_wp(proj_value[0], image_width)
        point_y = _ndc_to_pix_wp(proj_value[1], image_height)

        rect = _compute_tile_rect_wp(point_x, point_y, radius_upper, grid_x, grid_y)
        rect_area = ( rect[2] - rect[0] ) * ( rect[3] - rect[1] )
        return rect_area != 0


    @wp.kernel
    def _project_preprocess_visible_points_cov_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        cov3d_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        proj_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        visible_mask: wp.array(dtype=wp.int32),
        p_proj_out: wp.array(dtype=wp.vec3),
        p_view_z_out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        point = means3d[tid]
        base = tid * 6

        p_view_z = view_flat[2] * point[0] + view_flat[6] * point[1] + view_flat[10] * point[2] + view_flat[14]
        p_view_z_out[tid] = p_view_z
        if p_view_z <= VISIBILITY_NEAR_PLANE:
            visible_mask[tid] = int(0)
            return

        hom_x = proj_flat[0] * point[0] + proj_flat[4] * point[1] + proj_flat[8] * point[2] + proj_flat[12]
        hom_y = proj_flat[1] * point[0] + proj_flat[5] * point[1] + proj_flat[9] * point[2] + proj_flat[13]
        hom_z = proj_flat[2] * point[0] + proj_flat[6] * point[1] + proj_flat[10] * point[2] + proj_flat[14]
        hom_w = proj_flat[3] * point[0] + proj_flat[7] * point[1] + proj_flat[11] * point[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        proj_value = wp.vec3(hom_x * inv_w, hom_y * inv_w, hom_z * inv_w)

        cov3d_lambda_upper = _conservative_cull_radius_from_cov3d_wp(
            cov3d_flat[base + 0],
            cov3d_flat[base + 1],
            cov3d_flat[base + 2],
            cov3d_flat[base + 3],
            cov3d_flat[base + 4],
            cov3d_flat[base + 5],
        )
        radius_upper = _preprocess_radius_upper_wp(cov3d_lambda_upper, p_view_z, tanfovx, tanfovy, image_width, image_height)
        if not _preprocess_rect_visible_wp(proj_value, radius_upper, image_width, image_height, grid_x, grid_y):
            visible_mask[tid] = int(0)
            return

        visible_mask[tid] = int(1)
        p_proj_out[tid] = proj_value


    @wp.kernel
    def _project_preprocess_visible_points_scale_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        scales: wp.array(dtype=wp.vec3),
        scale_modifier: wp.float32,
        view_flat: wp.array(dtype=wp.float32),
        proj_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        grid_y: wp.int32,
        visible_mask: wp.array(dtype=wp.int32),
        p_proj_out: wp.array(dtype=wp.vec3),
        p_view_z_out: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        point = means3d[tid]

        p_view_z = view_flat[2] * point[0] + view_flat[6] * point[1] + view_flat[10] * point[2] + view_flat[14]
        p_view_z_out[tid] = p_view_z
        if p_view_z <= VISIBILITY_NEAR_PLANE:
            visible_mask[tid] = int(0)
            return

        hom_x = proj_flat[0] * point[0] + proj_flat[4] * point[1] + proj_flat[8] * point[2] + proj_flat[12]
        hom_y = proj_flat[1] * point[0] + proj_flat[5] * point[1] + proj_flat[9] * point[2] + proj_flat[13]
        hom_z = proj_flat[2] * point[0] + proj_flat[6] * point[1] + proj_flat[10] * point[2] + proj_flat[14]
        hom_w = proj_flat[3] * point[0] + proj_flat[7] * point[1] + proj_flat[11] * point[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        proj_value = wp.vec3(hom_x * inv_w, hom_y * inv_w, hom_z * inv_w)

        scale = scales[tid]
        scaled_max = scale_modifier * wp.max(wp.abs(scale[0]), wp.max(wp.abs(scale[1]), wp.abs(scale[2])))
        cov3d_lambda_upper = scaled_max * scaled_max
        radius_upper = _preprocess_radius_upper_wp(cov3d_lambda_upper, p_view_z, tanfovx, tanfovy, image_width, image_height)
        if not _preprocess_rect_visible_wp(proj_value, radius_upper, image_width, image_height, grid_x, grid_y):
            visible_mask[tid] = int(0)
            return

        visible_mask[tid] = int(1)
        p_proj_out[tid] = proj_value


    @wp.kernel
    def _duplicate_with_keys_warp_kernel(
        points_xy_image: wp.array(dtype=wp.vec2),
        point_offsets: wp.array(dtype=wp.int32),
        radii: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
        cov2d_inv: wp.array(dtype=wp.vec3),
        grid_x: wp.int32,
        grid_y: wp.int32,
        tile_ids_out: wp.array(dtype=wp.int32),
        point_list_out: wp.array(dtype=wp.int32),
    ):
        idx = wp.tid()
        radius = radii[idx]
        if radius <= 0:
            return

        off = int(0)
        if idx > 0:
            off = point_offsets[idx - 1]

        point = points_xy_image[idx]
        co = conic_opacity[idx]
        con_a = co[0]
        con_b = co[1]
        con_c = co[2]
        opac = co[3]
        
        cov = cov2d_inv[idx]
        rect = _compute_tile_rect_snugbox_cov2d_wp(point[0], point[1], cov[0], cov[2], opac, grid_x, grid_y)
        t_val = 2.0 * wp.log(wp.max(255.0 * opac, 1.0))
        det_c = con_a * con_c - con_b * con_b

        for tile_y in range(rect[1], rect[3]):
            dy = wp.clamp(point[1], float(tile_y * BLOCK_Y), float(tile_y * BLOCK_Y + BLOCK_Y - 1)) - point[1]
            row_range = _accutile_row_x_range_wp(point[0], con_a, con_b, det_c, t_val, dy, grid_x)
            row_x_min = wp.max(row_range[0], rect[0])
            row_x_max = wp.min(row_range[1], rect[2])
            for tile_x in range(row_x_min, row_x_max):
                dx = wp.clamp(point[0], float(tile_x * BLOCK_X), float(tile_x * BLOCK_X + BLOCK_X - 1)) - point[0]
                power = -0.5 * (con_a * dx * dx + con_c * dy * dy) - con_b * dx * dy
                if power > 0.0:
                    power = 0.0
                alpha = wp.min(0.99, opac * wp.exp(power))
                if alpha >= (1.0 / 255.0):
                    tile_ids_out[off] = tile_y * grid_x + tile_x
                    point_list_out[off] = idx
                    off = off + 1

        # Sentinel-fill remaining slots (AccuTile row range + alpha filter
        # may write fewer entries than preprocess SnugBox AABB allocated)
        end_off = point_offsets[idx]
        sentinel_tile = grid_x * grid_y
        while off < end_off:
            tile_ids_out[off] = sentinel_tile
            point_list_out[off] = 0
            off = off + 1

    @wp.kernel
    def _duplicate_with_packed_keys_warp_kernel(
        points_xy_image: wp.array(dtype=wp.vec2),
        point_offsets: wp.array(dtype=wp.int32),
        radii: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
        cov2d_inv: wp.array(dtype=wp.vec3),
        depths: wp.array(dtype=wp.float32),
        grid_x: wp.int32,
        grid_y: wp.int32,
        packed_keys_out: wp.array(dtype=wp.int64),
        point_list_out: wp.array(dtype=wp.int32),
    ):
        idx = wp.tid()
        radius = radii[idx]
        if radius <= 0:
            return

        off = int(0)
        if idx > 0:
            off = point_offsets[idx - 1]

        point = points_xy_image[idx]
        co = conic_opacity[idx]
        con_a = co[0]
        con_b = co[1]
        con_c = co[2]
        opac = co[3]
        cov = cov2d_inv[idx]
        rect = _compute_tile_rect_snugbox_cov2d_wp(point[0], point[1], cov[0], cov[2], opac, grid_x, grid_y)
        depth_bits = wp.cast(depths[idx], wp.int32)
        depth_key = wp.int64(depth_bits) & wp.int64(4294967295)
        t_val = 2.0 * wp.log(wp.max(255.0 * opac, 1.0))
        det_c = con_a * con_c - con_b * con_b

        for tile_y in range(rect[1], rect[3]):
            dy = wp.clamp(point[1], float(tile_y * BLOCK_Y), float(tile_y * BLOCK_Y + BLOCK_Y - 1)) - point[1]
            row_range = _accutile_row_x_range_wp(point[0], con_a, con_b, det_c, t_val, dy, grid_x)
            row_x_min = wp.max(row_range[0], rect[0])
            row_x_max = wp.min(row_range[1], rect[2])
            for tile_x in range(row_x_min, row_x_max):
                dx = wp.clamp(point[0], float(tile_x * BLOCK_X), float(tile_x * BLOCK_X + BLOCK_X - 1)) - point[0]
                power = -0.5 * (con_a * dx * dx + con_c * dy * dy) - con_b * dx * dy
                if power > 0.0:
                    power = 0.0
                alpha = wp.min(0.99, opac * wp.exp(power))
                if alpha >= (1.0 / 255.0):
                    tile_id = tile_y * grid_x + tile_x
                    packed_keys_out[off] = (wp.int64(tile_id) << wp.int64(32)) | depth_key
                    point_list_out[off] = idx
                    off = off + 1
        # Sentinel-fill remaining slots
        end_off = point_offsets[idx]
        sentinel_key = wp.int64(grid_x * grid_y) << wp.int64(32)
        while off < end_off:
            packed_keys_out[off] = sentinel_key
            point_list_out[off] = 0
            off = off + 1

    @wp.kernel
    def _duplicate_with_keys_from_order_warp_kernel(
        points_xy_image: wp.array(dtype=wp.vec2),
        radii: wp.array(dtype=wp.int32),
        conic_opacity: wp.array(dtype=wp.vec4),
        cov2d_inv: wp.array(dtype=wp.vec3),
        point_ids: wp.array(dtype=wp.int32),
        point_offsets: wp.array(dtype=wp.int32),
        grid_x: wp.int32,
        grid_y: wp.int32,
        tile_ids_out: wp.array(dtype=wp.int32),
        point_list_out: wp.array(dtype=wp.int32),
    ):
        idx = wp.tid()
        point_id = point_ids[idx]
        radius = radii[point_id]
        if radius <= 0:
            return

        off = int(0)
        if idx > 0:
            off = point_offsets[idx - 1]

        point = points_xy_image[point_id]
        co = conic_opacity[point_id]
        con_a = co[0]
        con_b = co[1]
        con_c = co[2]
        opac = co[3]
        cov = cov2d_inv[point_id]
        rect = _compute_tile_rect_snugbox_cov2d_wp(point[0], point[1], cov[0], cov[2], opac, grid_x, grid_y)
        t_val = 2.0 * wp.log(wp.max(255.0 * opac, 1.0))
        det_c = con_a * con_c - con_b * con_b

        for tile_y in range(rect[1], rect[3]):
            dy = wp.clamp(point[1], float(tile_y * BLOCK_Y), float(tile_y * BLOCK_Y + BLOCK_Y - 1)) - point[1]
            row_range = _accutile_row_x_range_wp(point[0], con_a, con_b, det_c, t_val, dy, grid_x)
            row_x_min = wp.max(row_range[0], rect[0])
            row_x_max = wp.min(row_range[1], rect[2])
            for tile_x in range(row_x_min, row_x_max):
                dx = wp.clamp(point[0], float(tile_x * BLOCK_X), float(tile_x * BLOCK_X + BLOCK_X - 1)) - point[0]
                power = -0.5 * (con_a * dx * dx + con_c * dy * dy) - con_b * dx * dy
                if power > 0.0:
                    power = 0.0
                alpha = wp.min(0.99, opac * wp.exp(power))
                if alpha >= (1.0 / 255.0):
                    tile_ids_out[off] = tile_y * grid_x + tile_x
                    point_list_out[off] = point_id
                    off = off + 1
        end_off = point_offsets[idx]
        sentinel_tile = grid_x * grid_y
        while off < end_off:
            tile_ids_out[off] = sentinel_tile
            point_list_out[off] = 0
            off = off + 1

    @wp.kernel
    def _pack_binning_keys_warp_kernel(
        tile_ids: wp.array(dtype=wp.int32),
        point_list: wp.array(dtype=wp.int32),
        depths: wp.array(dtype=wp.float32),
        packed_keys_out: wp.array(dtype=wp.int64),
    ):
        idx = wp.tid()
        point_idx = point_list[idx]
        depth_bits = wp.cast(depths[point_idx], wp.int32)
        depth_key = wp.int64(depth_bits) & wp.int64(4294967295)
        tile_key = wp.int64(tile_ids[idx]) << wp.int64(32)
        packed_keys_out[idx] = tile_key | depth_key


    @wp.kernel
    def _identify_tile_ranges_warp_kernel(
        tile_ids: wp.array(dtype=wp.int32),
        range_flat: wp.array(dtype=wp.int32),
        length: wp.int32,
    ):
        idx = wp.tid()
        if idx >= length:
            return

        curr_tile = tile_ids[idx]
        if idx == 0:
            range_flat[curr_tile * 2] = 0
        else:
            prev_tile = tile_ids[idx - 1]
            if curr_tile != prev_tile:
                range_flat[prev_tile * 2 + 1] = idx
                range_flat[curr_tile * 2] = idx

        if idx == length - 1:
            range_flat[curr_tile * 2 + 1] = length


    @wp.kernel
    def _identify_tile_ranges_from_packed_keys_warp_kernel(
        packed_keys: wp.array(dtype=wp.int64),
        range_flat: wp.array(dtype=wp.int32),
        length: wp.int32,
    ):
        idx = wp.tid()
        if idx >= length:
            return

        curr_tile = wp.int32(packed_keys[idx] >> wp.int64(32))
        if idx == 0:
            range_flat[curr_tile * 2] = 0
        else:
            prev_tile = wp.int32(packed_keys[idx - 1] >> wp.int64(32))
            if curr_tile != prev_tile:
                range_flat[prev_tile * 2 + 1] = idx
                range_flat[curr_tile * 2] = idx

        if idx == length - 1:
            range_flat[curr_tile * 2 + 1] = length


    # ---- W1: Warp32 backward render — 32-thread warp-level gradient reduction ----
    # Each 32-thread warp covers a spatial sub-tile (2 consecutive rows of a 16×16 screen tile).
    # Benefits vs non-tiled: 32× fewer atomic writes per Gaussian.
    @wp.kernel
    def _backward_render_tiles_warp32_kernel(
        ranges_flat: wp.array(dtype=wp.int32),
        point_list: wp.array(dtype=wp.int32),
        points_xy_image: wp.array(dtype=wp.vec2),
        features_flat: wp.array(dtype=wp.float32),
        depths: wp.array(dtype=wp.float32),
        conic_opacity: wp.array(dtype=wp.vec4),
        background: wp.array(dtype=wp.float32),
        out_alpha_flat: wp.array(dtype=wp.float32),
        n_contrib: wp.array(dtype=wp.int32),
        grad_color_flat: wp.array(dtype=wp.float32),
        grad_depth_flat: wp.array(dtype=wp.float32),
        grad_alpha_flat: wp.array(dtype=wp.float32),
        image_width: wp.int32,
        image_height: wp.int32,
        grid_x: wp.int32,
        num_tiles: wp.int32,
        compute_depth: wp.int32,
        grad_points_xy: wp.array(dtype=wp.vec2),
        grad_depths: wp.array(dtype=wp.float32),
        grad_conic_opacity: wp.array(dtype=wp.vec4),
        grad_feature: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        block_id = tid // _BLOCK_PIXELS_C
        local_id = tid % _BLOCK_PIXELS_C

        if block_id >= num_tiles:
            return

        tile_x = block_id % grid_x
        tile_y = block_id // grid_x
        pix_x = tile_x * BLOCK_X + (local_id % BLOCK_X)
        pix_y = tile_y * BLOCK_Y + (local_id // BLOCK_X)
        total_pixels = image_width * image_height

        inside = int(0)
        if pix_x < image_width and pix_y < image_height:
            inside = 1
        pixel_id = pix_y * image_width + pix_x

        start = ranges_flat[block_id * 2]
        end = ranges_flat[block_id * 2 + 1]
        n_gs = end - start

        T_final = float(0.0)
        T = float(0.0)
        last_contributor = int(0)
        ddelx_dx = 0.5 * float(image_width)
        ddely_dy = 0.5 * float(image_height)
        pixf_x = float(pix_x)
        pixf_y = float(pix_y)
        dL_dpixel = wp.vec3(0.0, 0.0, 0.0)
        dL_dpixel_depth = float(0.0)
        dL_dalpha = float(0.0)
        bg_dot = float(0.0)

        if inside != 0 and n_gs > 0:
            T_final = 1.0 - out_alpha_flat[pixel_id]
            T = T_final
            last_contributor = n_contrib[pixel_id]
            dL_dpixel = wp.vec3(
                grad_color_flat[pixel_id],
                grad_color_flat[total_pixels + pixel_id],
                grad_color_flat[2 * total_pixels + pixel_id],
            )
            if compute_depth != 0:
                dL_dpixel_depth = grad_depth_flat[pixel_id]
            dL_dalpha = grad_alpha_flat[pixel_id]
            bg = wp.vec3(background[0], background[1], background[2])
            bg_dot = wp.dot(bg, dL_dpixel)

        accum_rec = wp.vec3(0.0, 0.0, 0.0)
        accum_depth_rec = float(0.0)
        accum_alpha_rec = float(0.0)
        last_alpha = float(0.0)
        last_color = wp.vec3(0.0, 0.0, 0.0)
        last_depth = float(0.0)

        # W1: Warp-level max contributor — each 32-pixel sub-tile determines its own loop bound.
        t_lc = wp.tile(last_contributor)
        t_max_lc = wp.tile_reduce(wp.max, t_lc)
        warp_max_lc = wp.tile_extract(t_max_lc, 0)

        for step in range(warp_max_lc):
            pos = warp_max_lc - step - 1
            coll_id = point_list[start + pos]

            my_grad_feat = wp.vec3(0.0, 0.0, 0.0)
            my_grad_depth = float(0.0)
            my_grad_xy = wp.vec2(0.0, 0.0)
            my_grad_co = wp.vec4(0.0, 0.0, 0.0, 0.0)

            if inside != 0 and pos < last_contributor:
                xy = points_xy_image[coll_id]
                con_o = conic_opacity[coll_id]
                d_x = xy[0] - pixf_x
                d_y = xy[1] - pixf_y
                power = _compute_power(con_o, d_x, d_y)

                if power <= 0.0:
                    G = wp.exp(power)
                    alpha = _compute_alpha(con_o, power)
                    if alpha >= (1.0 / 255.0):
                        one_minus_alpha = wp.max(1.0 - alpha, ONE_MINUS_ALPHA_MIN)
                        T = T / one_minus_alpha
                        dchannel_dcolor = alpha * T

                        feature_base = coll_id * NUM_CHANNELS
                        color = wp.vec3(
                            features_flat[feature_base + 0],
                            features_flat[feature_base + 1],
                            features_flat[feature_base + 2],
                        )
                        accum_rec = last_alpha * last_color + (1.0 - last_alpha) * accum_rec
                        last_color = color
                        dL_dopa = wp.dot(color - accum_rec, dL_dpixel)
                        my_grad_feat = dchannel_dcolor * dL_dpixel

                        if compute_depth != 0:
                            depth_value = depths[coll_id]
                            accum_depth_rec = last_alpha * last_depth + (1.0 - last_alpha) * accum_depth_rec
                            last_depth = depth_value
                            dL_dopa = dL_dopa + (depth_value - accum_depth_rec) * dL_dpixel_depth
                            my_grad_depth = dchannel_dcolor * dL_dpixel_depth

                        accum_alpha_rec = last_alpha + (1.0 - last_alpha) * accum_alpha_rec
                        dL_dopa = dL_dopa + (1.0 - accum_alpha_rec) * dL_dalpha
                        dL_dopa = dL_dopa * T
                        last_alpha = alpha
                        dL_dopa = dL_dopa + (-T_final / one_minus_alpha) * bg_dot

                        dL_dG = con_o[3] * dL_dopa
                        gdx = G * d_x
                        gdy = G * d_y
                        dG_ddelx = -gdx * con_o[0] - gdy * con_o[1]
                        dG_ddely = -gdy * con_o[2] - gdx * con_o[1]

                        my_grad_xy = wp.vec2(dL_dG * dG_ddelx * ddelx_dx, dL_dG * dG_ddely * ddely_dy)
                        my_grad_co = wp.vec4(-0.5 * gdx * d_x * dL_dG, -0.5 * gdx * d_y * dL_dG, -0.5 * gdy * d_y * dL_dG, G * dL_dopa)

            # W1: Warp-level reduction — ALL 32 threads participate (UNIFORM path)
            t_feat = wp.tile(my_grad_feat, preserve_type=True)
            s_feat = wp.tile_reduce(wp.add, t_feat)
            wp.tile_atomic_add(grad_feature, s_feat, offset=coll_id)

            if compute_depth != 0:
                t_depth = wp.tile(my_grad_depth)
                s_depth = wp.tile_sum(t_depth)
                wp.tile_atomic_add(grad_depths, s_depth, offset=coll_id)

            t_xy = wp.tile(my_grad_xy, preserve_type=True)
            s_xy = wp.tile_reduce(wp.add, t_xy)
            wp.tile_atomic_add(grad_points_xy, s_xy, offset=coll_id)

            t_co = wp.tile(my_grad_co, preserve_type=True)
            s_co = wp.tile_reduce(wp.add, t_co)
            wp.tile_atomic_add(grad_conic_opacity, s_co, offset=coll_id)

    @wp.kernel
    def _backward_cov2d_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        radii: wp.array(dtype=wp.int32),
        cov3d_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        tanfovx: wp.float32,
        tanfovy: wp.float32,
        focal_x: wp.float32,
        focal_y: wp.float32,
        grad_conic_flat: wp.array(dtype=wp.float32),
        grad_conic_2d_flat: wp.array(dtype=wp.float32),
        grad_means_out: wp.array(dtype=wp.vec3),
        grad_cov_out_flat: wp.array(dtype=wp.float32),
    ):
        tid = wp.tid()
        if radii[tid] <= 0:
            return

        mean = means3d[tid]
        base = tid * 6
        grad_conic_base = tid * 3
        grad_conic2_base = tid * 3

        x = view_flat[0] * mean[0] + view_flat[4] * mean[1] + view_flat[8] * mean[2] + view_flat[12]
        y = view_flat[1] * mean[0] + view_flat[5] * mean[1] + view_flat[9] * mean[2] + view_flat[13]
        z = view_flat[2] * mean[0] + view_flat[6] * mean[1] + view_flat[10] * mean[2] + view_flat[14]

        limx = 1.3 * tanfovx
        limy = 1.3 * tanfovy
        txtz = x / z
        tytz = y / z
        tx = wp.clamp(txtz, -limx, limx) * z
        ty = wp.clamp(tytz, -limy, limy) * z
        x_grad_mul = float(1.0)
        y_grad_mul = float(1.0)
        if txtz < -limx or txtz > limx:
            x_grad_mul = float(0.0)
        if tytz < -limy or tytz > limy:
            y_grad_mul = float(0.0)

        j00 = focal_x / z
        j02 = -(focal_x * tx) / (z * z)
        j11 = focal_y / z
        j12 = -(focal_y * ty) / (z * z)

        w00 = view_flat[0]
        w01 = view_flat[1]
        w02 = view_flat[2]
        w10 = view_flat[4]
        w11 = view_flat[5]
        w12 = view_flat[6]
        w20 = view_flat[8]
        w21 = view_flat[9]
        w22 = view_flat[10]

        t00 = w00 * j00 + w02 * j02
        t10 = w10 * j00 + w12 * j02
        t20 = w20 * j00 + w22 * j02
        t01 = w01 * j11 + w02 * j12
        t11 = w11 * j11 + w12 * j12
        t21 = w21 * j11 + w22 * j12

        v00 = cov3d_flat[base + 0]
        v01 = cov3d_flat[base + 1]
        v02 = cov3d_flat[base + 2]
        v11 = cov3d_flat[base + 3]
        v12 = cov3d_flat[base + 4]
        v22 = cov3d_flat[base + 5]

        vt0x = v00 * t00 + v01 * t10 + v02 * t20
        vt0y = v01 * t00 + v11 * t10 + v12 * t20
        vt0z = v02 * t00 + v12 * t10 + v22 * t20
        vt1x = v00 * t01 + v01 * t11 + v02 * t21
        vt1y = v01 * t01 + v11 * t11 + v12 * t21
        vt1z = v02 * t01 + v12 * t11 + v22 * t21

        a = t00 * vt0x + t10 * vt0y + t20 * vt0z + 0.3
        b = t00 * vt1x + t10 * vt1y + t20 * vt1z
        c = t01 * vt1x + t11 * vt1y + t21 * vt1z + 0.3
        denom = a * c - b * b
        denom2inv = _conic_denom2inv_wp(denom)

        total_conic0 = grad_conic_flat[grad_conic_base + 0] + grad_conic_2d_flat[grad_conic2_base + 0]
        total_conic1 = grad_conic_flat[grad_conic_base + 1] + grad_conic_2d_flat[grad_conic2_base + 1]
        total_conic2 = grad_conic_flat[grad_conic_base + 2] + grad_conic_2d_flat[grad_conic2_base + 2]

        dL_da = denom2inv * (-c * c * total_conic0 + 2.0 * b * c * total_conic1 + (denom - a * c) * total_conic2)
        dL_dc = denom2inv * (-a * a * total_conic2 + 2.0 * a * b * total_conic1 + (denom - a * c) * total_conic0)
        dL_db = denom2inv * 2.0 * (b * c * total_conic0 - (denom + 2.0 * b * b) * total_conic1 + a * b * total_conic2)

        grad_cov_out_flat[base + 0] = t00 * t00 * dL_da + t00 * t01 * dL_db + t01 * t01 * dL_dc
        grad_cov_out_flat[base + 3] = t10 * t10 * dL_da + t10 * t11 * dL_db + t11 * t11 * dL_dc
        grad_cov_out_flat[base + 5] = t20 * t20 * dL_da + t20 * t21 * dL_db + t21 * t21 * dL_dc
        grad_cov_out_flat[base + 1] = 2.0 * t00 * t10 * dL_da + (t00 * t11 + t01 * t10) * dL_db + 2.0 * t01 * t11 * dL_dc
        grad_cov_out_flat[base + 2] = 2.0 * t00 * t20 * dL_da + (t00 * t21 + t01 * t20) * dL_db + 2.0 * t01 * t21 * dL_dc
        grad_cov_out_flat[base + 4] = 2.0 * t10 * t20 * dL_da + (t10 * t21 + t11 * t20) * dL_db + 2.0 * t11 * t21 * dL_dc

        dL_dT00 = 2.0 * (t00 * v00 + t10 * v01 + t20 * v02) * dL_da + (t01 * v00 + t11 * v01 + t21 * v02) * dL_db
        dL_dT01 = 2.0 * (t00 * v01 + t10 * v11 + t20 * v12) * dL_da + (t01 * v01 + t11 * v11 + t21 * v12) * dL_db
        dL_dT02 = 2.0 * (t00 * v02 + t10 * v12 + t20 * v22) * dL_da + (t01 * v02 + t11 * v12 + t21 * v22) * dL_db
        dL_dT10 = 2.0 * (t01 * v00 + t11 * v01 + t21 * v02) * dL_dc + (t00 * v00 + t10 * v01 + t20 * v02) * dL_db
        dL_dT11 = 2.0 * (t01 * v01 + t11 * v11 + t21 * v12) * dL_dc + (t00 * v01 + t10 * v11 + t20 * v12) * dL_db
        dL_dT12 = 2.0 * (t01 * v02 + t11 * v12 + t21 * v22) * dL_dc + (t00 * v02 + t10 * v12 + t20 * v22) * dL_db

        dL_dJ00 = w00 * dL_dT00 + w10 * dL_dT01 + w20 * dL_dT02
        dL_dJ02 = w02 * dL_dT00 + w12 * dL_dT01 + w22 * dL_dT02
        dL_dJ11 = w01 * dL_dT10 + w11 * dL_dT11 + w21 * dL_dT12
        dL_dJ12 = w02 * dL_dT10 + w12 * dL_dT11 + w22 * dL_dT12

        tz_inv = 1.0 / z
        tz2 = tz_inv * tz_inv
        tz3 = tz2 * tz_inv
        dL_dtx = x_grad_mul * -focal_x * tz2 * dL_dJ02
        dL_dty = y_grad_mul * -focal_y * tz2 * dL_dJ12
        dL_dtz = -focal_x * tz2 * dL_dJ00 - focal_y * tz2 * dL_dJ11 + (2.0 * focal_x * tx) * tz3 * dL_dJ02 + (2.0 * focal_y * ty) * tz3 * dL_dJ12

        grad_means_out[tid] = wp.vec3(
            view_flat[0] * dL_dtx + view_flat[1] * dL_dty + view_flat[2] * dL_dtz,
            view_flat[4] * dL_dtx + view_flat[5] * dL_dty + view_flat[6] * dL_dtz,
            view_flat[8] * dL_dtx + view_flat[9] * dL_dty + view_flat[10] * dL_dtz,
        )


    @wp.kernel
    def _backward_projected_means_warp_kernel(
        means3d: wp.array(dtype=wp.vec3),
        radii: wp.array(dtype=wp.int32),
        proj_flat: wp.array(dtype=wp.float32),
        view_flat: wp.array(dtype=wp.float32),
        grad_mean2d: wp.array(dtype=wp.vec2),
        grad_proj_2d: wp.array(dtype=wp.vec2),
        grad_depths: wp.array(dtype=wp.float32),
        out_grad_means: wp.array(dtype=wp.vec3),
    ):
        tid = wp.tid()
        mean = means3d[tid]

        hom_x = proj_flat[0] * mean[0] + proj_flat[4] * mean[1] + proj_flat[8] * mean[2] + proj_flat[12]
        hom_y = proj_flat[1] * mean[0] + proj_flat[5] * mean[1] + proj_flat[9] * mean[2] + proj_flat[13]
        hom_w = proj_flat[3] * mean[0] + proj_flat[7] * mean[1] + proj_flat[11] * mean[2] + proj_flat[15]
        inv_w = 1.0 / (hom_w + 0.0000001)
        mul1 = hom_x * inv_w * inv_w
        mul2 = hom_y * inv_w * inv_w

        grad_xy = grad_mean2d[tid]
        if radii[tid] > 0:
            grad_xy = grad_xy + grad_proj_2d[tid]
        out_x = (proj_flat[0] * inv_w - proj_flat[3] * mul1) * grad_xy[0] + (proj_flat[1] * inv_w - proj_flat[3] * mul2) * grad_xy[1]
        out_y = (proj_flat[4] * inv_w - proj_flat[7] * mul1) * grad_xy[0] + (proj_flat[5] * inv_w - proj_flat[7] * mul2) * grad_xy[1]
        out_z = (proj_flat[8] * inv_w - proj_flat[11] * mul1) * grad_xy[0] + (proj_flat[9] * inv_w - proj_flat[11] * mul2) * grad_xy[1]

        mul3 = view_flat[2] * mean[0] + view_flat[6] * mean[1] + view_flat[10] * mean[2] + view_flat[14]
        depth_grad = grad_depths[tid]
        out_x = out_x + (view_flat[2] - view_flat[3] * mul3) * depth_grad
        out_y = out_y + (view_flat[6] - view_flat[7] * mul3) * depth_grad
        out_z = out_z + (view_flat[10] - view_flat[11] * mul3) * depth_grad
        out_grad_means[tid] = wp.vec3(out_x, out_y, out_z)


    @wp.kernel
    def _backward_cov3d_from_scale_rotation_warp_kernel(
        scales: wp.array(dtype=wp.vec3),
        rotations: wp.array(dtype=wp.vec4),
        scale_modifier: wp.float32,
        grad_cov3d_flat: wp.array(dtype=wp.float32),
        grad_scales: wp.array(dtype=wp.vec3),
        grad_rotations: wp.array(dtype=wp.vec4),
    ):
        tid = wp.tid()
        s = scales[tid] * scale_modifier
        q = rotations[tid]
        grad_base = tid * 6

        r = q[0]
        x = q[1]
        y = q[2]
        z = q[3]

        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y + r * z)
        r02 = 2.0 * (x * z - r * y)
        r10 = 2.0 * (x * y - r * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z + r * x)
        r20 = 2.0 * (x * z + r * y)
        r21 = 2.0 * (y * z - r * x)
        r22 = 1.0 - 2.0 * (x * x + y * y)

        m00 = s[0] * r00
        m01 = s[0] * r01
        m02 = s[0] * r02
        m10 = s[1] * r10
        m11 = s[1] * r11
        m12 = s[1] * r12
        m20 = s[2] * r20
        m21 = s[2] * r21
        m22 = s[2] * r22

        sigma00 = grad_cov3d_flat[grad_base + 0]
        sigma01 = 0.5 * grad_cov3d_flat[grad_base + 1]
        sigma02 = 0.5 * grad_cov3d_flat[grad_base + 2]
        sigma11 = grad_cov3d_flat[grad_base + 3]
        sigma12 = 0.5 * grad_cov3d_flat[grad_base + 4]
        sigma22 = grad_cov3d_flat[grad_base + 5]

        dM00 = 2.0 * (m00 * sigma00 + m01 * sigma01 + m02 * sigma02)
        dM01 = 2.0 * (m00 * sigma01 + m01 * sigma11 + m02 * sigma12)
        dM02 = 2.0 * (m00 * sigma02 + m01 * sigma12 + m02 * sigma22)
        dM10 = 2.0 * (m10 * sigma00 + m11 * sigma01 + m12 * sigma02)
        dM11 = 2.0 * (m10 * sigma01 + m11 * sigma11 + m12 * sigma12)
        dM12 = 2.0 * (m10 * sigma02 + m11 * sigma12 + m12 * sigma22)
        dM20 = 2.0 * (m20 * sigma00 + m21 * sigma01 + m22 * sigma02)
        dM21 = 2.0 * (m20 * sigma01 + m21 * sigma11 + m22 * sigma12)
        dM22 = 2.0 * (m20 * sigma02 + m21 * sigma12 + m22 * sigma22)

        grad_scales[tid] = wp.vec3(
            scale_modifier * (dM00 * r00 + dM01 * r01 + dM02 * r02),
            scale_modifier * (dM10 * r10 + dM11 * r11 + dM12 * r12),
            scale_modifier * (dM20 * r20 + dM21 * r21 + dM22 * r22),
        )

        dR00 = dM00 * s[0]
        dR01 = dM01 * s[0]
        dR02 = dM02 * s[0]
        dR10 = dM10 * s[1]
        dR11 = dM11 * s[1]
        dR12 = dM12 * s[1]
        dR20 = dM20 * s[2]
        dR21 = dM21 * s[2]
        dR22 = dM22 * s[2]

        grad_rotations[tid] = wp.vec4(
            2.0 * z * (dR01 - dR10) + 2.0 * y * (dR20 - dR02) + 2.0 * x * (dR12 - dR21),
            2.0 * y * (dR10 + dR01) + 2.0 * z * (dR20 + dR02) + 2.0 * r * (dR12 - dR21) - 4.0 * x * (dR22 + dR11),
            2.0 * x * (dR10 + dR01) + 2.0 * r * (dR20 - dR02) + 2.0 * z * (dR12 + dR21) - 4.0 * y * (dR22 + dR00),
            2.0 * r * (dR01 - dR10) + 2.0 * x * (dR20 + dR02) + 2.0 * y * (dR12 + dR21) - 4.0 * z * (dR11 + dR00),
        )


def _make_empty_forward_outputs(means3D, image_height, image_width):
    point_count = means3D.shape[0]
    out_color = _allocate_scalar_tensor((NUM_CHANNELS, image_height, image_width), torch.float32, means3D.device, fill_value=0.0)
    out_depth = _allocate_scalar_tensor((1, image_height, image_width), torch.float32, means3D.device, fill_value=0.0)
    out_alpha = _allocate_scalar_tensor((1, image_height, image_width), torch.float32, means3D.device, fill_value=0.0)
    radii = _allocate_scalar_tensor((point_count,), torch.int32, means3D.device, fill_value=0)
    proj_2d = _allocate_scalar_tensor((point_count, 2), torch.float32, means3D.device, fill_value=0.0)
    conic_2d = _allocate_scalar_tensor((point_count, 3), torch.float32, means3D.device, fill_value=0.0)
    conic_2d_inv = _allocate_scalar_tensor((point_count, 3), torch.float32, means3D.device, fill_value=0.0)
    gs_per_pixel = torch.empty((0,), dtype=torch.float32, device=means3D.device)
    weight_per_gs_pixel = torch.empty((0,), dtype=torch.float32, device=means3D.device)
    x_mu = torch.empty((0,), dtype=torch.float32, device=means3D.device)
    geom_buffer = _allocate_scalar_tensor((0,), torch.uint8, means3D.device)
    binning_buffer = _allocate_scalar_tensor((0,), torch.uint8, means3D.device)
    img_buffer = _allocate_scalar_tensor((0,), torch.uint8, means3D.device)

    return (
        0,
        out_color,
        out_depth,
        out_alpha,
        radii,
        geom_buffer,
        binning_buffer,
        img_buffer,
        proj_2d,
        conic_2d,
        conic_2d_inv,
        gs_per_pixel,
        weight_per_gs_pixel,
        x_mu,
    )


def _compute_cov3d_from_scale_rotation_warp(scales, scale_modifier, rotations):
    if scales.numel() == 0:
        return torch.zeros((0, 6), dtype=torch.float32, device=scales.device)

    scales = scales.detach().contiguous()
    rotations = rotations.detach().contiguous()
    out_cov3d = torch.empty((scales.shape[0], 6), dtype=torch.float32, device=scales.device)
    wp.launch(
        kernel=_cov3d_from_scale_rotation_warp_kernel,
        dim=scales.shape[0],
        inputs=[
            wp.from_torch(scales, dtype=wp.vec3),
            wp.from_torch(rotations, dtype=wp.vec4),
            float(scale_modifier),
        ],
        outputs=[wp.from_torch(out_cov3d.reshape(-1), dtype=wp.float32)],
        device=str(scales.device),
    )
    return out_cov3d


def _project_visible_points_warp(means3D, viewmatrix, projmatrix):
    point_count = means3D.shape[0]
    visible_mask, p_proj, p_view_z = _get_project_visible_buffers(means3D.device, point_count)
    if point_count == 0:
        return visible_mask, p_proj, p_view_z

    wp.launch(
        kernel=_project_visible_points_warp_kernel,
        dim=point_count,
        inputs=[
            wp.from_torch(means3D.detach().contiguous(), dtype=wp.vec3),
            wp.from_torch(viewmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
            wp.from_torch(projmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
        ],
        outputs=[
            wp.from_torch(visible_mask, dtype=wp.int32),
            wp.from_torch(p_proj, dtype=wp.vec3),
            wp.from_torch(p_view_z, dtype=wp.float32),
        ],
        device=str(means3D.device),
    )
    return visible_mask, p_proj, p_view_z


def _project_preprocess_visible_points_warp(
    means3D,
    viewmatrix,
    projmatrix,
    tanfovx,
    tanfovy,
    image_width,
    image_height,
    grid_x,
    grid_y,
    cov3D_precomp=None,
    scales=None,
    scale_modifier=1.0,
):
    point_count = means3D.shape[0]
    visible_mask, p_proj, p_view_z = _get_project_visible_buffers(means3D.device, point_count)
    if point_count == 0:
        return visible_mask, p_proj, p_view_z

    outputs = [
        wp.from_torch(visible_mask, dtype=wp.int32),
        wp.from_torch(p_proj, dtype=wp.vec3),
        wp.from_torch(p_view_z, dtype=wp.float32),
    ]

    if cov3D_precomp is not None and cov3D_precomp.numel() != 0:
        wp.launch(
            kernel=_project_preprocess_visible_points_cov_warp_kernel,
            dim=point_count,
            inputs=[
                wp.from_torch(means3D.detach().contiguous(), dtype=wp.vec3),
                wp.from_torch(cov3D_precomp.detach().contiguous().reshape(-1), dtype=wp.float32),
                wp.from_torch(viewmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
                wp.from_torch(projmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
                float(tanfovx),
                float(tanfovy),
                int(image_width),
                int(image_height),
                int(grid_x),
                int(grid_y),
            ],
            outputs=outputs,
            device=str(means3D.device),
        )
        return visible_mask, p_proj, p_view_z

    if scales is None or scales.numel() == 0:
        raise ValueError("preprocess visibility requires either cov3D_precomp or scales")

    wp.launch(
        kernel=_project_preprocess_visible_points_scale_warp_kernel,
        dim=point_count,
        inputs=[
            wp.from_torch(means3D.detach().contiguous(), dtype=wp.vec3),
            wp.from_torch(scales.detach().contiguous(), dtype=wp.vec3),
            float(scale_modifier),
            wp.from_torch(viewmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
            wp.from_torch(projmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
            float(tanfovx),
            float(tanfovy),
            int(image_width),
            int(image_height),
            int(grid_x),
            int(grid_y),
        ],
        outputs=outputs,
        device=str(means3D.device),
    )
    return visible_mask, p_proj, p_view_z


def _compute_rgb_from_sh_warp(means3D, campos, shs, degree):
    point_count = means3D.shape[0]
    if point_count == 0 or shs.numel() == 0:
        rgb = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.float32, device=means3D.device)
        clamped_int = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.int32, device=means3D.device)
        return rgb, clamped_int

    rgb = torch.empty((point_count, NUM_CHANNELS), dtype=torch.float32, device=means3D.device)
    clamped_int = torch.empty((point_count, NUM_CHANNELS), dtype=torch.int32, device=means3D.device)

    coeff_count = shs.shape[1]
    _dev = str(means3D.device)
    # P2b: vec3 SH forward — 3× fewer load/store instructions
    _inp = [
        wp.from_torch(means3D.detach().contiguous(), dtype=wp.vec3),
        wp.from_torch(campos.detach().contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(shs.detach().contiguous().reshape(-1, 3), dtype=wp.vec3),
        int(degree),
        int(coeff_count),
    ]
    _out = [
        wp.from_torch(rgb.reshape(-1, 3), dtype=wp.vec3),
        wp.from_torch(clamped_int.reshape(-1), dtype=wp.int32),
    ]
    _key = (_dev, point_count)
    _cmd = _C4_LAUNCH_CACHE_FWD_SH.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_forward_rgb_from_sh_v3_warp_kernel, dim=point_count,
                         inputs=_inp, outputs=_out, device=_dev, record_cmd=True)
        _C4_LAUNCH_CACHE_FWD_SH[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    return rgb, clamped_int


def _backward_rgb_from_sh_warp(means3D, campos, shs, degree, clamped, grad_color):
    point_count = means3D.shape[0]
    grad_means = torch.empty(means3D.shape, dtype=means3D.dtype, device=means3D.device)
    # S1: grad_sh allocated per-branch to avoid peak overlap
    if point_count == 0 or shs.numel() == 0:
        grad_sh = torch.zeros(shs.shape, dtype=shs.dtype, device=shs.device)
        return grad_means, grad_sh

    coeff_count = shs.shape[1]
    # L2: skip conversion when clamped already int32 (from geom_buffer unpack or SH forward)
    clamped_i32 = clamped.contiguous() if clamped.dtype == torch.int32 else clamped.to(torch.int32).contiguous()
    _dev = str(means3D.device)

    _w_means = wp.from_torch(means3D.contiguous(), dtype=wp.vec3)
    _w_campos = wp.from_torch(campos.contiguous().reshape(-1), dtype=wp.float32)
    _w_grad_means = wp.from_torch(grad_means, dtype=wp.vec3)

    if degree <= 1:
        # degree 0-1: use monolithic kernel (scalar arrays, no deg 2-3 work)
        grad_sh = torch.zeros(shs.shape, dtype=shs.dtype, device=shs.device)
        _w_shs = wp.from_torch(shs.contiguous().reshape(-1), dtype=wp.float32)
        _w_clamped = wp.from_torch(clamped_i32.reshape(-1), dtype=wp.int32)
        _w_grad_color = wp.from_torch(grad_color.contiguous().reshape(-1), dtype=wp.float32)
        _w_grad_sh = wp.from_torch(grad_sh.reshape(-1), dtype=wp.float32)
        _inp = [_w_means, _w_campos, _w_shs, int(degree), int(coeff_count), _w_clamped, _w_grad_color]
        _out = [_w_grad_means, _w_grad_sh]
        _key = (_dev, point_count)
        _cmd = _C4_LAUNCH_CACHE_SH.get(_key)
        if _cmd is None:
            _cmd = wp.launch(kernel=_backward_rgb_from_sh_warp_kernel, dim=point_count,
                             inputs=_inp, outputs=_out, device=_dev, record_cmd=True)
            _C4_LAUNCH_CACHE_SH[_key] = _cmd
        else:
            for _i, _v in enumerate(_inp + _out):
                _cmd.set_param_at_index(_i, _v)
        _cmd.launch()
    else:
        # SoA layout for coalesced GPU access: (P, K, 3) → (K, P, 3)
        # Adjacent threads access adjacent vec3s → eliminates 86% excessive sectors
        _w_shs_v3 = wp.from_torch(shs.permute(1, 0, 2).contiguous().reshape(-1, 3), dtype=wp.vec3)
        grad_sh_soa = torch.zeros((coeff_count, point_count, 3), dtype=shs.dtype, device=means3D.device)
        _w_grad_sh_v3 = wp.from_torch(grad_sh_soa.reshape(-1, 3), dtype=wp.vec3)
        # M2: pass grad_color + clamped directly to kernel (avoids _masked_grad allocation)
        _w_grad_color_v3 = wp.from_torch(grad_color.contiguous().reshape(-1, 3), dtype=wp.vec3)
        _w_clamped_i32 = wp.from_torch(clamped_i32.reshape(-1), dtype=wp.int32)
        _key = (_dev, point_count)

        _inp = [_w_means, _w_campos, _w_shs_v3, int(degree), int(coeff_count), int(point_count), _w_grad_color_v3, _w_clamped_i32]
        _out = [_w_grad_means, _w_grad_sh_v3]
        _cmd = _C4_LAUNCH_CACHE_SH_V3.get(_key)
        if _cmd is None:
            _cmd = wp.launch(kernel=_backward_rgb_from_sh_v3_warp_kernel, dim=point_count,
                             inputs=_inp, outputs=_out, device=_dev, record_cmd=True)
            _C4_LAUNCH_CACHE_SH_V3[_key] = _cmd
        else:
            for _i, _v in enumerate(_inp + _out):
                _cmd.set_param_at_index(_i, _v)
        _cmd.launch()
        # Transpose grad_sh from SoA (K, P, 3) back to AoS (P, K, 3)
        grad_sh = grad_sh_soa.permute(1, 0, 2).contiguous()

    return grad_means, grad_sh


def _backward_cov3d_from_scale_rotation_warp(scales, scale_modifier, rotations, grad_cov3d):
    if scales.numel() == 0 or rotations.numel() == 0:
        return torch.zeros_like(scales), torch.zeros_like(rotations)

    grad_scales = torch.empty(scales.shape, dtype=scales.dtype, device=scales.device)
    grad_rot = torch.empty(rotations.shape, dtype=rotations.dtype, device=rotations.device)
    N = scales.shape[0]
    # C4: cache wp.from_torch wrappers + Launch object
    _inp = [
        # C5: skip redundant .detach().contiguous()
        wp.from_torch(scales.contiguous(), dtype=wp.vec3),
        wp.from_torch(rotations.contiguous(), dtype=wp.vec4),
        float(scale_modifier),
        wp.from_torch(grad_cov3d.contiguous().reshape(-1), dtype=wp.float32),
    ]
    _out = [
        wp.from_torch(grad_scales, dtype=wp.vec3),
        wp.from_torch(grad_rot, dtype=wp.vec4),
    ]
    _key = (str(scales.device), N)
    _cmd = _C4_LAUNCH_CACHE_COV3D.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_backward_cov3d_from_scale_rotation_warp_kernel, dim=N,
                         inputs=_inp, outputs=_out, device=str(scales.device), record_cmd=True)
        _C4_LAUNCH_CACHE_COV3D[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    return grad_scales, grad_rot


def _render_tiles_warp(preprocess_outputs, binning_state, feature_ptr, background, image_height, image_width):
    device = feature_ptr.device
    total_pixels = image_height * image_width
    out_color = torch.empty((NUM_CHANNELS, image_height, image_width), dtype=torch.float32, device=device)
    out_depth = torch.empty((1, image_height, image_width), dtype=torch.float32, device=device)
    out_alpha = torch.empty((1, image_height, image_width), dtype=torch.float32, device=device)
    n_contrib = torch.empty((total_pixels,), dtype=torch.int32, device=device)
    gs_per_pixel = torch.empty((0,), dtype=torch.float32, device=device)
    weight_per_gs_pixel = torch.empty((0,), dtype=torch.float32, device=device)
    x_mu = torch.empty((0,), dtype=torch.float32, device=device)

    if binning_state["num_rendered"] == 0:
        out_color.zero_()
        out_depth.zero_()
        out_alpha.zero_()
        n_contrib.zero_()
        return out_color, out_depth, out_alpha, gs_per_pixel, weight_per_gs_pixel, x_mu, n_contrib

    # F6: preprocess outputs are already contiguous from torch.empty(); skip redundant .detach().contiguous()
    points_xy_image = preprocess_outputs["points_xy_image"]
    conic_opacity = preprocess_outputs["conic_opacity"]
    depths = preprocess_outputs["depths"]
    feature_ptr = feature_ptr.detach().contiguous()
    background = background.detach().to(dtype=torch.float32, device=device).contiguous()
    ranges = binning_state["ranges"].reshape(-1)
    point_list = binning_state["point_list"]

    _wp_ranges = wp.from_torch(ranges, dtype=wp.int32)
    _wp_point_list = wp.from_torch(point_list, dtype=wp.int32)
    _wp_points_xy = wp.from_torch(points_xy_image, dtype=wp.vec2)
    _wp_features = wp.from_torch(feature_ptr.reshape(-1), dtype=wp.float32)
    _wp_depths = wp.from_torch(depths, dtype=wp.float32)
    _wp_conic_opacity = wp.from_torch(conic_opacity, dtype=wp.vec4)
    _wp_bg = wp.from_torch(background.reshape(-1), dtype=wp.float32)
    _wp_out_color = wp.from_torch(out_color.reshape(-1), dtype=wp.float32)
    _wp_out_depth = wp.from_torch(out_depth.reshape(-1), dtype=wp.float32)
    _wp_out_alpha = wp.from_torch(out_alpha.reshape(-1), dtype=wp.float32)
    _wp_n_contrib = wp.from_torch(n_contrib, dtype=wp.int32)
    _compute_depth_flag = int(1 if _COMPUTE_DEPTH else 0)
    # T1: Tiled-256 cooperative forward render
    _grid_x_fwd = int(binning_state["grid_x"])
    _grid_y_fwd = (image_height + BLOCK_Y - 1) // BLOCK_Y
    _num_tiles_fwd = _grid_x_fwd * _grid_y_fwd
    _dim = _num_tiles_fwd * (BLOCK_X * BLOCK_Y)
    _inp = [_wp_ranges, _wp_point_list, _wp_points_xy, _wp_features, _wp_depths,
            _wp_conic_opacity, _wp_bg, int(image_width), int(image_height),
            _grid_x_fwd, _num_tiles_fwd, _compute_depth_flag]
    _out = [_wp_out_color, _wp_out_depth, _wp_out_alpha, _wp_n_contrib]
    _key = (str(device), _dim)
    _cmd = _C4_LAUNCH_CACHE_FWD_RENDER_TILED256.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_render_tiles_tiled256_warp_kernel, dim=_dim,
                         inputs=_inp, outputs=_out, device=str(device), record_cmd=True,
                         block_dim=256)
        _C4_LAUNCH_CACHE_FWD_RENDER_TILED256[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    return out_color, out_depth, out_alpha, gs_per_pixel, weight_per_gs_pixel, x_mu, n_contrib


def _build_binning_state(
    preprocess_outputs,
    image_height,
    image_width,
    sort_mode: str | None = None,
):
        radii = preprocess_outputs["radii"]
        points_xy_image = preprocess_outputs["points_xy_image"]
        depths = preprocess_outputs["depths"]
        tiles_touched = _as_detached_contiguous_dtype(preprocess_outputs["tiles_touched"], torch.int32)
        point_count = radii.shape[0]
        device = radii.device
        grid_x = (image_width + BLOCK_X - 1) // BLOCK_X
        grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
        tile_count = grid_x * grid_y

        requested_sort_mode = sort_mode
        if sort_mode is None:
            sort_mode, _ = _select_auto_binning_sort_mode(device, point_count)

        if sort_mode not in BINNING_SORT_MODES:
            raise ValueError("sort_mode must be one of 'torch', 'warp_radix', or 'warp_depth_stable_tile'")

        sorted_point_ids: torch.Tensor | None = None
        if point_count > 0 and sort_mode == "warp_depth_stable_tile":
            device_key = str(device)

            # Always sort — the O6 depth-order-unchanged check requires a GPU→CPU
            # sync (.item()) that costs more than the sort itself during training.
            # Use the shared i32 radix sort buffers for depth sorting, then clone
            # sorted_point_ids before those buffers are reused by the duplicate sort.
            _ds_key_buf, _ds_val_buf, _, _ = _get_radix_sort_i32_buffers(device, point_count * 2)
            point_depth_keys = _ds_key_buf[:point_count]
            sorted_point_ids = _ds_val_buf[:point_count]
            point_depth_keys.copy_(depths.view(torch.int32))
            sorted_point_ids.copy_(_get_sequence_buffer(device, point_count))
            point_depth_keys, sorted_point_ids = _warp_radix_sort_i32_pairs_in_place(_ds_key_buf, _ds_val_buf, point_count)
            sorted_point_ids = sorted_point_ids.clone()
            _DEPTH_SORT_ORDER_CACHE[device_key] = (sorted_point_ids, point_count)

            sorted_tiles_touched = _gather_i32_by_index(tiles_touched, sorted_point_ids)
            point_offsets = _inclusive_scan_i32(sorted_tiles_touched)
        else:
            point_offsets = _inclusive_scan_i32(tiles_touched) if point_count > 0 else _allocate_scalar_tensor((0,), torch.int32, device)

        # ----- Prepare data for binning while GPU finishes the scan -----
        # Moving data prep before the .item() sync allows the host to do useful
        # work while waiting for the inclusive-scan result.
        ranges_warp = None
        ranges = torch.zeros((tile_count + 1, 2), dtype=torch.int32, device=device)
        point_list_keys = _allocate_scalar_tensor((0,), torch.int64, device)
        if point_count > 0:
            points_xy_image_vec2 = _as_detached_contiguous_dtype(points_xy_image, torch.float32)
            radii_i32 = _as_detached_contiguous_dtype(radii, torch.int32)
            conic_opacity_f32 = _as_detached_contiguous_dtype(preprocess_outputs["conic_opacity"], torch.float32)
            conic_opacity_wp = wp.from_torch(conic_opacity_f32, dtype=wp.vec4)
            cov2d_inv_f32 = _as_detached_contiguous_dtype(preprocess_outputs["conic_2d_inv"], torch.float32)
            cov2d_inv_wp = wp.from_torch(cov2d_inv_f32, dtype=wp.vec3)
            point_offsets_i32 = _as_detached_contiguous_dtype(point_offsets, torch.int32)

        # ----- Read num_rendered via GPU→CPU sync -----
        device_key = str(device)
        if point_count > 0:
            num_rendered = int(point_offsets_i32[-1].item())
        else:
            num_rendered = 0

        if num_rendered == 0:
            state = {
                "grid_x": grid_x,
                "grid_y": grid_y,
                "point_offsets": point_offsets,
                "point_list": _allocate_scalar_tensor((0,), torch.int32, device),
                "point_list_keys": _allocate_scalar_tensor((0,), torch.int64, device),
                "ranges": _allocate_scalar_tensor((tile_count, 2), torch.int32, device, fill_value=0),
                "num_rendered": 0,
                "requested_sort_mode": requested_sort_mode,
                "selected_sort_mode": sort_mode,
            }
            return state

        point_list = _allocate_scalar_tensor((num_rendered,), torch.int32, device)
        if sort_mode == "warp_radix":
            point_list_keys_buffer, point_list_buffer, _, _ = _get_radix_sort_buffers(device, num_rendered * 2)
            point_list_keys = point_list_keys_buffer[:num_rendered]
            point_list = point_list_buffer[:num_rendered]
            depths_f32 = _as_detached_contiguous_dtype(depths, torch.float32)
            wp.launch(
                kernel=_duplicate_with_packed_keys_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(points_xy_image_vec2, dtype=wp.vec2),
                    wp.from_torch(point_offsets_i32, dtype=wp.int32),
                    wp.from_torch(radii_i32, dtype=wp.int32),
                    conic_opacity_wp,
                    cov2d_inv_wp,
                    wp.from_torch(depths_f32, dtype=wp.float32),
                    int(grid_x),
                    int(grid_y),
                ],
                outputs=[
                    wp.from_torch(point_list_keys, dtype=wp.int64),
                    wp.from_torch(point_list, dtype=wp.int32),
                ],
                device=str(radii.device),
            )

            point_list_keys, point_list = _warp_radix_sort_pairs_in_place(point_list_keys_buffer, point_list_buffer, num_rendered)
        elif sort_mode == "warp_depth_stable_tile":
            tile_id_buffer, point_list_buffer, _, _ = _get_radix_sort_i32_buffers(device, num_rendered * 2)
            sorted_point_ids_i32 = _as_detached_contiguous_dtype(sorted_point_ids, torch.int32)
            _dev = str(radii.device)
            _g1_inp = [
                wp.from_torch(points_xy_image_vec2, dtype=wp.vec2),
                wp.from_torch(radii_i32, dtype=wp.int32),
                conic_opacity_wp,
                cov2d_inv_wp,
                wp.from_torch(sorted_point_ids_i32, dtype=wp.int32),
                wp.from_torch(point_offsets_i32, dtype=wp.int32),
                int(grid_x),
                int(grid_y),
            ]
            _g1_out = [
                wp.from_torch(tile_id_buffer[:num_rendered], dtype=wp.int32),
                wp.from_torch(point_list_buffer[:num_rendered], dtype=wp.int32),
            ]
            _g1_key = (_dev, point_count)
            _g1_cmd = _C4_LAUNCH_CACHE_BINNING_DUPLICATE.get(_g1_key)
            if _g1_cmd is None:
                _g1_cmd = wp.launch(
                    kernel=_duplicate_with_keys_from_order_warp_kernel,
                    dim=point_count,
                    inputs=_g1_inp, outputs=_g1_out,
                    device=_dev, record_cmd=True)
                _C4_LAUNCH_CACHE_BINNING_DUPLICATE[_g1_key] = _g1_cmd
            else:
                for _i, _v in enumerate(_g1_inp + _g1_out):
                    _g1_cmd.set_param_at_index(_i, _v)
            _g1_cmd.launch()
            tile_ids = tile_id_buffer[:num_rendered]
            point_list = point_list_buffer[:num_rendered]
            tile_ids, point_list = _warp_radix_sort_i32_pairs_in_place(tile_id_buffer, point_list_buffer, num_rendered)
        else:
            if _can_use_warp_scalar_alloc(device):
                tile_ids_warp, tile_ids = _allocate_warp_scalar_array(num_rendered, torch.int32, device)
                point_list_warp, point_list = _allocate_warp_scalar_array(num_rendered, torch.int32, device)
            else:
                tile_ids_warp = None
                point_list_warp = None
                tile_ids = torch.empty((num_rendered,), dtype=torch.int32, device=device)
                point_list = torch.empty((num_rendered,), dtype=torch.int32, device=device)
            wp.launch(
                kernel=_duplicate_with_keys_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(points_xy_image_vec2, dtype=wp.vec2),
                    wp.from_torch(point_offsets_i32, dtype=wp.int32),
                    wp.from_torch(radii_i32, dtype=wp.int32),
                    conic_opacity_wp,
                    cov2d_inv_wp,
                    int(grid_x),
                    int(grid_y),
                ],
                outputs=[
                    tile_ids_warp if tile_ids_warp is not None else wp.from_torch(tile_ids, dtype=wp.int32),
                    point_list_warp if point_list_warp is not None else wp.from_torch(point_list, dtype=wp.int32),
                ],
                device=str(radii.device),
            )

            if num_rendered <= TORCH_SINGLE_SORT_THRESHOLD:
                point_list_keys = _pack_binning_sort_keys(tile_ids, point_list, depths)

                order = torch.argsort(point_list_keys, stable=True)
                point_list = point_list[order]
                point_list_keys = point_list_keys[order]
                tile_ids = torch.bitwise_right_shift(point_list_keys, 32).to(torch.int32)
            else:
                point_depth_keys = _gather_i32_by_index(depths.view(torch.int32), point_list)

                order = torch.argsort(point_depth_keys, stable=True)
                tile_ids = tile_ids[order]
                point_list = point_list[order]

                order = torch.argsort(tile_ids, stable=True)
                point_list = point_list[order]
                tile_ids = tile_ids[order]
                point_list_keys = tile_ids.to(torch.int64)

        if sort_mode == "warp_radix":
            wp.launch(
                kernel=_identify_tile_ranges_from_packed_keys_warp_kernel,
                dim=point_list_keys.shape[0],
                inputs=[
                    wp.from_torch(point_list_keys, dtype=wp.int64),
                    ranges_warp if ranges_warp is not None else wp.from_torch(ranges.reshape(-1), dtype=wp.int32),
                    int(point_list_keys.shape[0]),
                ],
                device=str(radii.device),
            )
        else:
            tile_ids_i32 = _as_detached_contiguous_dtype(tile_ids, torch.int32)
            wp.launch(
                kernel=_identify_tile_ranges_warp_kernel,
                dim=tile_ids_i32.shape[0],
                inputs=[
                    wp.from_torch(tile_ids_i32, dtype=wp.int32),
                    ranges_warp if ranges_warp is not None else wp.from_torch(ranges.reshape(-1), dtype=wp.int32),
                    int(tile_ids_i32.shape[0]),
                ],
                device=str(radii.device),
            )
        # T1: pad point_list with RENDER_TILE_BATCH zeros so tile_load at the
        # tail of any tile's range never reads out-of-bounds memory.
        point_list = torch.cat([point_list, torch.zeros(RENDER_TILE_BATCH, dtype=point_list.dtype, device=device)])

        state = {
            "grid_x": grid_x,
            "grid_y": grid_y,
            "point_offsets": point_offsets,
            "point_list": point_list,
            "point_list_keys": point_list_keys,
            "ranges": ranges[:tile_count],
            "num_rendered": int(point_list.numel()) - RENDER_TILE_BATCH,
            "requested_sort_mode": requested_sort_mode,
            "selected_sort_mode": sort_mode,
        }
        return state

def _backward_render_tiles_warp(
    preprocess_outputs,
    binning_state,
    feature_ptr,
    background,
    image_height,
    image_width,
    out_alpha,
    n_contrib,
    grad_color,
    grad_depth,
    grad_alpha,
):
    device = feature_ptr.device
    point_count = feature_ptr.shape[0]

    # C3: Combined allocation — 1 memset instead of 4
    _stride = 2 + 1 + 4 + NUM_CHANNELS  # 10
    _bwd_total = point_count * _stride
    _combined = torch.zeros(_bwd_total, dtype=torch.float32, device=device)
    _off = 0
    grad_points_xy = _combined[_off:_off + point_count * 2].reshape(point_count, 2); _off += point_count * 2
    grad_depths = _combined[_off:_off + point_count]; _off += point_count
    grad_conic_opacity = _combined[_off:_off + point_count * 4].reshape(point_count, 4); _off += point_count * 4
    grad_feature = _combined[_off:_off + point_count * NUM_CHANNELS].reshape(point_count, NUM_CHANNELS)

    if binning_state["num_rendered"] == 0:
        outputs = (grad_points_xy, grad_depths, grad_conic_opacity, grad_feature)
        return outputs

    # C5: preprocess outputs are already contiguous (torch.empty/zeros); skip .detach().contiguous()
    points_xy_image = preprocess_outputs["points_xy_image"]
    conic_opacity = preprocess_outputs["conic_opacity"]
    depths = preprocess_outputs["depths"]
    feature_ptr = feature_ptr.contiguous()
    background = background.to(dtype=torch.float32, device=device)
    ranges = binning_state["ranges"].reshape(-1)
    point_list = binning_state["point_list"]
    out_alpha = out_alpha.reshape(-1)
    n_contrib = n_contrib.to(dtype=torch.int32, device=device).reshape(-1)

    grad_color = grad_color.contiguous().reshape(-1)
    grad_depth = grad_depth.contiguous().reshape(-1)
    grad_alpha = grad_alpha.contiguous().reshape(-1)

    # W1: Smart dispatch — warp32 for high density, non-tiled for low density
    _grid_x = int(binning_state["grid_x"])
    _grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
    _num_tiles = _grid_x * _grid_y
    _num_rendered = int(binning_state["num_rendered"])

    _wp_ranges = wp.from_torch(ranges, dtype=wp.int32)
    _wp_point_list = wp.from_torch(point_list, dtype=wp.int32)
    _wp_points_xy = wp.from_torch(points_xy_image, dtype=wp.vec2)
    _wp_features = wp.from_torch(feature_ptr.reshape(-1), dtype=wp.float32)
    _wp_depths = wp.from_torch(depths, dtype=wp.float32)
    _wp_conic_opacity = wp.from_torch(conic_opacity, dtype=wp.vec4)
    _wp_bg = wp.from_torch(background.reshape(-1), dtype=wp.float32)
    _wp_out_alpha = wp.from_torch(out_alpha, dtype=wp.float32)
    _wp_n_contrib = wp.from_torch(n_contrib, dtype=wp.int32)
    _wp_grad_color = wp.from_torch(grad_color, dtype=wp.float32)
    _wp_grad_depth = wp.from_torch(grad_depth, dtype=wp.float32)
    _wp_grad_alpha = wp.from_torch(grad_alpha, dtype=wp.float32)
    _wp_grad_xy = wp.from_torch(grad_points_xy, dtype=wp.vec2)
    _wp_grad_d = wp.from_torch(grad_depths, dtype=wp.float32)
    _wp_grad_co = wp.from_torch(grad_conic_opacity, dtype=wp.vec4)
    _wp_grad_f = wp.from_torch(grad_feature, dtype=wp.vec3)
    _compute_depth_flag = int(1 if _COMPUTE_DEPTH else 0)

    _dim = _num_tiles * (BLOCK_X * BLOCK_Y)
    # W1: warp32 backward — warp-level tile_reduce gives 32× fewer atomicAdd
    _inp = [_wp_ranges, _wp_point_list, _wp_points_xy, _wp_features, _wp_depths,
            _wp_conic_opacity, _wp_bg, _wp_out_alpha, _wp_n_contrib,
            _wp_grad_color, _wp_grad_depth, _wp_grad_alpha,
            int(image_width), int(image_height), _grid_x, _num_tiles, _compute_depth_flag]
    _out = [_wp_grad_xy, _wp_grad_d, _wp_grad_co, _wp_grad_f]
    _key = ("w32", str(device), _dim)
    _cmd = _C4_LAUNCH_CACHE_RENDER_BWD.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_backward_render_tiles_warp32_kernel, dim=_dim,
                         inputs=_inp, outputs=_out, device=str(device), record_cmd=True,
                         block_dim=32)
        _C4_LAUNCH_CACHE_RENDER_BWD[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    outputs = (grad_points_xy, grad_depths, grad_conic_opacity, grad_feature)
    return outputs


# -----------------------------------------------------------------------------
# Backward rendering and geometric gradients
# -----------------------------------------------------------------------------

def _backward_projected_means_warp(means3D, radii, projmatrix, viewmatrix,       grad_mean2d, grad_proj_2d, grad_depths):
    if means3D.numel() == 0:
        return torch.zeros_like(means3D)

    out = torch.empty(means3D.shape, dtype=means3D.dtype, device=means3D.device)
    N = means3D.shape[0]
    # C4: cache wp.from_torch wrappers + Launch object
    _inp = [
        # C5: skip redundant .detach().contiguous()
        wp.from_torch(means3D.contiguous(), dtype=wp.vec3),
        wp.from_torch(radii.contiguous(), dtype=wp.int32),
        wp.from_torch(projmatrix.contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(viewmatrix.contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(grad_mean2d.contiguous(), dtype=wp.vec2),
        wp.from_torch(grad_proj_2d.contiguous(), dtype=wp.vec2),
        wp.from_torch(grad_depths.contiguous(), dtype=wp.float32),
    ]
    _out = [wp.from_torch(out, dtype=wp.vec3)]
    _key = (str(means3D.device), N)
    _cmd = _C4_LAUNCH_CACHE_PROJ_MEANS.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_backward_projected_means_warp_kernel, dim=N,
                         inputs=_inp, outputs=_out, device=str(means3D.device), record_cmd=True)
        _C4_LAUNCH_CACHE_PROJ_MEANS[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    return out


def _backward_cov2d_warp(means3D, radii, cov3D, viewmatrix, tanfovx, tanfovy, focal_x, focal_y, grad_conic, grad_conic_2d):
    grad_means = torch.zeros(means3D.shape, dtype=means3D.dtype, device=means3D.device)
    grad_cov = torch.zeros(cov3D.shape, dtype=cov3D.dtype, device=cov3D.device)
    if means3D.numel() == 0:
        return grad_means, grad_cov

    N = means3D.shape[0]
    viewmatrix = viewmatrix.contiguous()
    # C4: cache wp.from_torch wrappers + Launch object
    _inp = [
        # C5: skip redundant .detach().contiguous()
        wp.from_torch(means3D.contiguous(), dtype=wp.vec3),
        wp.from_torch(radii.contiguous(), dtype=wp.int32),
        wp.from_torch(cov3D.contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(viewmatrix.reshape(-1), dtype=wp.float32),
        float(tanfovx),
        float(tanfovy),
        float(focal_x),
        float(focal_y),
        wp.from_torch(grad_conic.contiguous().reshape(-1), dtype=wp.float32),
        wp.from_torch(grad_conic_2d.contiguous().reshape(-1), dtype=wp.float32),
    ]
    _out = [
        wp.from_torch(grad_means, dtype=wp.vec3),
        wp.from_torch(grad_cov.reshape(-1), dtype=wp.float32),
    ]
    _key = (str(means3D.device), N)
    _cmd = _C4_LAUNCH_CACHE_COV2D.get(_key)
    if _cmd is None:
        _cmd = wp.launch(kernel=_backward_cov2d_warp_kernel, dim=N,
                         inputs=_inp, outputs=_out, device=str(means3D.device), record_cmd=True)
        _C4_LAUNCH_CACHE_COV2D[_key] = _cmd
    else:
        for _i, _v in enumerate(_inp + _out):
            _cmd.set_param_at_index(_i, _v)
    _cmd.launch()
    return grad_means, grad_cov


# -----------------------------------------------------------------------------
# Forward preprocessing pipeline
# -----------------------------------------------------------------------------


def preprocess_gaussians(
    means3D,
    viewmatrix,
    projmatrix,
    image_height,
    image_width,
    tanfovx,
    tanfovy,
    cov3D_precomp=None,
    scales=None,
    rotations=None,
    scale_modifier=1.0,
    shs=None,
    degree=0,
    campos=None,
    colors_precomp=None,
    opacities=None,
    prefiltered=False,
):
        _require_warp()
        if means3D.ndim != 2 or means3D.shape[1] != 3:
            raise ValueError("means3D must have dimensions (num_points, 3)")
        if prefiltered:
            raise NotImplementedError("prefiltered=True is not supported in the Warp preprocess path yet.")

        has_precomputed_cov = cov3D_precomp is not None and cov3D_precomp.numel() != 0
        has_scale_rotation = scales is not None and rotations is not None and scales.numel() != 0 and rotations.numel() != 0

        if has_precomputed_cov == has_scale_rotation:
            raise ValueError("Provide exactly one of cov3D_precomp or scales/rotations")

        device = means3D.device

        cov3d_all = None
        # E1: for scale_rotation Warp path, defer cov3d to fused kernel
        if has_precomputed_cov:
            if cov3D_precomp.ndim != 2 or cov3D_precomp.shape[1] != 6:
                raise ValueError("cov3D_precomp must have dimensions (num_points, 6)")
            if means3D.shape[0] != cov3D_precomp.shape[0]:
                raise ValueError("means3D and cov3D_precomp must have the same number of points")
            cov3d_all = cov3D_precomp
        else:
            if has_scale_rotation:
                # E1: cov3d computed inside fused kernel; allocate output tensor here
                cov3d_all = torch.empty((means3D.shape[0], 6), dtype=torch.float32, device=device)
            else:
                cov3d_all = _compute_cov3d_from_scale_rotation_warp(scales, scale_modifier, rotations)
            if means3D.shape[0] != cov3d_all.shape[0]:
                raise ValueError("means3D and computed covariances must have the same number of points")

        point_count = means3D.shape[0]
        # B5: auto-detect pruning (>20% point drop) and clear stale grow-only caches
        global _PREV_POINT_COUNT
        if _PREV_POINT_COUNT is not None and point_count < int(_PREV_POINT_COUNT * 0.8):
            clear_warp_caches()
        _PREV_POINT_COUNT = point_count
        proj_2d = torch.empty((point_count, 2), dtype=torch.float32, device=device)
        conic_2d = torch.empty((point_count, 3), dtype=torch.float32, device=device)
        conic_2d_inv = torch.empty((point_count, 3), dtype=torch.float32, device=device)
        radii = torch.empty((point_count,), dtype=torch.int32, device=device)
        depths = torch.empty((point_count,), dtype=torch.float32, device=device)
        points_xy_image = torch.empty((point_count, 2), dtype=torch.float32, device=device)
        tiles_touched = torch.empty((point_count,), dtype=torch.int32, device=device)
        conic_opacity = torch.empty((point_count, 4), dtype=torch.float32, device=device)
        rgb = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.float32, device=device)
        clamped = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.bool, device=device)
        if opacities is None:
            opacities = torch.zeros((point_count, 1), dtype=torch.float32, device=device)
        else:
            opacities = opacities.reshape(point_count, -1).to(dtype=torch.float32)
        focal_x = image_width / (2.0 * tanfovx)
        focal_y = image_height / (2.0 * tanfovy)
        grid_x = (image_width + BLOCK_X - 1) // BLOCK_X
        grid_y = (image_height + BLOCK_Y - 1) // BLOCK_Y
        if has_scale_rotation:
          # E1: fused project + cov3d + cov2d — visibility, cov3d, and all
          # preprocess outputs produced in a single kernel launch.
          visible_mask = torch.empty((point_count,), dtype=torch.int32, device=device)
          if point_count > 0:
              _e1_inp = [
                  wp.from_torch(means3D.detach().contiguous(), dtype=wp.vec3),
                  wp.from_torch(scales.detach().contiguous(), dtype=wp.vec3),
                  wp.from_torch(rotations.detach().contiguous(), dtype=wp.vec4),
                  float(scale_modifier),
                  wp.from_torch(opacities.reshape(-1).detach().contiguous(), dtype=wp.float32),
                  wp.from_torch(viewmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
                  wp.from_torch(projmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
                  float(tanfovx),
                  float(tanfovy),
                  float(focal_x),
                  float(focal_y),
                  int(image_width),
                  int(image_height),
                  int(grid_x),
                  int(grid_y),
              ]
              _e1_out = [
                  wp.from_torch(cov3d_all.reshape(-1), dtype=wp.float32),
                  wp.from_torch(visible_mask, dtype=wp.int32),
                  wp.from_torch(depths, dtype=wp.float32),
                  wp.from_torch(radii, dtype=wp.int32),
                  wp.from_torch(proj_2d, dtype=wp.vec2),
                  wp.from_torch(conic_2d, dtype=wp.vec3),
                  wp.from_torch(conic_2d_inv, dtype=wp.vec3),
                  wp.from_torch(points_xy_image, dtype=wp.vec2),
                  wp.from_torch(tiles_touched, dtype=wp.int32),
                  wp.from_torch(conic_opacity, dtype=wp.vec4),
              ]
              _e1_key = (str(device), point_count)
              _e1_cmd = _C4_LAUNCH_CACHE_FWD_PREPROCESS.get(_e1_key)
              if _e1_cmd is None:
                  _e1_cmd = wp.launch(
                      kernel=_fused_project_cov3d_cov2d_preprocess_sr_warp_kernel,
                      dim=point_count,
                      inputs=_e1_inp,
                      outputs=_e1_out,
                      device=str(device),
                      record_cmd=True,
                      block_dim=_get_block_dim("preprocess"),
                  )
                  _C4_LAUNCH_CACHE_FWD_PREPROCESS[_e1_key] = _e1_cmd
              else:
                  for _i, _v in enumerate(_e1_inp + _e1_out):
                      _e1_cmd.set_param_at_index(_i, _v)
              _e1_cmd.launch()
          p_proj_all = None
          p_view_z_all = None
        elif has_precomputed_cov:
            visible_mask, p_proj_all, p_view_z_all = _project_preprocess_visible_points_warp(
                means3D,
                viewmatrix,
                projmatrix,
                tanfovx,
                tanfovy,
                image_width,
                image_height,
                grid_x,
                grid_y,
                cov3D_precomp=cov3d_all,
            )
        else:
            visible_mask, p_proj_all, p_view_z_all = _project_visible_points_warp(means3D, viewmatrix, projmatrix)

        visible = visible_mask.to(torch.bool)

        if cov3d_all is None:
            cov3d_all = torch.zeros((point_count, 6), dtype=torch.float32, device=device)

        # F3: E1 fused path processes all points — skip GPU→CPU sync
        visible_count = point_count if has_scale_rotation else (int(visible_mask.sum().item()) if point_count > 0 else 0)

        if point_count == 0 or visible_count == 0:
            depths.zero_()
            radii.zero_()
            proj_2d.zero_()
            conic_2d.zero_()
            conic_2d_inv.zero_()
            points_xy_image.zero_()
            tiles_touched.zero_()
            conic_opacity.zero_()
            outputs = {
                "visible": visible,
                "depths": depths,
                "radii": radii,
                "proj_2d": proj_2d,
                "conic_2d": conic_2d,
                "conic_2d_inv": conic_2d_inv,
                "points_xy_image": points_xy_image,
                "tiles_touched": tiles_touched,
                "rgb": rgb,
                "clamped": clamped,
                "conic_opacity": conic_opacity,
                "cov3d_all": cov3d_all.to(dtype=torch.float32),
            }
            return outputs

        if has_scale_rotation:
            pass  # E1: preprocess already done by fused kernel
        elif has_precomputed_cov:
            wp.launch(
                kernel=_cov2d_preprocess_masked_pack_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(visible_mask.detach().contiguous(), dtype=wp.int32),
                    wp.from_torch(means3D.detach().contiguous(), dtype=wp.vec3),
                    wp.from_torch(cov3d_all.detach().contiguous().reshape(-1), dtype=wp.float32),
                    wp.from_torch(viewmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
                    wp.from_torch(p_proj_all.detach().contiguous(), dtype=wp.vec3),
                    wp.from_torch(p_view_z_all.detach().contiguous(), dtype=wp.float32),
                    wp.from_torch(opacities.reshape(-1).detach().contiguous(), dtype=wp.float32),
                    float(tanfovx),
                    float(tanfovy),
                    float(focal_x),
                    float(focal_y),
                    int(image_width),
                    int(image_height),
                    int(grid_x),
                    int(grid_y),
                ],
                outputs=[
                    wp.from_torch(depths, dtype=wp.float32),
                    wp.from_torch(radii, dtype=wp.int32),
                    wp.from_torch(proj_2d, dtype=wp.vec2),
                    wp.from_torch(conic_2d, dtype=wp.vec3),
                    wp.from_torch(conic_2d_inv, dtype=wp.vec3),
                    wp.from_torch(points_xy_image, dtype=wp.vec2),
                    wp.from_torch(tiles_touched, dtype=wp.int32),
                    wp.from_torch(conic_opacity, dtype=wp.vec4),
                ],
                device=str(device),
            )
        elif not has_scale_rotation:
            wp.launch(
                kernel=_cov2d_preprocess_masked_pack_scale_rotation_warp_kernel,
                dim=point_count,
                inputs=[
                    wp.from_torch(visible_mask.detach().contiguous(), dtype=wp.int32),
                    wp.from_torch(means3D.detach().contiguous(), dtype=wp.vec3),
                    wp.from_torch(scales.detach().contiguous(), dtype=wp.vec3),
                    wp.from_torch(rotations.detach().contiguous(), dtype=wp.vec4),
                    float(scale_modifier),
                    wp.from_torch(viewmatrix.detach().contiguous().reshape(-1), dtype=wp.float32),
                    wp.from_torch(p_proj_all.detach().contiguous(), dtype=wp.vec3),
                    wp.from_torch(p_view_z_all.detach().contiguous(), dtype=wp.float32),
                    wp.from_torch(opacities.reshape(-1).detach().contiguous(), dtype=wp.float32),
                    float(tanfovx),
                    float(tanfovy),
                    float(focal_x),
                    float(focal_y),
                    int(image_width),
                    int(image_height),
                    int(grid_x),
                    int(grid_y),
                ],
                outputs=[
                    wp.from_torch(depths, dtype=wp.float32),
                    wp.from_torch(radii, dtype=wp.int32),
                    wp.from_torch(proj_2d, dtype=wp.vec2),
                    wp.from_torch(conic_2d, dtype=wp.vec3),
                    wp.from_torch(conic_2d_inv, dtype=wp.vec3),
                    wp.from_torch(points_xy_image, dtype=wp.vec2),
                    wp.from_torch(tiles_touched, dtype=wp.int32),
                    wp.from_torch(conic_opacity, dtype=wp.vec4),
                ],
                device=str(device),
            )

        if colors_precomp is not None and colors_precomp.numel() != 0:
            rgb = colors_precomp.reshape(point_count, NUM_CHANNELS).to(dtype=torch.float32)
        elif shs is not None and shs.numel() != 0:
            if campos is None:
                raise ValueError("campos is required when computing colors from SH coefficients")
            shs = shs.to(device=device, dtype=torch.float32)
            campos = campos.to(device=device, dtype=torch.float32)
            rgb, clamped = _compute_rgb_from_sh_warp(means3D, campos, shs, degree)

        outputs = {
            "visible": visible,
            "depths": depths,
            "radii": radii,
            "proj_2d": proj_2d,
            "conic_2d": conic_2d,
            "conic_2d_inv": conic_2d_inv,
            "points_xy_image": points_xy_image,
            "tiles_touched": tiles_touched,
            "rgb": rgb,
            "clamped": clamped,
            "conic_opacity": conic_opacity,
            "cov3d_all": cov3d_all.to(dtype=torch.float32),
        }

        return outputs


# -----------------------------------------------------------------------------
# Backward rasterization replay
# -----------------------------------------------------------------------------


def _rasterize_gaussians_backward_python(*args: Any):
        (
            _background,
            means3D,
            _radii,
            _colors,
            _opacities,
            _scales,
            _rotations,
            _scale_modifier,
            _cov3D_precomp,
            _viewmatrix,
            _projmatrix,
            _tan_fovx,
            _tan_fovy,
            grad_color,
            grad_depth,
            grad_alpha,
            grad_proj_2D,
            grad_conic_2D,
            _grad_conic_2D_inv,
            _dummy_gs_per_pixel,
            _dummy_weight_per_gs_pixel,
            _grad_x_mu,
            _sh,
            _degree,
            _campos,
            _geomBuffer,
            _num_rendered,
            _binningBuffer,
            _imgBuffer,
            _alphas,
        ) = args

        point_count = means3D.shape[0]
        device = means3D.device
        # C3: grad_means2D and grad_means3D allocated later by fused accumulation kernel
        # O3: empty_like for grads that are always overwritten before return
        grad_colors = torch.empty_like(_colors)
        grad_opacities = torch.empty_like(_opacities)
        grad_cov3D = torch.empty_like(_cov3D_precomp)
        # L1: grad_scales/grad_rotations deferred to branch interiors (saves ~7 MB @262K on fused path)
        if point_count == 0:
            grad_means2D = torch.zeros((point_count, 3), dtype=torch.float32, device=device)
            grad_means3D = torch.zeros_like(means3D)
            grad_sh = torch.empty_like(_sh)
            grad_scales = torch.zeros_like(_scales)
            grad_rotations = torch.zeros_like(_rotations)
            return grad_means2D, grad_colors, grad_opacities, grad_means3D, grad_cov3D, grad_sh, grad_scales, grad_rotations

        image_height = grad_color.shape[1]
        image_width = grad_color.shape[2]
        cached_forward_state = _unpack_forward_aux_buffers(_geomBuffer, _binningBuffer, _imgBuffer, _num_rendered, image_height, image_width)

        if cached_forward_state is not None:
            preprocess_outputs, binning_state, n_contrib = cached_forward_state
            # Shallow copy to avoid mutating the cached dict
            preprocess_outputs = {**preprocess_outputs, "radii": _radii}
            cov3d_all = preprocess_outputs["cov3d_all"]
        else:
            # Only the cov3d source differs between precomp and scale/rotation paths
            if _cov3D_precomp.numel() != 0:
                cov3d_all = _cov3D_precomp
            else:
                cov3d_all = _compute_cov3d_from_scale_rotation_warp(_scales, _scale_modifier, _rotations)
            preprocess_outputs = preprocess_gaussians(
                    means3D,
                    _viewmatrix,
                    _projmatrix,
                    image_height,
                    image_width,
                    _tan_fovx,
                    _tan_fovy,
                    cov3D_precomp=cov3d_all,
                    shs=_sh.reshape(point_count, -1, NUM_CHANNELS) if _sh.numel() != 0 else None,
                    degree=_degree,
                    campos=_campos,
                    colors_precomp=_colors if _colors.numel() != 0 else None,
                    opacities=_opacities,
                    prefiltered=False,
                )
            # Build binning unconditionally — num_rendered (Python int) avoids
            # the host sync that bool((radii > 0).any()) would trigger.
            binning_state = _build_binning_state(preprocess_outputs, image_height, image_width)
            # Recover n_contrib from the saved img buffer instead of re-running
            # the full forward render (which allocates unused per-pixel outputs).
            _expected_img_bytes = image_height * image_width * 4  # int32 element_size
            if _imgBuffer.numel() == _expected_img_bytes:
                n_contrib = _imgBuffer.view(torch.int32).reshape(image_height, image_width)
            else:
                n_contrib = None

        # Unified: preprocess_gaussians already stores the correct rgb
        # (colors_precomp when provided, SH-evaluated colors otherwise).
        feature_ptr = preprocess_outputs["rgb"]
        # Always use the saved alpha tensor from forward.
        render_alpha = _alphas
        background_float = _background.to(dtype=torch.float32)
        # num_rendered is a Python int — no device sync.
        has_active_points = binning_state["num_rendered"] != 0

        if has_active_points:
            # Fallback: if n_contrib could not be recovered, re-render.
            if n_contrib is None:
                _, _, _, _, _, _, n_contrib = _render_tiles_warp(
                        preprocess_outputs,
                        binning_state,
                        feature_ptr,
                        background_float,
                        image_height,
                        image_width,
                    )
            render_grad_points, render_grad_depths, render_grad_conic_opacity, render_grad_feature = _backward_render_tiles_warp(
                preprocess_outputs,
                binning_state,
                feature_ptr,
                background_float,
                image_height,
                image_width,
                render_alpha,
                n_contrib.reshape(image_height, image_width),
                grad_color,
                grad_depth,
                grad_alpha,
            )
        else:
            render_grad_points = torch.zeros((point_count, 2), dtype=torch.float32, device=device)
            render_grad_depths = torch.zeros((point_count,), dtype=torch.float32, device=device)
            render_grad_conic_opacity = torch.zeros((point_count, 4), dtype=torch.float32, device=device)
            render_grad_feature = torch.zeros((point_count, NUM_CHANNELS), dtype=torch.float32, device=device)


        grad_proj_2d_active = grad_proj_2D
        grad_conic_2d_active = grad_conic_2D

        focal_x = image_width / (2.0 * _tan_fovx)
        focal_y = image_height / (2.0 * _tan_fovy)

        # C6: Fuse cov2d + cov3d into single kernel when scales/rotations are available
        _use_fused_cov = (_cov3D_precomp.numel() == 0
                          and _scales.numel() != 0
                          and _rotations.numel() != 0)

        # opacity / color grads (independent of preprocess backward)
        if grad_opacities.numel() != 0:
            grad_opacities = render_grad_conic_opacity[:, 3:4]
        if grad_colors.numel() != 0:
            grad_colors = render_grad_feature.reshape_as(_colors)
        # SH backward (must run before E2 fused kernel)
        _has_sh = _sh.numel() != 0
        if _has_sh:
                _grad_mean_sh, grad_sh_local = _backward_rgb_from_sh_warp(
                    means3D,
                    _campos.to(device=device, dtype=torch.float32),
                    _sh.reshape(point_count, -1, NUM_CHANNELS),
                    _degree,
                    preprocess_outputs["clamped"],
                    render_grad_feature,
                )
                grad_sh = grad_sh_local.reshape_as(_sh)
        else:
                grad_sh = torch.empty_like(_sh)
        
        if _use_fused_cov:
                grad_means3D = torch.empty(means3D.shape, dtype=means3D.dtype, device=device)
                grad_means2D = torch.empty((point_count, 3), dtype=torch.float32, device=device)
                grad_scales = torch.empty(_scales.shape, dtype=_scales.dtype, device=device)
                grad_rotations = torch.empty(_rotations.shape, dtype=_rotations.dtype, device=device)
                # L3: when has_sh=False the kernel never reads grad_sh_means — reuse means3D to avoid zero alloc
                _sh_grad = _grad_mean_sh if _has_sh else means3D
                _proj_flat = _projmatrix.contiguous().reshape(-1)
                _view_flat = _viewmatrix.contiguous().reshape(-1)
                _grad_conic_2d_flat = grad_conic_2d_active.contiguous().reshape(-1)
                _cov3d_flat = cov3d_all.contiguous().reshape(-1)
                _e2_inp = [
                    wp.from_torch(means3D.contiguous(), dtype=wp.vec3),
                    wp.from_torch(preprocess_outputs["radii"].contiguous(), dtype=wp.int32),
                    wp.from_torch(_proj_flat, dtype=wp.float32),
                    wp.from_torch(_view_flat, dtype=wp.float32),
                    wp.from_torch(render_grad_points, dtype=wp.vec2),
                    wp.from_torch(grad_proj_2d_active.contiguous(), dtype=wp.vec2),
                    wp.from_torch(render_grad_depths, dtype=wp.float32),
                    wp.from_torch(_cov3d_flat, dtype=wp.float32),
                    float(_tan_fovx),
                    float(_tan_fovy),
                    float(focal_x),
                    float(focal_y),
                    wp.from_torch(render_grad_conic_opacity, dtype=wp.vec4),
                    wp.from_torch(_grad_conic_2d_flat, dtype=wp.float32),
                    wp.from_torch(_scales.contiguous(), dtype=wp.vec3),
                    wp.from_torch(_rotations.contiguous(), dtype=wp.vec4),
                    float(_scale_modifier),
                    wp.from_torch(_sh_grad, dtype=wp.vec3),
                    int(_has_sh),
                    wp.from_torch(render_grad_points, dtype=wp.vec2),
                ]
                _e2_out = [
                    wp.from_torch(grad_means3D, dtype=wp.vec3),
                    wp.from_torch(grad_means2D, dtype=wp.vec3),
                    wp.from_torch(grad_scales, dtype=wp.vec3),
                    wp.from_torch(grad_rotations, dtype=wp.vec4),
                ]
                _e2_key = (str(device), point_count, int(_has_sh))
                _e2_cmd = _C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS.get(_e2_key)
                if _e2_cmd is None:
                    _e2_cmd = wp.launch(kernel=_fused_backward_preprocess_accumulate_warp_kernel, dim=point_count,
                                        inputs=_e2_inp, outputs=_e2_out, device=str(device), record_cmd=True,
                                        block_dim=_get_block_dim("backward_preprocess"))
                    _C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS[_e2_key] = _e2_cmd
                else:
                    for _i, _v in enumerate(_e2_inp + _e2_out):
                        _e2_cmd.set_param_at_index(_i, _v)
                _e2_cmd.launch()
        else:
                _grad_projected = _backward_projected_means_warp(
                    means3D,
                    preprocess_outputs["radii"],
                    _projmatrix,
                    _viewmatrix,
                    render_grad_points,
                    grad_proj_2d_active,
                    render_grad_depths,
                )
                _grad_means_cov, grad_cov_from_cov2d = _backward_cov2d_warp(
                    means3D,
                    preprocess_outputs["radii"],
                    cov3d_all,
                    _viewmatrix,
                    _tan_fovx,
                    _tan_fovy,
                    focal_x,
                    focal_y,
                    render_grad_conic_opacity[:, :3],
                    grad_conic_2d_active,
                )

                # C3: Fused accumulation — replaces 2 torch.zeros + 3 torch.add + 1 slice-assign
                grad_means3D = torch.empty(means3D.shape, dtype=means3D.dtype, device=device)
                grad_means2D = torch.empty((point_count, 3), dtype=torch.float32, device=device)
                _sh_grad_input = _grad_mean_sh if _has_sh else _grad_projected
                # C4: cache wp.from_torch + Launch object
                _acc_inp = [
                    wp.from_torch(_grad_projected, dtype=wp.vec3),
                    wp.from_torch(_grad_means_cov, dtype=wp.vec3),
                    wp.from_torch(_sh_grad_input, dtype=wp.vec3),
                    int(_has_sh),
                    wp.from_torch(render_grad_points, dtype=wp.vec2),
                ]
                _acc_out = [
                    wp.from_torch(grad_means3D, dtype=wp.vec3),
                    wp.from_torch(grad_means2D, dtype=wp.vec3),
                ]
                _acc_key = (str(device), point_count, int(_has_sh))
                _acc_cmd = _C4_LAUNCH_CACHE_ACCUM.get(_acc_key)
                if _acc_cmd is None:
                    _acc_cmd = wp.launch(kernel=_fused_backward_accumulate_warp_kernel, dim=point_count,
                                         inputs=_acc_inp, outputs=_acc_out, device=str(device), record_cmd=True)
                    _C4_LAUNCH_CACHE_ACCUM[_acc_key] = _acc_cmd
                else:
                    for _i, _v in enumerate(_acc_inp + _acc_out):
                        _acc_cmd.set_param_at_index(_i, _v)
                _acc_cmd.launch()

                # L1: allocate grad_scales/grad_rotations per-branch
                if _cov3D_precomp.numel() != 0:
                    grad_cov3D = grad_cov_from_cov2d
                    grad_scales = torch.zeros_like(_scales)
                    grad_rotations = torch.zeros_like(_rotations)
                elif _scales.numel() != 0 and _rotations.numel() != 0:
                    grad_scales, grad_rotations = _backward_cov3d_from_scale_rotation_warp(
                        _scales,
                        _scale_modifier,
                        _rotations,
                        grad_cov_from_cov2d,
                    )
                else:
                    grad_scales = torch.zeros_like(_scales)
                    grad_rotations = torch.zeros_like(_rotations)

        return grad_means2D, grad_colors, grad_opacities, grad_means3D, grad_cov3D, grad_sh, grad_scales, grad_rotations


    # -----------------------------------------------------------------------------
    # Exported rasterization entry points
    # -----------------------------------------------------------------------------


def rasterize_gaussians(*args: Any):
        _require_warp()
        (
            _background,
            means3D,
            _colors,
            _opacity,
            _scales,
            _rotations,
            _scale_modifier,
            _cov3D_precomp,
            _viewmatrix,
            _projmatrix,
            _tan_fovx,
            _tan_fovy,
            image_height,
            image_width,
            _sh,
            _degree,
            _campos,
            _prefiltered,
        ) = args

        if means3D.ndim != 2 or means3D.shape[1] != 3:
            raise ValueError("means3D must have dimensions (num_points, 3)")

        if means3D.shape[0] == 0:
            return _make_empty_forward_outputs(means3D, image_height, image_width)

        feature_ptr = None

        if _cov3D_precomp.numel() != 0:
            preprocess_outputs = preprocess_gaussians(
                    means3D,
                    _viewmatrix,
                    _projmatrix,
                    image_height,
                    image_width,
                    _tan_fovx,
                    _tan_fovy,
                    cov3D_precomp=_cov3D_precomp,
                    shs=_sh.reshape(means3D.shape[0], -1, NUM_CHANNELS) if _sh.numel() != 0 else None,
                    degree=_degree,
                    campos=_campos,
                    colors_precomp=_colors if _colors.numel() != 0 else None,
                    opacities=_opacity,
                    prefiltered=_prefiltered,
                )
            feature_ptr = _colors.reshape(means3D.shape[0], NUM_CHANNELS).to(dtype=torch.float32) if _colors.numel() != 0 else preprocess_outputs["rgb"]

        if _cov3D_precomp.numel() == 0 and _scales.numel() != 0 and _rotations.numel() != 0:
            preprocess_outputs = preprocess_gaussians(
                    means3D,
                    _viewmatrix,
                    _projmatrix,
                    image_height,
                    image_width,
                    _tan_fovx,
                    _tan_fovy,
                    scales=_scales,
                    rotations=_rotations,
                    scale_modifier=_scale_modifier,
                    shs=_sh.reshape(means3D.shape[0], -1, NUM_CHANNELS) if _sh.numel() != 0 else None,
                    degree=_degree,
                    campos=_campos,
                    colors_precomp=_colors if _colors.numel() != 0 else None,
                    opacities=_opacity,
                    prefiltered=_prefiltered,
                )
            feature_ptr = _colors.reshape(means3D.shape[0], NUM_CHANNELS).to(dtype=torch.float32) if _colors.numel() != 0 else preprocess_outputs["rgb"]

        binning_state = _build_binning_state(preprocess_outputs, image_height, image_width)
        out_color, out_depth, out_alpha, gs_per_pixel, weight_per_gs_pixel, x_mu, _n_contrib = _render_tiles_warp(
                preprocess_outputs,
                binning_state,
                feature_ptr,
                _background.to(dtype=torch.float32),
                image_height,
                image_width,
            )

        geom_buffer, binning_buffer, img_buffer = _pack_forward_aux_buffers(preprocess_outputs, binning_state, _n_contrib)
        return (
            binning_state["num_rendered"],
            out_color,
            out_depth,
            out_alpha,
            preprocess_outputs["radii"],
            geom_buffer,
            binning_buffer,
            img_buffer,
            preprocess_outputs["proj_2d"],
            preprocess_outputs["conic_2d"],
            preprocess_outputs["conic_2d_inv"],
            gs_per_pixel,
            weight_per_gs_pixel,
            x_mu,
        )


def mark_visible(*args: Any):
    _require_warp()
    means3D, viewmatrix, projmatrix = args
    points = means3D.contiguous()

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("means3D must have dimensions (num_points, 3)")

    if points.shape[0] == 0:
        return torch.empty((0,), dtype=torch.bool, device=points.device)
    visible, _p_proj, _p_view_z = _project_visible_points_warp(points, viewmatrix, projmatrix)
    return visible


def rasterize_gaussians_backward(*args: Any):
    _require_warp()
    return _rasterize_gaussians_backward_python(*args)
