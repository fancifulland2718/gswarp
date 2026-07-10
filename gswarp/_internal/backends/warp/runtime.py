from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch
import warp as wp

from ...._tuning import (
    normalize_device as _normalize_runtime_device,
    query_device_info as _query_runtime_device_info,
    query_sm_properties as _query_sm_properties,
    register_kernel_class as _register_kernel_class,
    get_tuned_block_dim,
    initialize_tuning as _tuning_initialize,
    FAMILY_COMPUTE,
    FAMILY_WARP_SPECIALIZED,
)

from .constants import *

DEFAULT_BACKWARD_MODE = "manual"
_BACKWARD_MODE = DEFAULT_BACKWARD_MODE
DEFAULT_BINNING_SORT_MODE = "warp_depth_stable_tile"
_BINNING_SORT_MODE = DEFAULT_BINNING_SORT_MODE
_COMPUTE_DEPTH = os.environ.get("GSWARP_COMPUTE_DEPTH", "1") != "0"
_RUNTIME_TUNING_CACHE: dict[str, dict[str, Any]] = {}
_RUNTIME_TUNING_LOGGED_DEVICES: set[str] = set()
_RUNTIME_BINNING_POLICY_STATE: dict[str, dict[str, Any]] = {}
_AUTO_TUNE_ENABLED = True
_AUTO_TUNE_VERBOSE = True
_WARP_INITIALIZED = False


@dataclass(frozen=True, slots=True)
class ExecutionOptions:
    """Immutable runtime choices owned by one frontend invocation."""

    backward_mode: str
    binning_sort_mode: str
    compute_depth: bool
    auto_tune: bool
    auto_tune_verbose: bool
    compute_flow_aux: bool | None = None


_ACTIVE_EXECUTION_OPTIONS: ContextVar[ExecutionOptions | None] = ContextVar(
    "gswarp_active_execution_options", default=None
)


@contextmanager
def execution_options(options: ExecutionOptions):
    token = _ACTIVE_EXECUTION_OPTIONS.set(options)
    try:
        yield options
    finally:
        _ACTIVE_EXECUTION_OPTIONS.reset(token)


def get_active_compute_depth() -> bool:
    options = _ACTIVE_EXECUTION_OPTIONS.get()
    return _COMPUTE_DEPTH if options is None else options.compute_depth


def get_active_binning_sort_mode() -> str:
    options = _ACTIVE_EXECUTION_OPTIONS.get()
    return _BINNING_SORT_MODE if options is None else options.binning_sort_mode


def get_active_auto_tuning_config() -> tuple[bool, bool]:
    options = _ACTIVE_EXECUTION_OPTIONS.get()
    if options is None:
        return _AUTO_TUNE_ENABLED, _AUTO_TUNE_VERBOSE
    return options.auto_tune, options.auto_tune_verbose


def get_active_compute_flow_aux(default: bool) -> bool:
    options = _ACTIVE_EXECUTION_OPTIONS.get()
    if options is None or options.compute_flow_aux is None:
        return default
    return options.compute_flow_aux

_ANSI_RESET = "\033[0m"
_ANSI_HEADER = "\033[96m"
_ANSI_DEVICE = "\033[92m"
_ANSI_MEMORY = "\033[93m"
_ANSI_TILE = "\033[95m"
_ANSI_STRATEGY = "\033[94m"
_ANSI_HW = "\033[33m"

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
    """Build the rasterizer runtime-tuning report, delegating block_dim computation
    to the shared ``_tuning`` module so that SSIM and KNN kernels on the same
    device inherit the plan without redundant device queries."""
    runtime_device = _normalize_runtime_device(device)
    device_props, free_memory, total_memory, runtime_device = _query_runtime_device_info(runtime_device)
    device_key = str(runtime_device)

    recommended_tile_x, recommended_tile_y = _recommend_tile_shape(device_props, free_memory)
    recommended_binning_sort_mode, binning_policy = _recommend_binning_sort_mode(runtime_device, device_props, free_memory)

    # Delegate block_dim planning to the shared tuning module.
    # This also populates the per-device plan that get_tuned_block_dim() uses,
    # so subsequent calls from ssim.py / knn.py on this device are free.
    tuning_report = _tuning_initialize(runtime_device, verbose=False)
    block_dim_plan = tuning_report.get("block_dim_plan", {})
    sm_props = tuning_report.get("sm_properties", {})
    occupancy_snapshot = tuning_report.get("occupancy_snapshot", {})

    report = {
        "device": device_key,
        "device_type": runtime_device.type,
        "device_name": getattr(device_props, "name", runtime_device.type.upper()),
        "compute_capability": None if device_props is None else f"{device_props.major}.{device_props.minor}",
        "sm_count": None if device_props is None else int(device_props.multi_processor_count),
        "warp_size": None if device_props is None else int(getattr(device_props, "warp_size", 32)),
        "free_memory_bytes": None if free_memory is None else int(free_memory),
        "total_memory_bytes": None if total_memory is None else int(total_memory),
        "sm_properties": sm_props,
        "current_tile": (BLOCK_X, BLOCK_Y),
        "recommended_tile": (recommended_tile_x, recommended_tile_y),
        "tile_runtime_mutable": False,
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
    updated["warp_size"] = None if device_props is None else int(getattr(device_props, "warp_size", 32))
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
        # Drive the shared tuning module first (verbose=True so it prints
        # block_dim / occupancy info on first call for this device).
        _tuning_initialize(runtime_device, verbose=verbose)
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


def is_available() -> bool:
    return wp is not None


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
    auto_tune, auto_tune_verbose = get_active_auto_tuning_config()
    if auto_tune:
        initialize_runtime_tuning(verbose=auto_tune_verbose)
    else:
        wp.init()
    _WARP_INITIALIZED = True


__all__ = [
    "get_default_parameter_info",
    "get_runtime_auto_tuning_config",
    "set_runtime_auto_tuning",
    "get_compute_depth",
    "set_compute_depth",
    "initialize_runtime_tuning",
    "get_runtime_tuning_report",
    "is_available",
    "get_backward_mode",
    "set_backward_mode",
    "get_binning_sort_mode",
    "set_binning_sort_mode",
    "ExecutionOptions",
    "execution_options",
    "get_active_compute_depth",
    "get_active_binning_sort_mode",
    "get_active_auto_tuning_config",
    "get_active_compute_flow_aux",
]
