from __future__ import annotations

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
from . import runtime as _runtime
from .binning_kernels import _gather_i32_by_index_warp_kernel, _pack_binning_keys_warp_kernel

_MAX_WORKSPACE_CACHE_DEVICES = 4
_MAX_LAUNCH_CACHE_ENTRIES = 32


class _BoundedCache(dict):
    """Insertion-ordered cache with a fixed number of retained entries."""

    def __init__(self, max_entries: int) -> None:
        super().__init__()
        self.max_entries = max_entries

    def __setitem__(self, key, value) -> None:
        if key not in self and len(self) >= self.max_entries:
            del self[next(iter(self))]
        super().__setitem__(key, value)


_RADIX_SORT_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor, Any | None, torch.Tensor]] = _BoundedCache(_MAX_WORKSPACE_CACHE_DEVICES)
_RADIX_SORT_I32_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor, Any | None, torch.Tensor]] = _BoundedCache(_MAX_WORKSPACE_CACHE_DEVICES)
_INDEX_GATHER_I32_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor]] = _BoundedCache(_MAX_WORKSPACE_CACHE_DEVICES)
_INDEX_GATHER_I64_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor]] = _BoundedCache(_MAX_WORKSPACE_CACHE_DEVICES)
_SCAN_I32_BUFFER_CACHE: dict[str, tuple[Any | None, torch.Tensor, int]] = _BoundedCache(_MAX_WORKSPACE_CACHE_DEVICES)
_PROJECT_VISIBLE_BUFFER_CACHE: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = _BoundedCache(_MAX_WORKSPACE_CACHE_DEVICES)
_SEQUENCE_BUFFER_CACHE: dict[str, torch.Tensor] = _BoundedCache(_MAX_WORKSPACE_CACHE_DEVICES)
_DEPTH_SORT_ORDER_CACHE: dict[str, tuple[torch.Tensor, int]] = _BoundedCache(_MAX_WORKSPACE_CACHE_DEVICES)
_DEPTH_SORT_CHECK_FLAG: dict[str, torch.Tensor] = _BoundedCache(_MAX_WORKSPACE_CACHE_DEVICES)
_C4_LAUNCH_CACHE_SH: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_COV3D: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_RENDER_BWD: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_PROJ_MEANS: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_COV2D: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_ACCUM: dict[tuple[str, int, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS: dict[tuple[str, int, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_SH_V3: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_FWD_SH: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_FWD_PREPROCESS: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_FWD_RENDER: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_BINNING_DUPLICATE: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_FWD_RENDER_TILED256: dict[tuple[Any, ...], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)
_C4_LAUNCH_CACHE_FLOW_GRAD: dict[tuple[str, int], Any] = _BoundedCache(_MAX_LAUNCH_CACHE_ENTRIES)

_WORKSPACE_CACHES = (
    _RADIX_SORT_BUFFER_CACHE, _RADIX_SORT_I32_BUFFER_CACHE, _INDEX_GATHER_I32_BUFFER_CACHE,
    _INDEX_GATHER_I64_BUFFER_CACHE, _SCAN_I32_BUFFER_CACHE, _PROJECT_VISIBLE_BUFFER_CACHE,
    _SEQUENCE_BUFFER_CACHE, _DEPTH_SORT_ORDER_CACHE, _DEPTH_SORT_CHECK_FLAG,
)
_COMMON_LAUNCH_CACHES = (
    _C4_LAUNCH_CACHE_SH, _C4_LAUNCH_CACHE_COV3D, _C4_LAUNCH_CACHE_RENDER_BWD,
    _C4_LAUNCH_CACHE_PROJ_MEANS, _C4_LAUNCH_CACHE_COV2D, _C4_LAUNCH_CACHE_ACCUM,
    _C4_LAUNCH_CACHE_BWD_FUSED_PREPROCESS, _C4_LAUNCH_CACHE_SH_V3, _C4_LAUNCH_CACHE_FWD_SH,
    _C4_LAUNCH_CACHE_FWD_PREPROCESS, _C4_LAUNCH_CACHE_FWD_RENDER,
    _C4_LAUNCH_CACHE_BINNING_DUPLICATE, _C4_LAUNCH_CACHE_FWD_RENDER_TILED256,
)
_LAUNCH_CACHES = _COMMON_LAUNCH_CACHES + (_C4_LAUNCH_CACHE_FLOW_GRAD,)


def _cached_tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, tuple):
        return sum(_cached_tensor_bytes(item) for item in value)
    return 0


def get_warp_cache_report() -> dict[str, Any]:
    """Return bounded-cache occupancy without synchronizing the device."""
    workspace_entries = sum(len(cache) for cache in _WORKSPACE_CACHES)
    launch_entries = sum(len(cache) for cache in _LAUNCH_CACHES)
    workspace_bytes = sum(
        _cached_tensor_bytes(value) for cache in _WORKSPACE_CACHES for value in cache.values()
    )
    return {
        "workspace_entries": workspace_entries,
        "workspace_tensor_bytes": workspace_bytes,
        "launch_entries": launch_entries,
        "workspace_device_limit": _MAX_WORKSPACE_CACHE_DEVICES,
        "launch_entry_limit": _MAX_LAUNCH_CACHE_ENTRIES,
        "by_cache": {
            name: {"entries": len(cache), "tensor_bytes": sum(_cached_tensor_bytes(value) for value in cache.values())}
            for name, cache in globals().items()
            if isinstance(cache, _BoundedCache)
        },
    }


def _clear_cache_entries(caches, device: torch.device | str | None) -> None:
    if device is None:
        for cache in caches:
            cache.clear()
        return
    device_key = str(torch.device(device))
    for cache in caches:
        for key in tuple(cache):
            cache_device = key[0] if isinstance(key, tuple) else key
            if cache_device == device_key:
                del cache[key]


def _synchronize_cache_device(device: torch.device | str | None) -> None:
    if not torch.cuda.is_available():
        return
    if device is None:
        torch.cuda.synchronize()
    elif torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def clear_common_warp_caches(device: torch.device | str | None = None) -> None:
    _synchronize_cache_device(device)
    _clear_cache_entries(_WORKSPACE_CACHES, device)
    _clear_cache_entries(_COMMON_LAUNCH_CACHES, device)


def clear_flow_warp_caches(device: torch.device | str | None = None) -> None:
    _synchronize_cache_device(device)
    _clear_cache_entries((_C4_LAUNCH_CACHE_FLOW_GRAD,), device)



def _can_use_warp_scalar_alloc(device: torch.device | str) -> bool:
    return wp is not None and _normalize_runtime_device(device).type == "cuda"


def _get_runtime_warp_device(device: torch.device | str) -> str:
    runtime_device = _normalize_runtime_device(device)
    auto_tune, _ = _runtime.get_active_auto_tuning_config()
    if not auto_tune or not _runtime._WARP_INITIALIZED:
        return str(runtime_device)
    return str(_runtime.get_runtime_tuning_report(runtime_device)["device"])


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
    # Zero-element warp CUDA arrays have a host/null pointer; older PyTorch
    # raises "pointer resides on host memory" in wp.to_torch.  Construct the
    # empty torch tensor directly to maintain compatibility.
    _numel = shape if isinstance(shape, int) else 1
    if not isinstance(shape, int):
        for s in shape:
            _numel *= s
    if _numel == 0:
        torch_shape = (shape,) if isinstance(shape, int) else shape
        return warp_array, torch.empty(torch_shape, dtype=dtype, device=warp_device)
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
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if tensor.requires_grad:
        tensor = tensor.detach()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


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


__all__ = [name for name in globals() if name.startswith("_") or name.startswith("clear_")]
