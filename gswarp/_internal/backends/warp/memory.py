from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any

import torch
import warp as wp

from gswarp._stream import (
    current_execution_context,
    resolve_execution_context,
    submission_guard,
)
from ...._tuning import normalize_device as _normalize_runtime_device

from .binning_kernels import _gather_i32_by_index_warp_kernel, _pack_binning_keys_warp_kernel

_MAX_WORKSPACE_CACHE_STREAMS = 4
_MAX_WORKSPACE_CACHE_DEVICES = _MAX_WORKSPACE_CACHE_STREAMS
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



@dataclass(slots=True)
class _WorkspaceSlot:
    device_key: str
    stream_handle: int
    warp_stream: Any | None
    radix_sort: tuple[Any | None, torch.Tensor, Any | None, torch.Tensor] | None = None
    radix_sort_i32: tuple[Any | None, torch.Tensor, Any | None, torch.Tensor] | None = None
    index_gather_i32: tuple[Any | None, torch.Tensor] | None = None
    index_gather_i64: tuple[Any | None, torch.Tensor] | None = None
    scan_i32: tuple[Any | None, torch.Tensor, int] | None = None
    project_visible: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    sequence: torch.Tensor | None = None

    @property
    def key(self) -> tuple[str, int]:
        return self.device_key, self.stream_handle

    def tensor_bytes(self) -> int:
        return sum(
            _cached_tensor_bytes(value)
            for value in (
                self.radix_sort,
                self.radix_sort_i32,
                self.index_gather_i32,
                self.index_gather_i64,
                self.scan_i32,
                self.project_visible,
                self.sequence,
            )
        )


_WORKSPACE_SLOT_CACHE: OrderedDict[tuple[str, int], _WorkspaceSlot] = OrderedDict()
_WORKSPACE_SLOT_LOCK = RLock()
_WORKSPACE_EVICTIONS = 0
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
    with _WORKSPACE_SLOT_LOCK:
        slots = tuple(_WORKSPACE_SLOT_CACHE.values())
    launch_entries = sum(len(cache) for cache in _LAUNCH_CACHES)
    workspace_bytes = sum(slot.tensor_bytes() for slot in slots)
    workspace_fields = {
        "_RADIX_SORT_BUFFER_CACHE": "radix_sort",
        "_RADIX_SORT_I32_BUFFER_CACHE": "radix_sort_i32",
        "_INDEX_GATHER_I32_BUFFER_CACHE": "index_gather_i32",
        "_INDEX_GATHER_I64_BUFFER_CACHE": "index_gather_i64",
        "_SCAN_I32_BUFFER_CACHE": "scan_i32",
        "_PROJECT_VISIBLE_BUFFER_CACHE": "project_visible",
        "_SEQUENCE_BUFFER_CACHE": "sequence",
    }
    by_cache = {
        name: {
            "entries": sum(getattr(slot, field) is not None for slot in slots),
            "tensor_bytes": sum(
                _cached_tensor_bytes(getattr(slot, field)) for slot in slots
            ),
        }
        for name, field in workspace_fields.items()
    }
    by_cache.update(
        {
            name: {
                "entries": len(cache),
                "tensor_bytes": sum(
                    _cached_tensor_bytes(value) for value in cache.values()
                ),
            }
            for name, cache in globals().items()
            if isinstance(cache, _BoundedCache)
        }
    )
    return {
        "workspace_entries": len(slots),
        "workspace_tensor_bytes": workspace_bytes,
        "launch_entries": launch_entries,
        "workspace_device_limit": _MAX_WORKSPACE_CACHE_DEVICES,
        "workspace_stream_limit": _MAX_WORKSPACE_CACHE_STREAMS,
        "workspace_evictions": _WORKSPACE_EVICTIONS,
        "workspace_by_stream": {
            f"{slot.device_key}@{slot.stream_handle}": {
                "tensor_bytes": slot.tensor_bytes()
            }
            for slot in slots
        },
        "launch_entry_limit": _MAX_LAUNCH_CACHE_ENTRIES,
        "by_cache": by_cache,
    }


def _clear_cache_entries(caches, device: torch.device | str | None) -> None:
    if device is None:
        for cache in caches:
            cache.clear()
        return
    device_key = str(_normalize_runtime_device(device))
    for cache in caches:
        for key in tuple(cache):
            cache_device = _cache_key_device(key)
            if cache_device == device_key:
                del cache[key]


def _cache_key_device(key: Any) -> str | None:
    values = key if isinstance(key, tuple) else (key,)
    for value in values:
        if isinstance(value, str) and (
            value == "cpu" or value == "cuda" or value.startswith("cuda:")
        ):
            return str(_normalize_runtime_device(value))
    return None


def _clear_workspace_slots(device: torch.device | str | None) -> None:
    device_key = None if device is None else str(_normalize_runtime_device(device))
    with _WORKSPACE_SLOT_LOCK:
        for key in tuple(_WORKSPACE_SLOT_CACHE):
            if device_key is None or key[0] == device_key:
                del _WORKSPACE_SLOT_CACHE[key]


def _workspace_device_keys() -> set[str]:
    with _WORKSPACE_SLOT_LOCK:
        return {key[0] for key in _WORKSPACE_SLOT_CACHE}


def _launch_device_keys(caches) -> set[str]:
    return {
        device_key
        for cache in caches
        for key in cache
        if (device_key := _cache_key_device(key)) is not None
    }


def _synchronize_workspace_slots(device_key: str) -> None:
    with _WORKSPACE_SLOT_LOCK:
        streams = {
            slot.stream_handle: slot.warp_stream
            for slot in _WORKSPACE_SLOT_CACHE.values()
            if slot.device_key == device_key and slot.warp_stream is not None
        }
    for stream in streams.values():
        wp.synchronize_stream(stream)


def clear_common_warp_caches(device: torch.device | str | None = None) -> None:
    device_keys = (
        {str(_normalize_runtime_device(device))}
        if device is not None
        else _workspace_device_keys() | _launch_device_keys(_COMMON_LAUNCH_CACHES)
    )
    for device_key in sorted(device_keys):
        with submission_guard(device_key):
            _synchronize_workspace_slots(device_key)
            _clear_workspace_slots(device_key)
            _clear_cache_entries(_COMMON_LAUNCH_CACHES, device_key)


def clear_flow_warp_caches(device: torch.device | str | None = None) -> None:
    caches = (_C4_LAUNCH_CACHE_FLOW_GRAD,)
    device_keys = (
        {str(_normalize_runtime_device(device))}
        if device is not None
        else _launch_device_keys(caches)
    )
    for device_key in sorted(device_keys):
        with submission_guard(device_key):
            _clear_cache_entries(caches, device_key)



def _can_use_warp_scalar_alloc(device: torch.device | str) -> bool:
    return wp is not None and _normalize_runtime_device(device).type == "cuda"


def _get_runtime_warp_device(device: torch.device | str) -> str:
    runtime_device = _normalize_runtime_device(device)
    if runtime_device.type == "cuda" and runtime_device.index is None and torch.cuda.is_available():
        runtime_device = torch.device("cuda", torch.cuda.current_device())
    return str(runtime_device)


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


def _get_workspace_slot(device: torch.device | str) -> _WorkspaceSlot:
    global _WORKSPACE_EVICTIONS

    context = current_execution_context()
    if context is not None:
        requested_device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        if requested_device != context.device:
            raise RuntimeError(
                f"active workspace device is {context.device}, requested {requested_device}"
            )
        key = context.workspace_key
        warp_stream = context.warp_stream
    else:
        runtime_device = _normalize_runtime_device(device)
        if runtime_device.type == "cuda":
            context = resolve_execution_context(runtime_device)
        else:
            context = None
        if context is not None:
            key = context.workspace_key
            warp_stream = context.warp_stream
        else:
            key = (str(runtime_device), 0)
            warp_stream = None

    slot = _WORKSPACE_SLOT_CACHE.get(key)
    if slot is not None:
        return slot

    with _WORKSPACE_SLOT_LOCK:
        slot = _WORKSPACE_SLOT_CACHE.get(key)
        if slot is not None:
            return slot

        same_device_keys = [
            slot_key
            for slot_key in _WORKSPACE_SLOT_CACHE
            if slot_key[0] == key[0]
        ]
        if len(same_device_keys) >= _MAX_WORKSPACE_CACHE_STREAMS:
            evicted = _WORKSPACE_SLOT_CACHE.pop(same_device_keys[0])
            if evicted.warp_stream is not None:
                wp.synchronize_stream(evicted.warp_stream)
            _WORKSPACE_EVICTIONS += 1

        slot = _WorkspaceSlot(
            device_key=key[0],
            stream_handle=key[1],
            warp_stream=warp_stream,
        )
        _WORKSPACE_SLOT_CACHE[key] = slot
        return slot


def _get_radix_sort_buffers(device: torch.device, required_count: int):
    slot = _get_workspace_slot(device)
    cached = slot.radix_sort
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
    slot.radix_sort = (key_warp, key_buffer, value_warp, value_buffer)
    return key_buffer, value_buffer, key_warp, value_warp


def _get_radix_sort_i32_buffers(device: torch.device, required_count: int):
    slot = _get_workspace_slot(device)
    cached = slot.radix_sort_i32
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
    slot.radix_sort_i32 = (key_warp, key_buffer, value_warp, value_buffer)
    return key_buffer, value_buffer, key_warp, value_warp


def _get_index_gather_i32_buffer(device: torch.device, required_count: int):
    slot = _get_workspace_slot(device)
    cached = slot.index_gather_i32
    if cached is not None:
        warp_buffer, tensor_buffer = cached
        if tensor_buffer.numel() >= required_count:
            return tensor_buffer[:required_count], warp_buffer

    if _can_use_warp_scalar_alloc(device):
        warp_buffer, buffer = _allocate_warp_scalar_array(required_count, torch.int32, device)
    else:
        warp_buffer = None
        buffer = torch.empty((required_count,), dtype=torch.int32, device=device)
    slot.index_gather_i32 = (warp_buffer, buffer)
    return buffer, warp_buffer


def _get_index_gather_i64_buffer(device: torch.device, required_count: int):
    slot = _get_workspace_slot(device)
    cached = slot.index_gather_i64
    if cached is not None:
        warp_buffer, tensor_buffer = cached
        if tensor_buffer.numel() >= required_count:
            return tensor_buffer[:required_count], warp_buffer

    if _can_use_warp_scalar_alloc(device):
        warp_buffer, buffer = _allocate_warp_scalar_array(required_count, torch.int64, device)
    else:
        warp_buffer = None
        buffer = torch.empty((required_count,), dtype=torch.int64, device=device)
    slot.index_gather_i64 = (warp_buffer, buffer)
    return buffer, warp_buffer


def _get_scan_i32_buffer(device: torch.device, required_count: int):
    slot = _get_workspace_slot(device)
    cached = slot.scan_i32
    if cached is not None:
        warp_buffer, tensor_buffer, warp_size = cached
        if tensor_buffer.numel() >= required_count:
            sliced = tensor_buffer[:required_count]
            if warp_buffer is not None and warp_size == required_count:
                return sliced, warp_buffer
            new_warp = wp.from_torch(sliced, dtype=wp.int32) if _can_use_warp_scalar_alloc(device) else None
            slot.scan_i32 = (new_warp, tensor_buffer, required_count)
            return sliced, new_warp

    if _can_use_warp_scalar_alloc(device):
        warp_buffer, buffer = _allocate_warp_scalar_array(required_count, torch.int32, device)
    else:
        warp_buffer = None
        buffer = torch.empty((required_count,), dtype=torch.int32, device=device)
    slot.scan_i32 = (warp_buffer, buffer, required_count)
    return buffer, warp_buffer


def _get_project_visible_buffers(device: torch.device, point_count: int):
    slot = _get_workspace_slot(device)
    cached = slot.project_visible
    if cached is not None:
        visible_mask, p_proj, p_view_z = cached
        if visible_mask.shape[0] >= point_count and p_proj.shape[0] >= point_count and p_view_z.shape[0] >= point_count:
            return visible_mask[:point_count], p_proj[:point_count], p_view_z[:point_count]

    visible_mask = torch.empty((point_count,), dtype=torch.int32, device=device)
    p_proj = torch.empty((point_count, 3), dtype=torch.float32, device=device)
    p_view_z = torch.empty((point_count,), dtype=torch.float32, device=device)
    slot.project_visible = (visible_mask, p_proj, p_view_z)
    return visible_mask, p_proj, p_view_z


def _get_sequence_buffer(device: torch.device, required_count: int):
    slot = _get_workspace_slot(device)
    cached = slot.sequence
    if cached is not None and cached.numel() >= required_count:
        return cached[:required_count]

    sequence = torch.arange(required_count, dtype=torch.int32, device=device)
    slot.sequence = sequence
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


def _warp_radix_sort_pairs_in_place(
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    count: int,
    key_warp=None,
    value_warp=None,
):
    wp.utils.radix_sort_pairs(
        key_warp if key_warp is not None else wp.from_torch(key_buffer, dtype=wp.int64),
        value_warp if value_warp is not None else wp.from_torch(value_buffer, dtype=wp.int32),
        count,
    )
    return key_buffer[:count], value_buffer[:count]


def _warp_radix_sort_i32_pairs_in_place(
    key_buffer: torch.Tensor,
    value_buffer: torch.Tensor,
    count: int,
    key_warp=None,
    value_warp=None,
):
    _key_arr = key_warp if key_warp is not None else wp.from_torch(key_buffer, dtype=wp.int32)
    _val_arr = value_warp if value_warp is not None else wp.from_torch(value_buffer, dtype=wp.int32)
    wp.utils.radix_sort_pairs(
        _key_arr,
        _val_arr,
        count,
    )
    return key_buffer[:count], value_buffer[:count]


__all__ = [name for name in globals() if name.startswith("_") or name.startswith("clear_")]
