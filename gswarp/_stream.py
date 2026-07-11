"""Call-scoped PyTorch/Warp CUDA stream interop."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Iterator

import torch
import warp as wp


@dataclass(frozen=True, slots=True)
class ExecutionContext:
    """Resolved CUDA execution identity for one gswarp submission scope."""

    device: torch.device
    device_index: int
    torch_stream: torch.cuda.Stream
    stream_handle: int
    warp_stream: wp.Stream
    workspace_key: tuple[str, int]


_ACTIVE_EXECUTION_CONTEXT: ContextVar[ExecutionContext | None] = ContextVar(
    "gswarp_active_execution_context", default=None
)
_STATE_LOCK = RLock()
_DEVICE_SUBMISSION_LOCKS: dict[int, RLock] = {}
_WARP_STREAMS: dict[tuple[int, int], wp.Stream] = {}
_EXECUTION_CONTEXTS: dict[tuple[int, int], ExecutionContext] = {}
_BOUND_STREAM_HANDLES: dict[int, int] = {}
_MAX_STREAM_WRAPPERS_PER_DEVICE = 16


def _normalize_cuda_device(device: torch.device | str | None) -> torch.device | None:
    if not torch.cuda.is_available():
        return None
    torch_device = (
        torch.device(device)
        if device is not None
        else torch.device("cuda", torch.cuda.current_device())
    )
    if torch_device.type != "cuda":
        return None
    device_index = (
        torch_device.index
        if torch_device.index is not None
        else torch.cuda.current_device()
    )
    return torch.device("cuda", device_index)


def _resolve_execution_context(
    device: torch.device | str | None,
) -> ExecutionContext | None:
    torch_device = _normalize_cuda_device(device)
    if torch_device is None:
        return None
    device_index = int(torch_device.index)
    torch_stream = torch.cuda.current_stream(device_index)
    stream_handle = int(torch_stream.cuda_stream)
    key = (device_index, stream_handle)
    cached = _EXECUTION_CONTEXTS.get(key)
    if cached is not None:
        return cached
    with _STATE_LOCK:
        cached = _EXECUTION_CONTEXTS.get(key)
        if cached is not None:
            return cached
        warp_stream = _WARP_STREAMS.get(key)
        if warp_stream is None:
            same_device_keys = [
                stream_key
                for stream_key in _WARP_STREAMS
                if stream_key[0] == device_index
            ]
            if len(same_device_keys) >= _MAX_STREAM_WRAPPERS_PER_DEVICE:
                evicted_key = same_device_keys[0]
                del _WARP_STREAMS[evicted_key]
                _EXECUTION_CONTEXTS.pop(evicted_key, None)
            warp_stream = wp.Stream(
                device=str(torch_device), cuda_stream=stream_handle
            )
            _WARP_STREAMS[key] = warp_stream
        resolved = ExecutionContext(
            device=torch_device,
            device_index=device_index,
            torch_stream=torch_stream,
            stream_handle=stream_handle,
            warp_stream=warp_stream,
            workspace_key=(str(torch_device), stream_handle),
        )
        _EXECUTION_CONTEXTS[key] = resolved
        return resolved


def resolve_execution_context(
    device: torch.device | str | None = None,
) -> ExecutionContext | None:
    """Resolve the current PyTorch stream without entering a submission scope."""

    return _resolve_execution_context(device)


def _submission_lock(device_index: int) -> RLock:
    lock = _DEVICE_SUBMISSION_LOCKS.get(device_index)
    if lock is not None:
        return lock
    with _STATE_LOCK:
        lock = _DEVICE_SUBMISSION_LOCKS.get(device_index)
        if lock is None:
            lock = RLock()
            _DEVICE_SUBMISSION_LOCKS[device_index] = lock
        return lock


@contextmanager
def submission_guard(
    device: torch.device | str | None,
) -> Iterator[None]:
    """Serialize host submission or cache mutation for one CUDA device."""

    torch_device = _normalize_cuda_device(device)
    if torch_device is None:
        yield
        return
    with _submission_lock(int(torch_device.index)):
        yield


@contextmanager
def execution_context(
    device: torch.device | str | None = None,
) -> Iterator[ExecutionContext | None]:
    """Bind Warp to the current PyTorch stream for one complete submission."""

    resolved = _resolve_execution_context(device)
    if resolved is None:
        yield None
        return

    lock = _submission_lock(resolved.device_index)
    with lock:
        previous = _ACTIVE_EXECUTION_CONTEXT.get()
        token = _ACTIVE_EXECUTION_CONTEXT.set(resolved)
        try:
            if (
                _BOUND_STREAM_HANDLES.get(resolved.device_index)
                != resolved.stream_handle
            ):
                wp.set_stream(
                    resolved.warp_stream, device=str(resolved.device), sync=False
                )
                _BOUND_STREAM_HANDLES[resolved.device_index] = (
                    resolved.stream_handle
                )
            yield resolved
        finally:
            _ACTIVE_EXECUTION_CONTEXT.reset(token)
            if (
                previous is not None
                and previous.device_index == resolved.device_index
                and previous.stream_handle != resolved.stream_handle
            ):
                wp.set_stream(
                    previous.warp_stream,
                    device=str(previous.device),
                    sync=False,
                )
                _BOUND_STREAM_HANDLES[previous.device_index] = (
                    previous.stream_handle
                )


def current_execution_context(
    device: torch.device | str | None = None,
) -> ExecutionContext | None:
    """Return the active context, validating an optional device."""

    active = _ACTIVE_EXECUTION_CONTEXT.get()
    if active is None:
        return None
    expected = _normalize_cuda_device(device) if device is not None else None
    if expected is not None and expected != active.device:
        raise RuntimeError(
            f"active Warp execution device is {active.device}, requested {expected}"
        )
    return active


def ensure_aligned(device: torch.device | str | None = None) -> None:
    """Compatibility helper for callers not yet using an execution context."""

    resolved = _resolve_execution_context(device)
    if resolved is None:
        return
    with _submission_lock(resolved.device_index):
        if _BOUND_STREAM_HANDLES.get(resolved.device_index) != resolved.stream_handle:
            wp.set_stream(
                resolved.warp_stream, device=str(resolved.device), sync=False
            )
            _BOUND_STREAM_HANDLES[resolved.device_index] = resolved.stream_handle


def torch_launch_array(tensor: torch.Tensor, dtype=None):
    """Wrap a tensor without attaching a second Warp autograd graph."""

    return wp.from_torch(
        tensor,
        dtype=dtype,
        requires_grad=False,
        return_ctype=False,
    )


def set_launch_params(command, values) -> None:
    """Update a recorded launch without repeating Warp's type conversion."""

    packed_values = []
    for value in values:
        to_ctype = getattr(value, "__ctype__", None)
        # Regular Warp arrays own the Torch reference; this descriptor only
        # carries the raw launch metadata retained by the cached command.
        packed_values.append(to_ctype() if to_ctype is not None else value)
    command.set_params_from_ctypes(packed_values)


def clear_execution_stream_cache(
    device: torch.device | str | None = None,
) -> None:
    """Drop cached external-stream wrappers after dependent work is complete."""

    torch_device = _normalize_cuda_device(device) if device is not None else None
    with _STATE_LOCK:
        for key in tuple(_WARP_STREAMS):
            if torch_device is None or key[0] == int(torch_device.index):
                del _WARP_STREAMS[key]
                _EXECUTION_CONTEXTS.pop(key, None)
        if torch_device is None:
            _BOUND_STREAM_HANDLES.clear()
        else:
            _BOUND_STREAM_HANDLES.pop(int(torch_device.index), None)


__all__ = [
    "ExecutionContext",
    "current_execution_context",
    "ensure_aligned",
    "execution_context",
    "resolve_execution_context",
    "submission_guard",
    "torch_launch_array",
    "set_launch_params",
    "clear_execution_stream_cache",
]
