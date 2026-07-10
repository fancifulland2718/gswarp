"""Align Warp's CUDA stream with PyTorch's to avoid implicit synchronization.

PyTorch defaults to legacy CUDA stream 0, while Warp creates its own
non-default stream.  Legacy stream 0 has special semantics: any operation
on stream 0 implicitly waits for ALL other streams and vice versa.
When Warp kernels and PyTorch ops interleave (which happens every training
iteration), each stream switch becomes an implicit cudaDeviceSynchronize(),
draining the GPU pipeline.

Fix: set Warp's active stream to the same CUDA stream PyTorch is using.
This is checked at every public CUDA entry point. The no-op fast path keeps
the current device/stream binding unchanged, while a stream switch or a
different CUDA device is rebound before launching Warp work.
"""

import warp as wp
import torch

_active_streams: dict[int, int] = {}
_warp_streams: dict[tuple[int, int], wp.Stream] = {}


def ensure_aligned(device: torch.device | str | None = None) -> None:
    """Make Warp use PyTorch's current stream for *device*.

    A single process can use multiple CUDA devices and non-default PyTorch
    streams. Warp commands must follow the stream active at each call, not the
    one that happened to be current during the first gswarp invocation.
    """
    if not torch.cuda.is_available():
        return
    torch_device = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    if torch_device.type != "cuda":
        return
    device_index = torch_device.index if torch_device.index is not None else torch.cuda.current_device()
    torch_stream = torch.cuda.current_stream(device_index)
    stream_handle = int(torch_stream.cuda_stream)
    if _active_streams.get(device_index) == stream_handle:
        return

    key = (device_index, stream_handle)
    warp_stream = _warp_streams.get(key)
    if warp_stream is None:
        warp_stream = wp.Stream(device=f"cuda:{device_index}", cuda_stream=stream_handle)
        _warp_streams[key] = warp_stream
    wp.set_stream(warp_stream)
    _active_streams[device_index] = stream_handle
