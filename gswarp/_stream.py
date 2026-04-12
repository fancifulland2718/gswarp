"""Align Warp's CUDA stream with PyTorch's to avoid implicit synchronization.

PyTorch defaults to legacy CUDA stream 0, while Warp creates its own
non-default stream.  Legacy stream 0 has special semantics: any operation
on stream 0 implicitly waits for ALL other streams and vice versa.
When Warp kernels and PyTorch ops interleave (which happens every training
iteration), each stream switch becomes an implicit cudaDeviceSynchronize(),
draining the GPU pipeline.

Fix: set Warp's active stream to the same CUDA stream PyTorch is using.
This is called once lazily on the first gswarp kernel invocation.
"""

import warp as wp
import torch

_aligned = False


def ensure_aligned():
    """Make Warp use PyTorch's current CUDA stream (one-time, idempotent)."""
    global _aligned
    if _aligned:
        return
    if not torch.cuda.is_available():
        return
    torch_stream = torch.cuda.current_stream()
    wp.set_stream(wp.Stream(device="cuda:0", cuda_stream=torch_stream.cuda_stream))
    _aligned = True
