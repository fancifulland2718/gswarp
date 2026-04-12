"""Warp kernel implementation of simple_knn (Morton-code-based 3-NN mean distance).

Algorithm (mirrors original CUDA exactly):
1. Compute axis-aligned bounding box (min/max) of all points
2. Encode each point as a 30-bit Morton code (Z-order curve)
3. Sort points by Morton code (preserves spatial locality)
4. Gather sorted points into contiguous SoA layout
5. Compute AABB for each box of 1024 sorted points
6. For each point: find 3 nearest neighbours using box pruning
7. Return mean of the 3 nearest-neighbour squared distances per point

GPU kernels use NVIDIA Warp.  torch is used only for:
  - global min/max reduction (equivalent to CUB DeviceReduce)
  - GPU memory allocation  (equivalent to cudaMalloc / thrust::device_vector)
No C++/CUDA compilation required.
"""

import torch
import warp as wp

from ._stream import ensure_aligned
from ._tuning import register_kernel_class, get_tuned_block_dim, FAMILY_COMPUTE, FAMILY_MEMORY, FAMILY_ATOMIC

wp.init()

__all__ = ["distCUDA2"]

# Register KNN kernel classes with estimated register usage.
# _box_mean_dist is the heaviest (nested loop over boxes for 3-NN search).
register_kernel_class("knn_morton", 32, FAMILY_COMPUTE)
register_kernel_class("knn_gather", 32, FAMILY_MEMORY)
register_kernel_class("knn_box_minmax", 32, FAMILY_ATOMIC)
register_kernel_class("knn_box_dist", 64, FAMILY_COMPUTE)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BOX_SIZE: int = 1024

FLT_MAX_VAL = wp.constant(wp.float32(3.4028235e+38))

# ---------------------------------------------------------------------------
# Warp helper functions
# ---------------------------------------------------------------------------


@wp.func
def prep_morton(x: wp.uint32) -> wp.uint32:
    x = (x | (x << wp.uint32(16))) & wp.uint32(50331903)   # 0x030000FF
    x = (x | (x << wp.uint32(8))) & wp.uint32(50393103)    # 0x0300F00F
    x = (x | (x << wp.uint32(4))) & wp.uint32(51130563)    # 0x030C30C3
    x = (x | (x << wp.uint32(2))) & wp.uint32(153391689)   # 0x09249249
    return x


@wp.func
def coord_to_morton(
    px: wp.float32, py: wp.float32, pz: wp.float32,
    min_x: wp.float32, min_y: wp.float32, min_z: wp.float32,
    max_x: wp.float32, max_y: wp.float32, max_z: wp.float32,
) -> wp.uint32:
    scale = wp.float32(1023.0)
    nx = wp.uint32((px - min_x) / (max_x - min_x) * scale)
    ny = wp.uint32((py - min_y) / (max_y - min_y) * scale)
    nz = wp.uint32((pz - min_z) / (max_z - min_z) * scale)
    return prep_morton(nx) | (prep_morton(ny) << wp.uint32(1)) | (prep_morton(nz) << wp.uint32(2))


@wp.func
def dist_box_point(
    box_min_x: wp.float32, box_min_y: wp.float32, box_min_z: wp.float32,
    box_max_x: wp.float32, box_max_y: wp.float32, box_max_z: wp.float32,
    px: wp.float32, py: wp.float32, pz: wp.float32,
) -> wp.float32:
    dx = wp.float32(0.0)
    dy = wp.float32(0.0)
    dz = wp.float32(0.0)
    if px < box_min_x or px > box_max_x:
        d1 = wp.abs(px - box_min_x)
        d2 = wp.abs(px - box_max_x)
        if d1 < d2:
            dx = d1
        else:
            dx = d2
    if py < box_min_y or py > box_max_y:
        d1 = wp.abs(py - box_min_y)
        d2 = wp.abs(py - box_max_y)
        if d1 < d2:
            dy = d1
        else:
            dy = d2
    if pz < box_min_z or pz > box_max_z:
        d1 = wp.abs(pz - box_min_z)
        d2 = wp.abs(pz - box_max_z)
        if d1 < d2:
            dz = d1
        else:
            dz = d2
    return dx * dx + dy * dy + dz * dz


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------


@wp.kernel
def _coord2morton_kernel(
    points: wp.array(dtype=wp.float32, ndim=2),
    min_x: wp.float32, min_y: wp.float32, min_z: wp.float32,
    max_x: wp.float32, max_y: wp.float32, max_z: wp.float32,
    codes: wp.array(dtype=wp.uint32),
):
    idx = wp.tid()
    codes[idx] = coord_to_morton(
        points[idx, 0], points[idx, 1], points[idx, 2],
        min_x, min_y, min_z, max_x, max_y, max_z,
    )


@wp.kernel
def _gather_sorted_soa_kernel(
    points: wp.array(dtype=wp.float32, ndim=2),
    sort_idx: wp.array(dtype=wp.int32),
    sorted_x: wp.array(dtype=wp.float32),
    sorted_y: wp.array(dtype=wp.float32),
    sorted_z: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    orig = sort_idx[i]
    sorted_x[i] = points[orig, 0]
    sorted_y[i] = points[orig, 1]
    sorted_z[i] = points[orig, 2]


@wp.kernel
def _box_min_max_kernel(
    sorted_x: wp.array(dtype=wp.float32),
    sorted_y: wp.array(dtype=wp.float32),
    sorted_z: wp.array(dtype=wp.float32),
    box_min_x: wp.array(dtype=wp.float32),
    box_min_y: wp.array(dtype=wp.float32),
    box_min_z: wp.array(dtype=wp.float32),
    box_max_x: wp.array(dtype=wp.float32),
    box_max_y: wp.array(dtype=wp.float32),
    box_max_z: wp.array(dtype=wp.float32),
    P: wp.int32,
):
    idx = wp.tid()
    if idx >= P:
        return
    box_id = idx / 1024
    px = sorted_x[idx]
    py = sorted_y[idx]
    pz = sorted_z[idx]
    wp.atomic_min(box_min_x, box_id, px)
    wp.atomic_min(box_min_y, box_id, py)
    wp.atomic_min(box_min_z, box_id, pz)
    wp.atomic_max(box_max_x, box_id, px)
    wp.atomic_max(box_max_y, box_id, py)
    wp.atomic_max(box_max_z, box_id, pz)


@wp.kernel
def _box_mean_dist_kernel(
    sorted_x: wp.array(dtype=wp.float32),
    sorted_y: wp.array(dtype=wp.float32),
    sorted_z: wp.array(dtype=wp.float32),
    sort_idx: wp.array(dtype=wp.int32),
    box_min_x: wp.array(dtype=wp.float32),
    box_min_y: wp.array(dtype=wp.float32),
    box_min_z: wp.array(dtype=wp.float32),
    box_max_x: wp.array(dtype=wp.float32),
    box_max_y: wp.array(dtype=wp.float32),
    box_max_z: wp.array(dtype=wp.float32),
    num_boxes: wp.int32,
    P: wp.int32,
    dists: wp.array(dtype=wp.float32),
):
    idx = wp.tid()
    if idx >= P:
        return

    px = sorted_x[idx]
    py = sorted_y[idx]
    pz = sorted_z[idx]

    # -- seed with local 3-NN (idx ± 3) --
    best0 = FLT_MAX_VAL
    best1 = FLT_MAX_VAL
    best2 = FLT_MAX_VAL

    lo = idx - 3
    if lo < 0:
        lo = 0
    hi = idx + 3
    if hi > P - 1:
        hi = P - 1

    for i in range(lo, hi + 1):
        if i == idx:
            continue
        dx = px - sorted_x[i]
        dy = py - sorted_y[i]
        dz = pz - sorted_z[i]
        d = dx * dx + dy * dy + dz * dz
        if d < best0:
            best2 = best1
            best1 = best0
            best0 = d
        elif d < best1:
            best2 = best1
            best1 = d
        elif d < best2:
            best2 = d

    reject = best2

    # -- full box-pruned 3-NN search --
    best0 = FLT_MAX_VAL
    best1 = FLT_MAX_VAL
    best2 = FLT_MAX_VAL

    for b in range(num_boxes):
        bd = dist_box_point(
            box_min_x[b], box_min_y[b], box_min_z[b],
            box_max_x[b], box_max_y[b], box_max_z[b],
            px, py, pz,
        )
        if bd > reject or bd > best2:
            continue

        box_start = b * 1024
        box_end = (b + 1) * 1024
        if box_end > P:
            box_end = P

        for i in range(box_start, box_end):
            if i == idx:
                continue
            dx = px - sorted_x[i]
            dy = py - sorted_y[i]
            dz = pz - sorted_z[i]
            d = dx * dx + dy * dy + dz * dz
            if d < best0:
                best2 = best1
                best1 = best0
                best0 = d
            elif d < best1:
                best2 = best1
                best1 = d
            elif d < best2:
                best2 = d

    dists[sort_idx[idx]] = (best0 + best1 + best2) / 3.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def distCUDA2(points: torch.Tensor) -> torch.Tensor:
    """Compute mean squared distance to 3 nearest neighbours for each point.

    Drop-in replacement for ``simple_knn._C.distCUDA2``.

    Parameters
    ----------
    points : torch.Tensor
        Shape ``(P, 3)``, dtype ``float32``, on a CUDA device.

    Returns
    -------
    torch.Tensor
        Shape ``(P,)``, dtype ``float32``, on the same device.
    """
    ensure_aligned()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected (P, 3) tensor, got {tuple(points.shape)}")

    P = points.shape[0]
    device = points.device

    if P == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    if P == 1:
        return torch.zeros((1,), dtype=torch.float32, device=device)

    points = points.contiguous().float()
    dev_str = str(device)

    # Step 1: compute global AABB
    pt_min = points.min(dim=0).values
    pt_max = points.max(dim=0).values
    min_x, min_y, min_z = pt_min[0].item(), pt_min[1].item(), pt_min[2].item()
    max_x, max_y, max_z = pt_max[0].item(), pt_max[1].item(), pt_max[2].item()

    eps = 1e-7
    if max_x - min_x < eps:
        max_x = min_x + eps
    if max_y - min_y < eps:
        max_y = min_y + eps
    if max_z - min_z < eps:
        max_z = min_z + eps

    # Step 2: Morton codes
    codes = torch.empty(2 * P, dtype=torch.int32, device=device)
    wp_points = wp.from_torch(points)

    wp.launch(
        _coord2morton_kernel,
        dim=P,
        inputs=[
            wp_points,
            wp.float32(min_x), wp.float32(min_y), wp.float32(min_z),
            wp.float32(max_x), wp.float32(max_y), wp.float32(max_z),
            wp.from_torch(codes[:P], dtype=wp.uint32),
        ],
        device=dev_str,
        block_dim=get_tuned_block_dim("knn_morton", device),
    )

    # Step 3: radix sort by Morton code
    sort_buf = torch.empty(2 * P, dtype=torch.int32, device=device)
    sort_buf[:P] = torch.arange(P, dtype=torch.int32, device=device)
    wp.utils.radix_sort_pairs(
        wp.from_torch(codes, dtype=wp.int32),
        wp.from_torch(sort_buf, dtype=wp.int32),
        P,
    )

    sort_indices = sort_buf[:P].contiguous()
    wp_sort_idx = wp.from_torch(sort_indices, dtype=wp.int32)

    # Step 4: gather into sorted SoA layout
    sorted_x = torch.empty(P, dtype=torch.float32, device=device)
    sorted_y = torch.empty(P, dtype=torch.float32, device=device)
    sorted_z = torch.empty(P, dtype=torch.float32, device=device)
    wp_sx = wp.from_torch(sorted_x)
    wp_sy = wp.from_torch(sorted_y)
    wp_sz = wp.from_torch(sorted_z)

    wp.launch(
        _gather_sorted_soa_kernel,
        dim=P,
        inputs=[wp_points, wp_sort_idx, wp_sx, wp_sy, wp_sz],
        device=dev_str,
        block_dim=get_tuned_block_dim("knn_gather", device),
    )

    # Step 5: compute AABB per box
    num_boxes = (P + BOX_SIZE - 1) // BOX_SIZE
    box_min_x_t = torch.full((num_boxes,), float('inf'), dtype=torch.float32, device=device)
    box_min_y_t = torch.full((num_boxes,), float('inf'), dtype=torch.float32, device=device)
    box_min_z_t = torch.full((num_boxes,), float('inf'), dtype=torch.float32, device=device)
    box_max_x_t = torch.full((num_boxes,), float('-inf'), dtype=torch.float32, device=device)
    box_max_y_t = torch.full((num_boxes,), float('-inf'), dtype=torch.float32, device=device)
    box_max_z_t = torch.full((num_boxes,), float('-inf'), dtype=torch.float32, device=device)

    wp.launch(
        _box_min_max_kernel,
        dim=P,
        inputs=[
            wp_sx, wp_sy, wp_sz,
            wp.from_torch(box_min_x_t), wp.from_torch(box_min_y_t), wp.from_torch(box_min_z_t),
            wp.from_torch(box_max_x_t), wp.from_torch(box_max_y_t), wp.from_torch(box_max_z_t),
            wp.int32(P),
        ],
        device=dev_str,
        block_dim=get_tuned_block_dim("knn_box_minmax", device),
    )

    # Step 6: 3-NN search with box pruning
    mean_dists = torch.zeros(P, dtype=torch.float32, device=device)

    wp.launch(
        _box_mean_dist_kernel,
        dim=P,
        inputs=[
            wp_sx, wp_sy, wp_sz, wp_sort_idx,
            wp.from_torch(box_min_x_t), wp.from_torch(box_min_y_t), wp.from_torch(box_min_z_t),
            wp.from_torch(box_max_x_t), wp.from_torch(box_max_y_t), wp.from_torch(box_max_z_t),
            wp.int32(num_boxes), wp.int32(P),
            wp.from_torch(mean_dists),
        ],
        device=dev_str,
        block_dim=get_tuned_block_dim("knn_box_dist", device),
    )

    return mean_dists
