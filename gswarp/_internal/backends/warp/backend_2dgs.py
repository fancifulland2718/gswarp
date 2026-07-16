"""Perspective-correct 2D Gaussian surfel provider.

The method keeps sorting/runtime ownership compatible with gswarp while its
ray-splat geometry is expressed with regular PyTorch operations. This preserves
the complete autograd graph for surfel parameters without adding a second,
partially overlapping manual backward implementation to the baseline backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from gswarp._internal.backends.warp import backend_3dgs as _base
from gswarp._internal.backends.warp.state import ForwardResult, RenderStageResult


BACKEND_CAPABILITIES = _base.BACKEND_CAPABILITIES | frozenset({"twodgs", "torch_autograd_geometry"})
ForwardState = _base.ForwardState


@dataclass(slots=True)
class TwoDPreprocessOutputs:
    visible: torch.Tensor
    depths: torch.Tensor
    radii: torch.Tensor
    proj_2d: torch.Tensor
    conic_2d: torch.Tensor
    conic_2d_inv: torch.Tensor
    points_xy_image: torch.Tensor
    tiles_touched: torch.Tensor
    rgb: torch.Tensor
    conic_opacity: torch.Tensor
    tu: torch.Tensor
    tv: torch.Tensor
    tw: torch.Tensor
    normals: torch.Tensor


@dataclass(slots=True)
class TwoDBinningState:
    order: torch.Tensor
    num_rendered: int


@dataclass(slots=True)
class TwoDForwardState:
    preprocess: TwoDPreprocessOutputs
    binning: TwoDBinningState
    n_contrib: torch.Tensor


def _quaternion_rotation(quaternions: torch.Tensor) -> torch.Tensor:
    q = quaternions / quaternions.square().sum(dim=1, keepdim=True).clamp_min(1.0e-12).sqrt()
    w, x, y, z = q.unbind(dim=1)
    return torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x.square() + z.square()),
            2.0 * (y * z - w * x),
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x.square() + y.square()),
        ),
        dim=1,
    ).reshape(-1, 3, 3)


def _evaluate_sh(inputs) -> torch.Tensor:
    if inputs.colors.numel() != 0:
        return inputs.colors.reshape(inputs.means3d.shape[0], 3)
    coeffs = inputs.sh.reshape(inputs.means3d.shape[0], -1, 3)
    direction = inputs.means3d - inputs.campos.reshape(1, 3)
    direction = direction / direction.square().sum(dim=1, keepdim=True).clamp_min(1.0e-12).sqrt()
    x, y, z = direction.unbind(dim=1)
    result = 0.28209479177387814 * coeffs[:, 0]
    if inputs.degree > 0:
        result = result + (-0.4886025119029199 * y)[:, None] * coeffs[:, 1]
        result = result + (0.4886025119029199 * z)[:, None] * coeffs[:, 2]
        result = result + (-0.4886025119029199 * x)[:, None] * coeffs[:, 3]
    if inputs.degree > 1:
        result = result + (1.0925484305920792 * x * y)[:, None] * coeffs[:, 4]
        result = result + (1.0925484305920792 * y * z)[:, None] * coeffs[:, 5]
        result = result + (0.31539156525252005 * (2.0 * z.square() - x.square() - y.square()))[:, None] * coeffs[:, 6]
        result = result + (1.0925484305920792 * x * z)[:, None] * coeffs[:, 7]
        result = result + (0.5462742152960396 * (x.square() - y.square()))[:, None] * coeffs[:, 8]
    if inputs.degree > 2:
        result = result + (0.5900435899266435 * y * (3.0 * x.square() - y.square()))[:, None] * coeffs[:, 9]
        result = result + (2.890611442640554 * x * y * z)[:, None] * coeffs[:, 10]
        result = result + (0.4570457994644658 * y * (4.0 * z.square() - x.square() - y.square()))[:, None] * coeffs[:, 11]
        result = result + (0.3731763325901154 * z * (2.0 * z.square() - 3.0 * x.square() - 3.0 * y.square()))[:, None] * coeffs[:, 12]
        result = result + (0.4570457994644658 * x * (4.0 * z.square() - x.square() - y.square()))[:, None] * coeffs[:, 13]
        result = result + (1.445305721320277 * z * (x.square() - y.square()))[:, None] * coeffs[:, 14]
        result = result + (0.5900435899266435 * x * (x.square() - 3.0 * y.square()))[:, None] * coeffs[:, 15]
    return (result + 0.5).clamp_min(0.0)


def _world_to_pixel(inputs) -> torch.Tensor:
    pixel = torch.zeros((4, 3), dtype=inputs.means3d.dtype, device=inputs.means3d.device)
    pixel[0, 0] = 0.5 * inputs.image_width
    pixel[1, 1] = 0.5 * inputs.image_height
    pixel[3, 0] = 0.5 * (inputs.image_width - 1)
    pixel[3, 1] = 0.5 * (inputs.image_height - 1)
    pixel[3, 2] = 1.0
    return inputs.projmatrix @ pixel


def preprocess_stage(inputs) -> TwoDPreprocessOutputs:
    count = inputs.means3d.shape[0]
    device = inputs.means3d.device
    dtype = inputs.means3d.dtype
    rotation = _quaternion_rotation(inputs.rotations)
    scale = inputs.scales * float(inputs.scale_modifier)
    basis_u = rotation[:, :, 0] * scale[:, 0:1]
    basis_v = rotation[:, :, 1] * scale[:, 1:2]
    normal_world = rotation[:, :, 2]

    center_h = torch.cat((inputs.means3d, torch.ones((count, 1), dtype=dtype, device=device)), dim=1)
    vector_pad = torch.zeros((count, 1), dtype=dtype, device=device)
    world_to_pixel = _world_to_pixel(inputs)
    tu = torch.cat((basis_u, vector_pad), dim=1) @ world_to_pixel
    tv = torch.cat((basis_v, vector_pad), dim=1) @ world_to_pixel
    tw = center_h @ world_to_pixel
    tw_z = tw[:, 2:3]
    safe_tw_z = torch.where(tw_z.abs() > 1.0e-8, tw_z, torch.ones_like(tw_z))
    projected = tw[:, :2] / safe_tw_z + inputs.means2d[:, :2]

    view_center = center_h @ inputs.viewmatrix
    normal_view = normal_world @ inputs.viewmatrix[:3, :3]
    normal_view = normal_view / normal_view.square().sum(dim=1, keepdim=True).clamp_min(1.0e-12).sqrt()
    facing = -(view_center[:, :3] * normal_view).sum(dim=1, keepdim=True)
    normal_view = torch.where(facing >= 0.0, normal_view, -normal_view)

    jac_u = (tu[:, :2] * tw_z - tw[:, :2] * tu[:, 2:3]) / safe_tw_z.square()
    jac_v = (tv[:, :2] * tw_z - tw[:, :2] * tv[:, 2:3]) / safe_tw_z.square()
    cov_xx = jac_u[:, 0].square() + jac_v[:, 0].square()
    cov_xy = jac_u[:, 0] * jac_u[:, 1] + jac_v[:, 0] * jac_v[:, 1]
    cov_yy = jac_u[:, 1].square() + jac_v[:, 1].square()
    determinant = cov_xx * cov_yy - cov_xy.square()
    visible = (view_center[:, 2] > float(inputs.near_plane)) & (determinant > 1.0e-12)
    inv_det = torch.where(visible, determinant.reciprocal(), torch.zeros_like(determinant))
    conic = torch.stack((cov_yy * inv_det, -cov_xy * inv_det, cov_xx * inv_det), dim=1)
    cov = torch.stack((cov_xx, cov_xy, cov_yy), dim=1)
    root = torch.sqrt(torch.clamp(0.25 * (cov_xx - cov_yy).square() + cov_xy.square(), min=0.0))
    radius = torch.ceil(3.0 * torch.sqrt(torch.clamp(0.5 * (cov_xx + cov_yy) + root, min=0.1))).to(torch.int32)
    radii = torch.where(visible, radius, torch.zeros_like(radius))
    conic = torch.where(visible[:, None], conic, torch.zeros_like(conic))
    cov = torch.where(visible[:, None], cov, torch.zeros_like(cov))
    opacity = inputs.opacities.reshape(count, 1)
    return TwoDPreprocessOutputs(
        visible=visible,
        depths=view_center[:, 2],
        radii=radii,
        proj_2d=projected,
        conic_2d=conic,
        conic_2d_inv=cov,
        points_xy_image=projected,
        tiles_touched=torch.zeros((count,), dtype=torch.int32, device=device),
        rgb=_evaluate_sh(inputs),
        conic_opacity=torch.cat((conic, opacity), dim=1),
        tu=tu,
        tv=tv,
        tw=tw,
        normals=normal_view,
    )


def feature_stage(_inputs, preprocess_outputs):
    return preprocess_outputs.rgb


def _build_binning_state(preprocess_outputs, _height, _width) -> TwoDBinningState:
    order = torch.argsort(preprocess_outputs.depths.detach(), stable=True)
    return TwoDBinningState(order=order, num_rendered=int(order.numel()))


def render_stage(inputs, preprocess_outputs, binning_state, features) -> RenderStageResult:
    height, width = inputs.image_height, inputs.image_width
    device, dtype = features.device, features.dtype
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    transmittance = torch.ones((height, width), device=device, dtype=dtype)
    color = torch.zeros((3, height, width), device=device, dtype=dtype)
    depth = torch.zeros((height, width), device=device, dtype=dtype)
    normal = torch.zeros((3, height, width), device=device, dtype=dtype)
    distortion = torch.zeros((height, width), device=device, dtype=dtype)
    median_depth = torch.zeros((height, width), device=device, dtype=dtype)
    moment1 = torch.zeros((height, width), device=device, dtype=dtype)
    moment2 = torch.zeros((height, width), device=device, dtype=dtype)
    contributors = torch.zeros((height, width), device=device, dtype=torch.int32)
    filter_inv_square = 1.0 / max(float(inputs.filter_radius) ** 2, 1.0e-12)
    near, far = float(inputs.near_plane), float(inputs.far_plane)

    for point_id in binning_state.order:
        tu = preprocess_outputs.tu[point_id]
        tv = preprocess_outputs.tv[point_id]
        tw = preprocess_outputs.tw[point_id]
        k = xs[..., None] * tw - tu
        line = ys[..., None] * tw - tv
        intersection = torch.linalg.cross(k, line, dim=-1)
        denominator = intersection[..., 2]
        safe_denominator = torch.where(denominator.abs() > 1.0e-8, denominator, torch.ones_like(denominator))
        local_uv = intersection[..., :2] / safe_denominator[..., None]
        rho3d = local_uv.square().sum(dim=-1)
        center = preprocess_outputs.points_xy_image[point_id]
        rho2d = ((xs - center[0]).square() + (ys - center[1]).square()) * filter_inv_square
        rho = torch.minimum(rho3d, rho2d)
        ray_depth = local_uv[..., 0] * tw[0] + local_uv[..., 1] * tw[1] + tw[2]
        valid = (
            preprocess_outputs.visible[point_id].to(dtype)
            * (denominator.abs() > 1.0e-8).to(dtype)
            * (ray_depth > near).to(dtype)
            * (rho <= 9.0).to(dtype)
        )
        alpha = torch.clamp(preprocess_outputs.conic_opacity[point_id, 3] * torch.exp(-0.5 * rho), max=0.99)
        alpha = torch.where(alpha >= (1.0 / 255.0), alpha, torch.zeros_like(alpha)) * valid
        next_transmittance = transmittance * (1.0 - alpha)
        accepted = next_transmittance >= 1.0e-4
        alpha = torch.where(accepted, alpha, torch.zeros_like(alpha))
        next_transmittance = torch.where(accepted, next_transmittance, transmittance)
        weight = alpha * transmittance
        color = color + features[point_id].reshape(3, 1, 1) * weight
        depth = depth + ray_depth * weight
        normal = normal + preprocess_outputs.normals[point_id].reshape(3, 1, 1) * weight
        normalized_depth = far / (far - near) * (1.0 - near / ray_depth.clamp_min(near))
        accumulated_alpha = 1.0 - transmittance
        distortion = distortion + (
            normalized_depth.square() * accumulated_alpha + moment2 - 2.0 * normalized_depth * moment1
        ) * weight
        moment1 = moment1 + normalized_depth * weight
        moment2 = moment2 + normalized_depth.square() * weight
        median_depth = torch.where(transmittance > 0.5, ray_depth, median_depth)
        contributors = contributors + accepted.to(torch.int32)
        transmittance = next_transmittance

    color = color + transmittance * inputs.background.reshape(3, 1, 1).to(dtype=dtype)
    alpha = 1.0 - transmittance
    return RenderStageResult(
        color=color,
        depth=depth.unsqueeze(0),
        alpha=alpha.unsqueeze(0),
        n_contrib=contributors,
        aux=(normal, distortion.unsqueeze(0), median_depth.unsqueeze(0)),
    )


def build_state_stage(preprocess_outputs, binning_state, render_result):
    return TwoDForwardState(preprocess_outputs, binning_state, render_result.n_contrib)


def empty_forward_stage(inputs) -> ForwardResult:
    height, width = inputs.image_height, inputs.image_width
    device = inputs.means3d.device
    color = inputs.background.reshape(3, 1, 1).expand(3, height, width).clone()
    zero_image = torch.zeros((1, height, width), dtype=torch.float32, device=device)
    count = inputs.means3d.shape[0]
    return ForwardResult(
        num_rendered=0,
        color=color,
        depth=zero_image,
        alpha=zero_image,
        radii=torch.zeros((count,), dtype=torch.int32, device=device),
        proj_2d=torch.zeros((count, 2), dtype=torch.float32, device=device),
        conic_2d=torch.zeros((count, 3), dtype=torch.float32, device=device),
        conic_2d_inv=torch.zeros((count, 3), dtype=torch.float32, device=device),
        state=None,
        aux=(torch.zeros((3, height, width), dtype=torch.float32, device=device), zero_image, zero_image),
    )


def rasterize_gaussians_backward_typed(*_args, **_kwargs):
    raise RuntimeError("2DGS uses PyTorch autograd and has no manual Warp backward stage")


def __getattr__(name):
    return getattr(_base, name)


__all__ = [
    "BACKEND_CAPABILITIES",
    "ForwardState",
    "_build_binning_state",
    "build_state_stage",
    "empty_forward_stage",
    "feature_stage",
    "mark_visible",
    "preprocess_stage",
    "rasterize_gaussians_backward_typed",
    "render_stage",
]
