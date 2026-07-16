"""Autograd bridge for the method-owned 2D Gaussian surfel provider."""

from __future__ import annotations

import torch

from gswarp._internal.frontend import common


def _run_forward(
    plan,
    raster_settings,
    means3d,
    screen_offsets,
    sh,
    colors,
    opacities,
    scales,
    rotations,
):
    args = (
        raster_settings.bg,
        means3d,
        screen_offsets,
        colors,
        opacities,
        scales,
        rotations,
        raster_settings.scale_modifier,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.filter_radius,
        raster_settings.near_plane,
        raster_settings.far_plane,
    )
    result, _options = common.run_typed_forward(plan, raster_settings, args)
    normal, distortion, median_depth = result.aux
    return (
        result.color,
        result.radii,
        result.depth,
        result.alpha,
        normal,
        distortion,
        median_depth,
        result.proj_2d,
        result.conic_2d,
        result.conic_2d_inv,
    )


def rasterize_surfels(
    plan,
    means3d,
    means2d,
    sh,
    colors,
    opacities,
    scales,
    rotations,
    raster_settings,
):
    """Render surfels while preserving the CUDA-style ``means2D`` gradient proxy."""

    return _TwoDSurfelRasterize.apply(
        plan,
        raster_settings,
        means3d,
        means2d,
        sh,
        colors,
        opacities,
        scales,
        rotations,
    )


class _TwoDSurfelRasterize(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        plan,
        raster_settings,
        means3d,
        means2d,
        sh,
        colors,
        opacities,
        scales,
        rotations,
    ):
        # In the standard CUDA API this tensor is a screen-space gradient
        # proxy, not a forward displacement. Keep it zero during rendering.
        screen_offsets = torch.zeros_like(means2d)
        outputs = _run_forward(
            plan,
            raster_settings,
            means3d,
            screen_offsets,
            sh,
            colors,
            opacities,
            scales,
            rotations,
        )
        ctx.plan = plan
        ctx.raster_settings = raster_settings
        ctx.save_for_backward(means3d, means2d, sh, colors, opacities, scales, rotations)
        return outputs

    @staticmethod
    def backward(
        ctx,
        grad_color,
        _grad_radii,
        grad_depth,
        grad_alpha,
        grad_normal,
        grad_distortion,
        grad_median_depth,
        grad_proj_2d,
        grad_conic_2d,
        grad_conic_2d_inv,
    ):
        means3d, means2d, sh, colors, opacities, scales, rotations = ctx.saved_tensors
        if means3d.shape[0] == 0:
            return (
                None,
                None,
                torch.zeros_like(means3d),
                torch.zeros_like(means2d),
                torch.zeros_like(sh),
                torch.zeros_like(colors),
                torch.zeros_like(opacities),
                torch.zeros_like(scales),
                torch.zeros_like(rotations),
            )
        with torch.enable_grad():
            replay_means3d = means3d.detach().requires_grad_(True)
            replay_offsets = torch.zeros_like(means2d, requires_grad=True)
            replay_sh = sh.detach().requires_grad_(True)
            replay_colors = colors.detach().requires_grad_(True)
            replay_opacities = opacities.detach().requires_grad_(True)
            replay_scales = scales.detach().requires_grad_(True)
            replay_rotations = rotations.detach().requires_grad_(True)
            outputs = _run_forward(
                ctx.plan,
                ctx.raster_settings,
                replay_means3d,
                replay_offsets,
                replay_sh,
                replay_colors,
                replay_opacities,
                replay_scales,
                replay_rotations,
            )
            differentiable_outputs = (
                outputs[0],
                outputs[2],
                outputs[3],
                outputs[4],
                outputs[5],
                outputs[6],
                outputs[7],
                outputs[8],
                outputs[9],
            )
            output_grads = (
                torch.zeros_like(outputs[0]) if grad_color is None else grad_color,
                torch.zeros_like(outputs[2]) if grad_depth is None else grad_depth,
                torch.zeros_like(outputs[3]) if grad_alpha is None else grad_alpha,
                torch.zeros_like(outputs[4]) if grad_normal is None else grad_normal,
                torch.zeros_like(outputs[5]) if grad_distortion is None else grad_distortion,
                torch.zeros_like(outputs[6]) if grad_median_depth is None else grad_median_depth,
                torch.zeros_like(outputs[7]) if grad_proj_2d is None else grad_proj_2d,
                torch.zeros_like(outputs[8]) if grad_conic_2d is None else grad_conic_2d,
                torch.zeros_like(outputs[9]) if grad_conic_2d_inv is None else grad_conic_2d_inv,
            )
            replay_inputs = (
                replay_means3d,
                replay_offsets,
                replay_sh,
                replay_colors,
                replay_opacities,
                replay_scales,
                replay_rotations,
            )
            gradients = torch.autograd.grad(
                differentiable_outputs,
                replay_inputs,
                grad_outputs=output_grads,
                allow_unused=True,
            )

        normalized = tuple(
            torch.zeros_like(value) if gradient is None else gradient
            for value, gradient in zip(replay_inputs, gradients, strict=True)
        )
        return (None, None, *normalized)


__all__ = ["rasterize_surfels"]
