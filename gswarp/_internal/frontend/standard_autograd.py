"""Standard rasterizer autograd implementation surface."""

from __future__ import annotations

import torch

from gswarp._internal.frontend import common


def rasterize_gaussians(
    plan,
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _WarpRasterizeGaussians.apply(
        plan,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _WarpRasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        plan,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):
        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
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
        )

        result, execution_options = common.run_typed_forward(plan, raster_settings, args)

        ctx.plan = plan
        ctx.raster_settings = raster_settings
        ctx.execution_options = execution_options
        ctx.num_rendered = result.num_rendered
        ctx.forward_state = result.state
        ctx.save_for_backward(
            colors_precomp,
            opacities,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            result.radii,
            sh,
            result.alpha,
        )
        return (
            result.color,
            result.radii,
            result.depth,
            result.alpha,
            result.proj_2d,
            result.conic_2d,
            result.conic_2d_inv,
        )

    @staticmethod
    def backward(
        ctx,
        grad_color,
        grad_radii,
        grad_depth,
        grad_alpha,
        grad_proj_2D,
        grad_conic_2D,
        grad_conic_2D_inv,
    ):
        plan = ctx.plan
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        (
            colors_precomp,
            opacities,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            alpha,
        ) = ctx.saved_tensors

        empty_buffer = torch.empty((0,), dtype=torch.uint8, device=means3D.device)

        args = (
            raster_settings.bg,
            means3D,
            radii,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            grad_color,
            grad_depth,
            grad_alpha,
            grad_proj_2D,
            grad_conic_2D,
            grad_conic_2D_inv,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            empty_buffer,
            num_rendered,
            empty_buffer,
            empty_buffer,
            alpha,
        )

        try:
            grads = common.run_typed_backward(
                plan, raster_settings, args, ctx.execution_options, ctx.forward_state
            )
        finally:
            # Arbitrary autograd context attributes are not released by PyTorch.
            ctx.forward_state = None

        (
            grad_means2D,
            grad_colors_precomp,
            grad_opacities,
            grad_means3D,
            grad_cov3Ds_precomp,
            grad_sh,
            grad_scales,
            grad_rotations,
        ) = grads

        return (
            None,
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )


__all__ = ["rasterize_gaussians"]
