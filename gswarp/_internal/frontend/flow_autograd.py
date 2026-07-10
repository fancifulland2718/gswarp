"""Flow rasterizer autograd implementation surface."""

from __future__ import annotations

import torch

from gswarp._internal.api.runtime_context import resolve_execution_options, run_with_runtime_overrides


def rasterize_gaussians(
    backend,
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
        backend,
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
        backend,
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

        execution_options = resolve_execution_options(backend, raster_settings, flow=True)
        result = run_with_runtime_overrides(
            backend,
            raster_settings,
            lambda: backend.rasterize_gaussians_typed(*args),
            flow=True,
            options=execution_options,
        )

        ctx.backend = backend
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
            *result.aux,
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
        dummy_gs_per_pixel,
        dummy_weight_per_gs_pixel,
        grad_x_mu,
    ):
        backend = ctx.backend
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
            dummy_gs_per_pixel,
            dummy_weight_per_gs_pixel,
            grad_x_mu,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            empty_buffer,
            num_rendered,
            empty_buffer,
            empty_buffer,
            alpha,
            raster_settings.enable_flow_grad,
        )

        try:
            grads = run_with_runtime_overrides(
                backend,
                raster_settings,
                lambda: backend.rasterize_gaussians_flow_backward_typed(
                    *args, forward_state=ctx.forward_state
                ),
                flow=True,
                options=ctx.execution_options,
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
