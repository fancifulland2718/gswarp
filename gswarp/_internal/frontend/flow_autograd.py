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
        outputs = run_with_runtime_overrides(
            backend,
            raster_settings,
            lambda: backend.rasterize_gaussians(*args),
            flow=True,
            options=execution_options,
        )

        (
            num_rendered,
            color,
            depth,
            alpha,
            radii,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            proj_2D,
            conic_2D,
            conic_2D_inv,
            gs_per_pixel,
            weight_per_gs_pixel,
            x_mu,
        ) = outputs

        ctx.backend = backend
        ctx.raster_settings = raster_settings
        ctx.execution_options = execution_options
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(
            colors_precomp,
            opacities,
            means3D,
            scales,
            rotations,
            cov3Ds_precomp,
            radii,
            sh,
            geomBuffer,
            binningBuffer,
            imgBuffer,
            alpha,
        )
        return color, radii, depth, alpha, proj_2D, conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu

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
            geomBuffer,
            binningBuffer,
            imgBuffer,
            alpha,
        ) = ctx.saved_tensors

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
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            alpha,
            raster_settings.enable_flow_grad,
        )

        grads = run_with_runtime_overrides(
            backend,
            raster_settings,
            lambda: backend.rasterize_gaussians_backward(*args),
            flow=True,
            options=ctx.execution_options,
        )

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
