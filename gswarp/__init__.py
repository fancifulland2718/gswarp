"""gswarp — Pure-Python Warp backend for 3D Gaussian Splatting."""

__version__ = "1.0.5"

from .rasterizer import (  # noqa: F401
    # Core rasterizer types
    GaussianRasterizationSettings,
    GaussianRasterizer,
    RasterizerMeta,
    # Core functions
    rasterize_gaussians,
    # Setup & memory management
    initialize_runtime_tuning,
    get_runtime_tuning_report,
    clear_warp_caches,
    # Runtime mode controls
    get_backward_mode,
    set_backward_mode,
    get_compute_depth,
    set_compute_depth,
    get_binning_sort_mode,
    set_binning_sort_mode,
)

from ._tuning import (  # noqa: F401
    # Auto-tuning API: inspect and customize block_dim recommendations
    get_tuned_block_dim,
    register_kernel_class,
    get_canonical_device,
)

from .fused_ssim import fused_ssim  # noqa: F401

from .knn import distCUDA2  # noqa: F401
