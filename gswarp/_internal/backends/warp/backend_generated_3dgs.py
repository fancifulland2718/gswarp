"""Standard-3DGS stage bindings for materialized dynamic or anchor Gaussians."""

from . import backend_3dgs as _base
from .backend_3dgs import *  # noqa: F403
from .backend_3dgs import _build_binning_state


BACKEND_CAPABILITIES = frozenset(
    {
        "stable_warp",
        "typed_forward",
        "typed_backward",
        "mark_visible",
        "generated_3dgs",
    }
)


def __getattr__(name):
    return getattr(_base, name)
