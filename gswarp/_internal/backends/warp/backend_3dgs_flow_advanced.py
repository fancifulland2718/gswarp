"""Advanced Warp backend placeholder for flow auxiliary 3DGS."""

BACKEND_AVAILABLE = False
BACKEND = None
MIN_WARP_VERSION = (1, 12, 0)
REQUIRED_WARP_CAPABILITIES = frozenset()
BACKEND_CAPABILITIES = frozenset()

__all__ = [
    "BACKEND_AVAILABLE",
    "BACKEND",
    "MIN_WARP_VERSION",
    "REQUIRED_WARP_CAPABILITIES",
    "BACKEND_CAPABILITIES",
]
