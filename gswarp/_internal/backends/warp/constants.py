from __future__ import annotations

import warp as wp

NUM_CHANNELS = 3
BLOCK_X = 16
BLOCK_Y = 16
sh_c0 = wp.constant(wp.float32(0.28209479177387814))
sh_c1 = wp.constant(wp.float32(0.4886025119029199))
sh_c2_0 = wp.constant(wp.float32(1.0925484305920792))
sh_c2_1 = wp.constant(wp.float32(-1.0925484305920792))
sh_c2_2 = wp.constant(wp.float32(0.31539156525252005))
sh_c2_3 = wp.constant(wp.float32(-1.0925484305920792))
sh_c2_4 = wp.constant(wp.float32(0.5462742152960396))

sh_c3_0 = wp.constant(wp.float32(-0.5900435899266435))
sh_c3_1 = wp.constant(wp.float32(2.890611442640554))
sh_c3_2 = wp.constant(wp.float32(-0.4570457994644658))
sh_c3_3 = wp.constant(wp.float32(0.3731763325901154))
sh_c3_4 = wp.constant(wp.float32(-0.4570457994644658))
sh_c3_5 = wp.constant(wp.float32(1.445305721320277))
sh_c3_6 = wp.constant(wp.float32(-0.5900435899266435))

BINNING_SORT_MODES = ("warp_radix", "torch", "warp_depth_stable_tile")
WARP_RADIX_DETERMINISTIC_TIEBREAK = False
TORCH_SINGLE_SORT_THRESHOLD = 1000000
FORWARD_GEOM_FLOAT_WIDTH = 16
FORWARD_GEOM_CLAMP_WIDTH = NUM_CHANNELS * 4
FORWARD_GEOM_STRIDE_BYTES = FORWARD_GEOM_FLOAT_WIDTH * 4 + FORWARD_GEOM_CLAMP_WIDTH
BINNING_AUTO_TUNE_GROWTH_CAP = 4.0
BINNING_AUTO_TUNE_SWITCH_RATIO = 2.5
BINNING_AUTO_TUNE_KEEP_RATIO = 2.0
BINNING_AUTO_TUNE_MIN_SWITCH_POINTS = 16384
BINNING_AUTO_TUNE_MIN_KEEP_POINTS = 12288
VISIBILITY_NEAR_PLANE = 0.2
PREPROCESS_CULL_SIGMA = 3.0
PREPROCESS_CULL_FOV_SCALE = 1.3
DET_EPSILON = 1.0e-7
ONE_MINUS_ALPHA_MIN = 1.0e-5
SNUGBOX_PIXEL_PADDING = 1.0

__all__ = [name for name in globals() if name.isupper() or name.startswith("sh_c")]
