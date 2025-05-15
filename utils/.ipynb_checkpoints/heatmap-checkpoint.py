import cv2
import numpy as np

SIGMA         = 15        # Gaussian std‑dev in pixels (≈ vehicle “radius”)
DECAY         = 1       # Exponential time‑decay per frame  (0.97 → keeps 97 %)
COLORMAP      = cv2.COLORMAP_JET
ACCUM_DTYPE   = np.float32  # Always accumulate in 32‑bit float

def make_kernel(sigma: int = SIGMA) -> np.ndarray:
    """
    Build a 2‑D Gaussian kernel with peak = 1.0 (float32).

    Returns
    -------
    kernel : np.ndarray, shape = (6*sigma+1, 6*sigma+1), dtype = float32
    """
    ksize  = int(6 * sigma + 1)
    g1d    = cv2.getGaussianKernel(ksize, sigma)
    kernel = (g1d @ g1d.T).astype(ACCUM_DTYPE)
    return kernel / kernel.max()           # normalise peak to 1.0


def _as_f32(arr: np.ndarray) -> np.ndarray:
    """Ensure the array is a proper contiguous float32 NumPy array."""
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr)
    return arr


def update_heatmap(accum: np.ndarray,
                   cx: int,
                   cy: int,
                   weight: float,
                   kernel: np.ndarray,
                   decay: float = DECAY) -> None:
    """
    Safely add a weighted Gaussian kernel to the float32 accumulator.
    Prevents dtype mismatches that cause TypeErrors.
    """
    accum  = _as_f32(accum)
    kernel = _as_f32(kernel)
    weight = float(weight)

    accum *= decay

    h, w = accum.shape
    kH, kW = kernel.shape
    kC_y, kC_x = kH // 2, kW // 2

    top    = max(0, cy - kC_y)
    left   = max(0, cx - kC_x)
    bottom = min(h, cy + kC_y + 1)
    right  = min(w, cx + kC_x + 1)

    if top >= bottom or left >= right:
        return

    k_top    = kC_y - (cy - top)
    k_left   = kC_x - (cx - left)
    k_bottom = k_top  + (bottom - top)
    k_right  = k_left + (right - left)

    roi_accum  = _as_f32(accum[top:bottom, left:right])
    roi_kernel = _as_f32(kernel[k_top:k_bottom, k_left:k_right])

    np.add(roi_accum, weight * roi_kernel, out=roi_accum, casting="unsafe")
    accum[top:bottom, left:right] = roi_accum

def render_heatmap(accum: np.ndarray,
                   base_frame: np.ndarray | None = None,
                   alpha: float = 0.6,
                   sigma_blur: int = SIGMA) -> np.ndarray:
    """
    Convert the float32 accumulator to a coloured heatmap (BGR).

    Parameters
    ----------
    accum       : float32 H×W array (the accumulator)
    base_frame  : optional BGR image to overlay the heatmap on
    alpha       : weight of `base_frame` in the overlay
    sigma_blur  : Gaussian blur radius for smoothing

    Returns
    -------
    heat_col : np.ndarray, BGR heatmap (same size as input frame)
    """
    # Smooth for visual appeal
    blurred = cv2.GaussianBlur(accum, (0, 0), sigma_blur)

    # Normalise to 0‑255, convert to uint8
    blurred_log = np.log1p(blurred)  # log1p for visual contrast
    normed = cv2.normalize(blurred_log, None, 0, 255, cv2.NORM_MINMAX)
    heat8 = normed.astype(np.uint8)


    # Apply colour map
    heat_col = cv2.applyColorMap(heat8, COLORMAP)

    # Optional overlay
    if base_frame is not None:
        heat_col = cv2.addWeighted(base_frame, alpha, heat_col, 1 - alpha, 0)

    return heat_col
