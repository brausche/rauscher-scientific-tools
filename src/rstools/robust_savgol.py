import numpy as np
from scipy.signal import savgol_coeffs

from ._backend import HAS_GPU, get_backend

if HAS_GPU:
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cpx_ndimage
    except ImportError:
        cpx_ndimage = None
else:
    cpx_ndimage = None

import scipy.ndimage as cpu_ndimage


def robust_savgol(
    D,
    window_length: int = 17,
    polyorder: int = 2,
    n_sigma: float = 3.5,
    in_place: bool = False,
):
    """
    Two-stage Hampel + Savitzky-Golay filter for outlier-contaminated time series.

    Stage 1 applies a moving Hampel filter (rolling median + rolling MAD) along
    the time-ordered pixel axis to detect and replace outliers with the local
    median. Stage 2 applies a 1D Savitzky-Golay convolution to the cleaned signal.

    Parameters
    ----------
    D : ndarray
        Ramp datacube with shape (nz, nout, npix) or 2D frame (nout, npix),
        residing in CPU host RAM (NumPy) or GPU VRAM (CuPy).
    window_length : int, optional
        Odd integer defining the filter window length along the pixel axis.
        Default is 17 (closest symmetric odd integer to 16).
    polyorder : int, optional
        Degree of the polynomial fit for Savitzky-Golay. Default is 2.
    n_sigma : float, optional
        Sigma threshold for Hampel outlier rejection. Default is 3.5.
    in_place : bool, optional
        If True, writes smoothed results directly back into D. If False,
        allocates a new array for the output. Default is False.

    Returns
    -------
    D_smooth : ndarray
        Smoothed array matching the shape and backend of D.
    """
    # Enforce odd window length
    if window_length % 2 == 0:
        window_length += 1

    xp = get_backend(D)
    dtype = D.dtype

    # Dispatch to GPU or CPU ndimage
    if HAS_GPU and isinstance(D, cp.ndarray):
        ndimage = cpx_ndimage
    else:
        ndimage = cpu_ndimage

    # Precompute Savitzky-Golay FIR convolution kernel in float32
    raw_coeffs = savgol_coeffs(window_length, polyorder)
    sg_weights = xp.asarray(raw_coeffs, dtype=dtype)

    # Pre-cast scalar constants to input dtype to prevent float64 upcasting
    scale_mad = xp.array(1.482602218505602, dtype=dtype)
    sigma_thresh = xp.array(n_sigma, dtype=dtype)
    eps = xp.array(1e-6, dtype=dtype)

    # Prepare output buffer
    out = D if in_place else xp.empty_like(D)

    # Handle 2D (nout, npix) or 3D (nz, nout, npix)
    is_3d = D.ndim == 3
    nz = D.shape[0] if is_3d else 1

    for z in range(nz):
        frame = D[z] if is_3d else D

        # --- Stage 1: Hampel Filter along fast-scan axis (axis 1) ---
        # 1. Rolling median along time stream
        med = ndimage.median_filter(frame, size=(1, window_length), mode="nearest")

        # 2. Rolling MAD
        dev = xp.abs(frame - med)
        mad = ndimage.median_filter(dev, size=(1, window_length), mode="nearest") * scale_mad

        # 3. Detect outliers and substitute with local median
        threshold = sigma_thresh * xp.maximum(mad, eps)
        is_outlier = dev > threshold
        cleaned_frame = xp.where(is_outlier, med, frame)

        # --- Stage 2: Savitzky-Golay Smoothing ---
        # Fast 1D convolution with precomputed polynomial weights
        smoothed_frame = ndimage.convolve1d(
            cleaned_frame, weights=sg_weights, axis=-1, mode="nearest"
        )

        if is_3d:
            out[z] = smoothed_frame
        else:
            out[:] = smoothed_frame

    return out
