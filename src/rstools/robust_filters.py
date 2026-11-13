import numpy as np
import scipy.ndimage as cpu_ndimage
from scipy.signal import savgol_coeffs

from ._backend import HAS_GPU, cp, get_backend, to_cpu
from .stats import mad  # update to match where mad is defined in your package

if HAS_GPU:
    import cupyx.scipy.ndimage as cpx_ndimage
else:
    cpx_ndimage = None

_np_gaussian_filter1d = cpu_ndimage.gaussian_filter1d
_cp_gaussian_filter1d = cpx_ndimage.gaussian_filter1d if cpx_ndimage else None

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


def _iter_batch_slices(batch_shape, axis_len, max_chunk_elems):
    """Yield slice tuples over batch_shape keeping each chunk <= max_chunk_elems."""
    if not batch_shape:
        yield ()
        return

    for dim_idx in range(len(batch_shape)):
        trailing_elems = int(np.prod(batch_shape[dim_idx + 1:])) * axis_len
        if trailing_elems <= max_chunk_elems or dim_idx == len(batch_shape) - 1:
            step = max(1, max_chunk_elems // trailing_elems)
            prefix_shape = batch_shape[:dim_idx]
            for prefix_idx in np.ndindex(*prefix_shape) if prefix_shape else [()]:
                for start in range(0, batch_shape[dim_idx], step):
                    end = min(start + step, batch_shape[dim_idx])
                    yield prefix_idx + (slice(start, end),)
            return


def robust_gaussian_filter1d(
    x,
    sigma,
    axis=-1,
    nsigma=5.0,
    bad_mask=None,
    mode='reflect',
    cval=0.0,
    truncate=4.0,
    max_samples=10000,
    seed=0,
    chunk_mb=128,
):
    """
    Smooth an array along a 1D axis with a Gaussian kernel while rejecting outliers.

    Uses MAD-based sigma clipping (via ``mad``) and normalized convolution so
    bad pixels, cosmic-ray hits, and masked gaps do not bias the smoothed output.
    Processes large multi-dimensional arrays in bounded chunks along non-smoothed
    axes to keep peak GPU/CPU memory usage low.

    Parameters
    ----------
    x : array_like
        Input array (NumPy or CuPy ndarray). Integer arrays are promoted to
        ``float32`` prior to filtering.
    sigma : float
        Standard deviation of the 1D Gaussian kernel in elements.
    axis : int, optional
        Axis along which to smooth. Default is -1.
    nsigma : float or None, optional
        Outlier rejection threshold in units of MAD-estimated standard
        deviation. Set to None to skip sigma clipping and use only ``bad_mask``
        and finiteness checks. Default is 5.0.
    bad_mask : array_like of bool, optional
        Boolean mask broadcastable to ``x.shape`` where True marks known bad or
        masked samples to exclude. Default is None.
    mode : str, optional
        Boundary extension mode passed to ``gaussian_filter1d``.
        Default is 'reflect'.
    cval : float, optional
        Constant fill value when ``mode='constant'``. Default is 0.0.
    truncate : float, optional
        Truncate the filter at this many standard deviations. Default is 4.0.
    max_samples : int, optional
        Maximum sample limit forwarded to ``mad``. Default is 10000.
    seed : int or None, optional
        Random seed forwarded to ``mad``. Default is 0.
    chunk_mb : float, optional
        Target maximum chunk size in MiB along non-smoothed axes. Lower this if
        VRAM is heavily constrained. Default is 128.

    Returns
    -------
    out : ndarray
        Smoothed array on the same backend (NumPy or CuPy) and with the same
        shape as ``x``.
    """
    xp = get_backend(x)
    gf1d = _cp_gaussian_filter1d if (HAS_GPU and xp is cp) else _np_gaussian_filter1d

    x = xp.asarray(x)
    is_float = xp.issubdtype(x.dtype, xp.floating)
    work_dtype = x.dtype if is_float else xp.float32
    itemsize = np.dtype(work_dtype).itemsize
    max_chunk_elems = max(1, int((chunk_mb * 1024 * 1024) // itemsize))

    # Move target axis to -1 via a zero-copy view
    x_moved = xp.moveaxis(x, axis, -1)
    out = xp.empty(x_moved.shape, dtype=work_dtype)

    if bad_mask is not None:
        mask_arr = (
            xp.asarray(bad_mask, dtype=bool)
            if (HAS_GPU and xp is cp)
            else np.asarray(to_cpu(bad_mask), dtype=bool)
        )
        mask_moved = xp.moveaxis(xp.broadcast_to(mask_arr, x.shape), axis, -1)
    else:
        mask_moved = None

    zero = xp.zeros((), dtype=work_dtype)
    one = xp.ones((), dtype=work_dtype)

    batch_shape = x_moved.shape[:-1]
    axis_len = x_moved.shape[-1]

    for slc in _iter_batch_slices(batch_shape, axis_len, max_chunk_elems):
        x_c = x_moved[slc]
        if not is_float:
            x_c = x_c.astype(work_dtype)
        out_c = out[slc]

        valid = xp.isfinite(x_c) if is_float else xp.ones(x_c.shape, dtype=bool)
        if mask_moved is not None:
            valid &= ~mask_moved[slc]

        if nsigma is not None:
            x_clean = xp.where(valid, x_c, zero) if (is_float or mask_moved is not None) else x_c
            mad_val, med = mad(
                x_clean,
                axis=-1,
                keepdims=True,
                scale='normal',
                max_samples=max_samples,
                return_median=True,
                seed=seed,
            )
            valid &= xp.abs(x_c - med) <= (nsigma * mad_val)
            del x_clean
        else:
            med = zero

        # Smooth masked values directly into the output slice
        work = xp.where(valid, x_c, zero)
        gf1d(work, sigma=sigma, axis=-1, output=out_c, mode=mode, cval=cval, truncate=truncate)

        # Smooth validity weights and normalize in place
        work = valid.astype(work_dtype)
        del valid
        den = xp.empty_like(out_c)
        gf1d(work, sigma=sigma, axis=-1, output=den, mode=mode, cval=cval, truncate=truncate)
        del work

        good = den > 1e-6
        den = xp.where(good, den, one)
        out_c /= den
        del den
        out_c[:] = xp.where(good, out_c, med)
        del good

    return xp.moveaxis(out, -1, axis)


