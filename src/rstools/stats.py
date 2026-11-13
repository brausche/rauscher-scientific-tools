from ._backend import get_backend


def mad(x, axis=None, keepdims=False, scale='normal', max_samples=10000, return_median=False, seed=0):
    """
    Compute the Median Absolute Deviation (MAD).

    Parameters
    ----------
    x : array_like
        Input array (NumPy or CuPy ndarray).
    axis : int, tuple of ints, or None, optional
        Axis or axes along which to compute statistics. Default is None.
    keepdims : bool, optional
        If True, reduced axes are retained with size 1. Default is False.
    scale : {'normal', None} or float, optional
        Scale factor for the output. 'normal' applies the standard normal 
        consistency factor (~1.4826). Default is 'normal'.
    max_samples : int, optional
        Maximum sample limit before subsampling is applied. Default is 10000.
    return_median : bool, optional
        If True, returns a tuple of (mad_val, med). Default is False.
    seed : int or None, optional
        Random seed for subsampling generator. Default is 0.

    Returns
    -------
    mad_val : ndarray or scalar
        Median Absolute Deviation.
    med : ndarray or scalar, optional
        Median value, returned if return_median is True.

    Notes
    -----
    Algorithm designed by Bernard J. Rauscher, Principal Scientist, Rauscher
    Scientific LLC. Implementation and documentation assisted by Gemini AI.
    """
    xp = get_backend(x)
    x = xp.asarray(x)
    orig_ndim = x.ndim

    # Determine reduction size
    if axis is None:
        total_size = x.size
        reduction_axes = None
    else:
        reduction_axes = (axis,) if isinstance(axis, int) else tuple(axis)
        total_size = 1
        for a in reduction_axes:
            total_size *= x.shape[a]

    # Apply random subsampling if exceeding sample threshold
    if total_size > max_samples:
        rng = xp.random.default_rng(seed)
        if axis is None:
            idx = rng.integers(0, total_size, size=max_samples)
            x_sub = x.ravel()[idx]
            calc_axis = None
        else:
            x_sub = x
            for a in reduction_axes:
                if x_sub.shape[a] > max_samples:
                    idx = rng.integers(0, x_sub.shape[a], size=max_samples)
                    x_sub = xp.take(x_sub, idx, axis=a)
            calc_axis = reduction_axes
    else:
        x_sub = x
        calc_axis = reduction_axes

    # Compute median and absolute deviation
    med = xp.median(x_sub, axis=calc_axis, keepdims=True)
    dev = xp.abs(x_sub - med)
    mad_val = xp.median(dev, axis=calc_axis, keepdims=True)

    # Scale factor
    if scale == 'normal':
        mad_val = mad_val * 1.482602218505602
    elif scale is not None:
        mad_val = mad_val * scale

    # Align keepdims dimensions to match original array rank
    if axis is None and orig_ndim > 1:
        if keepdims:
            mad_val = mad_val.reshape((1,) * orig_ndim)
            med = med.reshape((1,) * orig_ndim)
        else:
            mad_val = xp.squeeze(mad_val)
            med = xp.squeeze(med)
    elif not keepdims:
        mad_val = xp.squeeze(mad_val, axis=reduction_axes)
        med = xp.squeeze(med, axis=reduction_axes)

    if return_median:
        return mad_val, med

    return mad_val


def rmean(x, n_sigma=3.0, axis=None, keepdims=False, max_samples=10000, return_std=False, seed=0):
    """
    Compute a robust, sigma-clipped mean using MAD for outlier rejection.

    Parameters
    ----------
    x : array_like
        Input array (NumPy or CuPy ndarray).
    n_sigma : float, optional
        Sigma threshold for outlier rejection. Default is 3.0.
    axis : int, tuple of ints, or None, optional
        Axis or axes along which to compute statistics. Default is None.
    keepdims : bool, optional
        If True, reduced axes are retained with size 1. Default is False.
    max_samples : int, optional
        Maximum sample limit passed to mad. Default is 10000.
    return_std : bool, optional
        If True, returns a tuple of (mean_val, std_val). Default is False.
    seed : int or None, optional
        Random seed for subsampling generator. Default is 0.

    Returns
    -------
    mean_val : ndarray or scalar
        Robust sigma-clipped mean.
    std_val : ndarray or scalar, optional
        Robust standard deviation of unclipped elements, returned if return_std is True.

    Notes
    -----
    Algorithm designed by Bernard J. Rauscher, Principal Scientist, Rauscher
    Scientific LLC. Implementation and documentation assisted by Gemini AI.
    """
    xp = get_backend(x)
    x = xp.asarray(x)
    dtype = x.dtype

    # 1. Estimate robust center (median) and scale (MAD)
    scale_mad, med = mad(
        x, axis=axis, keepdims=True,
        return_median=True, max_samples=max_samples, seed=seed
    )

    # 2. Identify in-distribution pixels
    valid_mask = xp.abs(x - med) <= (n_sigma * scale_mad)
    count_valid = xp.sum(valid_mask, axis=axis, keepdims=True)
    
    # Cast count_clamp to x.dtype so division does not promote float32 to float64
    count_clamp = xp.maximum(count_valid, 1).astype(dtype)

    # 3. Compute sigma-clipped mean strictly in x.dtype
    zero = xp.array(0, dtype=dtype)
    sum_valid = xp.sum(xp.where(valid_mask, x, zero), axis=axis, keepdims=True)
    mean_val = sum_valid / count_clamp

    # 4. Handle output shape formatting
    reduction_axes = (axis,) if isinstance(axis, int) else (axis if axis is None else tuple(axis))

    def _format_shape(arr):
        if keepdims:
            return arr
        elif axis is None:
            return xp.squeeze(arr)
        else:
            return xp.squeeze(arr, axis=reduction_axes)

    mean_out = _format_shape(mean_val)

    # 5. Optionally compute standard deviation over unclipped sample
    if return_std:
        sq_diff = xp.where(valid_mask, (x - mean_val) ** 2, zero)
        var_val = xp.sum(sq_diff, axis=axis, keepdims=True) / count_clamp
        std_val = xp.sqrt(var_val)
        std_out = _format_shape(std_val)
        return mean_out, std_out

    return mean_out
