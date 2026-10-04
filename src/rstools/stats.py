import jax
import jax.numpy as jnp

@jax.jit(static_argnames=('axis', 'keepdims', 'scale', 'max_samples', 'return_median'))
def mad(x, key=None, axis=None, keepdims=False, scale='normal', max_samples=10000, return_median=False):
    """
    Compute the Median Absolute Deviation (MAD).

    Parameters
    ----------
    x : array_like
        Input array.
    key : PRNGKey, optional
        Random number generator key for subsampling. Default is PRNGKey(0).
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

    Returns
    -------
    mad_val : Array
        Median Absolute Deviation.
    med : Array, optional
        Median value, returned if return_median is True.

    Notes
    -----
    Algorithm designed by Bernard J. Rauscher, Principal Scientist, Rauscher
    Scientific LLC. Implementation and documentation assisted by Gemini AI.
    """
    x = jnp.asarray(x)
    
    if key is None:
        key = jax.random.PRNGKey(0)

    # Determine element count along reduction target
    if axis is None:
        total_size = x.size
    else:
        axes = (axis,) if isinstance(axis, int) else axis
        total_size = 1
        for a in axes:
            total_size *= x.shape[a]

    # Apply random subsampling if exceeding sample threshold
    if total_size > max_samples:
        if axis is None:
            idx = jax.random.randint(key, shape=(max_samples,), minval=0, maxval=total_size)
            x = x.ravel()[idx]
        else:
            for a in (axis if isinstance(axis, tuple) else (axis,)):
                if x.shape[a] > max_samples:
                    key, subkey = jax.random.split(key)
                    idx = jax.random.randint(subkey, shape=(max_samples,), minval=0, maxval=x.shape[a])
                    x = jnp.take(x, idx, axis=a)

    # Compute median (keepdims=True for broadcasting against x)
    med = jnp.median(x, axis=axis, keepdims=True)
    dev = jnp.abs(x - med)
    mad_val = jnp.median(dev, axis=axis, keepdims=keepdims)

    if scale == 'normal':
        mad_val = mad_val * 1.482602218505602
    elif scale is not None:
        mad_val = mad_val * scale

    if return_median:
        # Format median dimensions to match specified keepdims behavior
        if keepdims:
            med_out = med
        elif axis is None:
            med_out = jnp.squeeze(med)
        else:
            med_out = jnp.squeeze(med, axis=axis)
        return mad_val, med_out

    return mad_val



    @jax.jit(static_argnames=('axis', 'keepdims', 'max_samples', 'return_std'))
def rmean(x, n_sigma=3.0, key=None, axis=None, keepdims=False, max_samples=10000, return_std=False):
    """
    Compute a robust, sigma-clipped mean using MAD for outlier rejection.

    Parameters
    ----------
    x : array_like
        Input array.
    n_sigma : float, optional
        Sigma threshold for outlier rejection. Default is 3.0.
    key : PRNGKey, optional
        Random number generator key for subsampling inside mad. Default is PRNGKey(0).
    axis : int, tuple of ints, or None, optional
        Axis or axes along which to compute statistics. Default is None.
    keepdims : bool, optional
        If True, reduced axes are retained with size 1. Default is False.
    max_samples : int, optional
        Maximum sample limit passed to mad. Default is 10000.
    return_std : bool, optional
        If True, returns a tuple of (mean_val, std_val). Default is False.

    Returns
    -------
    mean_val : Array
        Robust sigma-clipped mean.
    std_val : Array, optional
        Robust standard deviation of unclipped elements, returned if return_std is True.

    Notes
    -----
    Algorithm designed by Bernard J. Rauscher, Principal Scientist, Rauscher
    Scientific LLC. Implementation and documentation assisted by Gemini AI.
    """
    x = jnp.asarray(x)
    
    # 1. Estimate robust center (median) and scale (MAD)
    scale_mad, med = mad(
        x, key=key, axis=axis, keepdims=True, 
        return_median=True, max_samples=max_samples
    )
    
    # 2. Identify in-distribution pixels
    valid_mask = jnp.abs(x - med) <= (n_sigma * scale_mad)
    count_valid = jnp.sum(jnp.where(valid_mask, 1.0, 0.0), axis=axis, keepdims=True)
    count_clamp = jnp.maximum(count_valid, 1.0)
    
    # 3. Compute sigma-clipped mean
    sum_valid = jnp.sum(jnp.where(valid_mask, x, 0.0), axis=axis, keepdims=True)
    mean_val = sum_valid / count_clamp

    # 4. Handle output shape formatting
    def _format_shape(arr):
        if keepdims:
            return arr
        elif axis is None:
            return jnp.squeeze(arr)
        else:
            return jnp.squeeze(arr, axis=axis)

    mean_out = _format_shape(mean_val)
    
    # 5. Optionally compute standard deviation over clipped sample
    if return_std:
        sq_diff = jnp.where(valid_mask, (x - mean_val) ** 2, 0.0)
        var_val = jnp.sum(sq_diff, axis=axis, keepdims=True) / count_clamp
        std_val = jnp.sqrt(var_val)
        std_out = _format_shape(std_val)
        return mean_out, std_out

    return mean_out
    