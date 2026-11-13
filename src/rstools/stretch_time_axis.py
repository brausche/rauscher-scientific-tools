import scipy.ndimage as cpu_ndimage
from ._backend import HAS_GPU, get_backend

if HAS_GPU:
    try:
        import cupy as cp
        import cupyx.scipy.ndimage as cpx_ndimage
    except ImportError:
        cpx_ndimage = None
else:
    cpx_ndimage = None


def stretch_time_axis(D, n_target: int = 655360, order: int = 1):
    """
    Stretch the fast-scan time axis of an HxRG datacube via spline interpolation.

    Parameters
    ----------
    D : ndarray
        Datacube with shape (nz, nout, npix_in), e.g. (55, 32, 524288).
    n_target : int, optional
        Target length for the time axis. Default is 655360.
    order : int, optional
        Spline interpolation order. Default is 1 (linear).

    Returns
    -------
    D_stretched : ndarray
        Stretched array with shape (nz, nout, n_target) in identical dtype and backend.
    """
    xp = get_backend(D)
    dtype = D.dtype

    nz, nout, n_orig = D.shape
    zoom_tuple = (1.0, n_target / n_orig)

    ndimage = (
        cpx_ndimage
        if (HAS_GPU and isinstance(D, cp.ndarray) and cpx_ndimage is not None)
        else cpu_ndimage
    )

    out = xp.empty((nz, nout, n_target), dtype=dtype)

    for z in range(nz):
        out[z] = ndimage.zoom(
            D[z],
            zoom=zoom_tuple,
            order=order,
            mode="nearest",
        )

    return out