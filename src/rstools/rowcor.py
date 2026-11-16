from ._backend import get_backend
from .stats import rmean


def rowcor(D, nout=4, top_only=False, sigrej=4.0, return_R=False):
    """
    Perform row-by-row reference correction on 3D ramp data.
    This is the most basic form of HxRG reference correction.

    Parameters
    ----------
    D : array_like
        Input 3D array with shape (nz, ny, nx) (NumPy or CuPy ndarray).
    nout : int, optional
        Number of output channels across the horizontal axis. Default is 4.
    top_only : bool, optional
        If True, computes correction using only the top reference rows. 
        If False, linearly interpolates between top and bottom reference rows. 
        Default is False.
    sigrej : float, optional
        Sigma threshold for outlier rejection in reference row averaging. 
        Default is 4.0.
    return_R : bool, optional
        If True, also returns the subtracted reference correction profile ``R``.
        Default is False.

    Returns
    -------
    D_corrected : ndarray
        Reference-corrected 3D data array with shape (nz, ny, nx).
    R : ndarray, optional
        Subtracted reference correction profile (returned only when ``return_R=True``).
        Shape is ``(nz, ny, nout)`` when ``top_only=False``, or ``(nz, 1, nout)``
        when ``top_only=True``.

    Notes
    -----
    Algorithm designed by Bernard J. Rauscher, Principal Scientist, Rauscher
    Scientific LLC. Implementation and documentation assisted by Gemini AI.
    """
    xp = get_backend(D)
    D = xp.asarray(D, dtype=xp.float32)
    nz, ny, nx = D.shape
    nx_per_out = nx // nout

    # 1. Extract inner two Top reference rows and compute channel means
    # Shape: (nz, 2, nout, nx_per_out) -> (nz, nout, 2 * nx_per_out)
    T = D[:, -3:-1, :].reshape(nz, 2, nout, nx_per_out).transpose(0, 2, 1, 3).reshape(nz, nout, -1)
    μ_T = rmean(T, n_sigma=sigrej, axis=2, keepdims=True).reshape(nz, 1, nout, 1)

    # 2. Determine correction profile R
    if top_only:
        R = μ_T
    else:
        # Extract inner two Bottom reference rows
        B = D[:, 1:3, :].reshape(nz, 2, nout, nx_per_out).transpose(0, 2, 1, 3).reshape(nz, nout, -1)
        μ_B = rmean(B, n_sigma=sigrej, axis=2, keepdims=True).reshape(nz, 1, nout, 1)

        # Pre-normalized linear weight profile w in range [0, 1] over row height
        w = ((xp.arange(ny, dtype=xp.float32) - 1.5) / (ny - 4.0)).reshape(1, ny, 1, 1)
        R = μ_B + (μ_T - μ_B) * w

    # 3. Apply reference correction
    D_corrected = (D.reshape(nz, ny, nout, nx_per_out) - R).reshape(nz, ny, nx)

    if return_R:
        return D_corrected, R

    return D_corrected