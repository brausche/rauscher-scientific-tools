"""Utilities for converting between spatial HxRG ramp cubes and readout time series."""

from __future__ import annotations

from typing import TypeVar

ArrayLike = TypeVar("ArrayLike")


def to_timeseries(
    cube: ArrayLike,
    n_out: int = 32,
    flip_odd: bool = True,
) -> ArrayLike:
    """Convert a 3D spatial ramp cube into per-output readout time series.

    Parameters
    ----------
    cube : array_like
        Input up-the-ramp datacube of shape ``(nz, ny, nx)`` (NumPy or CuPy).
    n_out : int, default=32
        Number of parallel readout outputs across the ``nx`` axis.
    flip_odd : bool, default=True
        If True, reverse the column scan direction of odd-indexed outputs
        (``1, 3, ...``), matching the alternating Roman/HxRG readout convention.

    Returns
    -------
    timeseries : array_like
        Array of shape ``(nz, n_out, ny * (nx // n_out))`` ordered by sample clock.
    """
    if cube.ndim != 3:
        raise ValueError(f"Expected a 3D array (nz, ny, nx), got shape {cube.shape}.")

    nz, ny, nx = cube.shape
    if nx % n_out != 0:
        raise ValueError(f"Cube width ({nx}) must be divisible by n_out ({n_out}).")

    w_out = nx // n_out

    # Transpose into readout axis order and materialize a single contiguous copy
    # so the input array is never mutated in place.
    ts = cube.reshape(nz, ny, n_out, w_out).transpose(0, 2, 1, 3).copy()

    if flip_odd:
        ts[:, 1::2, :, :] = ts[:, 1::2, :, ::-1].copy()

    return ts.reshape(nz, n_out, ny * w_out)


def from_timeseries(
    timeseries: ArrayLike,
    ny: int = 4096,
    flip_odd: bool = True,
) -> ArrayLike:
    """Reconstruct a 3D spatial ramp cube from per-output readout time series.

    Parameters
    ----------
    timeseries : array_like
        Input time-series array of shape ``(nz, n_out, ny * w_out)`` (NumPy or CuPy).
    ny : int, default=4096
        Number of detector rows along the spatial Y axis.
    flip_odd : bool, default=True
        If True, un-flip the column scan direction of odd-indexed outputs
        (``1, 3, ...``).

    Returns
    -------
    cube : array_like
        Spatial datacube of shape ``(nz, ny, n_out * w_out)``.
    """
    if timeseries.ndim != 3:
        raise ValueError(
            f"Expected a 3D array (nz, n_out, n_samples), got shape {timeseries.shape}."
        )

    nz, n_out, n_samples = timeseries.shape
    if n_samples % ny != 0:
        raise ValueError(
            f"Time-series length ({n_samples}) must be divisible by ny ({ny})."
        )

    w_out = n_samples // ny

    cube = timeseries.reshape(nz, n_out, ny, w_out).transpose(0, 2, 1, 3).copy()

    if flip_odd:
        cube[:, :, 1::2, :] = cube[:, :, 1::2, ::-1].copy()

    return cube.reshape(nz, ny, n_out * w_out)