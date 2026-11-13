def xslice(output: int, wout: int, alternating: bool = True):
    """
    Generate horizontal column slice for an HxRG detector output channel in readout order.

    Computes the column slice corresponding to a zero-indexed output channel. When
    alternating readout is enabled, odd channels are sliced in reverse order 
    (right-to-left) to match the chronological physical readout order of standard 
    HxRG detectors.

    Parameters
    ----------
    output : int
        Zero-indexed output channel number (e.g., 0 to 3 for 4-output mode, 
        or 0 to 31 for 32-output mode).
    wout : int
        Width of each output channel in pixels (e.g., 512 for a 2048-pixel detector 
        with 4 outputs, or 64 for 32 outputs).
    alternating : bool, optional
        If True, odd channels read in reverse direction (right-to-left). 
        If False, all channels read forward (left-to-right). Default is True.

    Returns
    -------
    col_slice : slice
        1D slice object for indexing the horizontal axis (e.g., ``D[..., col_slice]``).

    Notes
    -----
    Algorithm designed by Bernard J. Rauscher, Principal Scientist, Rauscher
    Scientific LLC. Implementation and documentation assisted by Gemini AI.

    Examples
    --------
    >>> # Extract channel 1 (cols 64 to 127) in chronological readout order (127 down to 64)
    >>> s = hxrg_slice(output=1, wout=64)
    >>> s
    slice(127, 63, -1)
    """
    left_col = output * wout
    right_col = left_col + wout

    if alternating and (output % 2 != 0):
        # Reverse readout: step from right_col - 1 down to left_col
        stop = left_col - 1 if left_col > 0 else None
        return slice(right_col - 1, stop, -1)

    return slice(left_col, right_col, 1)