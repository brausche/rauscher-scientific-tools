# Legendre polynomial fitting and modeling operators for up-the-ramp data.
# NumPy serves as the fallback for CuPy, but NumPy always remains pure NumPy.

import numpy as np

try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    HAS_GPU = True
except Exception:
    cp = None
    HAS_GPU = False

def get_backend(*args):
    """Returns cupy if any arg is on GPU, otherwise numpy."""
    if HAS_GPU:
        return cp.get_array_module(*args)
    return np

def to_cpu(x):
    """Always returns a numpy array on CPU."""
    return x.get() if (HAS_GPU and isinstance(x, cp.ndarray)) else np.asarray(x)

def to_gpu(x):
    """Always returns a cupy array on GPU (falls back to numpy if no GPU)."""
    return cp.asarray(x) if HAS_GPU else np.asarray(x)


class Legendre:
    """
    Legendre polynomial basis and operator matrices for up-the-ramp data.

    Parameters
    ----------
    nz : int
        Total number of samples up the ramp (indices 0 to nz - 1).
    deg : int, optional
        Polynomial degree of the fit. Default is 1.
    drop_first : int, optional
        Number of initial samples to exclude from fitting. Default is 1.
    use_gpu : bool, optional
        If True and a GPU is available, place matrices in GPU VRAM (CuPy).
        If False, retain them in host system RAM (NumPy). Default is True.

    Attributes
    ----------
    nz : int
        Total samples up the ramp.
    deg : int
        Polynomial degree.
    drop_first : int
        Initial samples dropped.
    x : Array
        Normalized domain coordinates for active samples in range [-1, +1].
    B : Array
        Basis matrix mapping polynomial coefficients to sample values.
    F : Array
        Fitting matrix (pseudoinverse of B) mapping sample values to coefficients.
    M : Array
        Modeling matrix (hat matrix, B @ F) mapping sample values directly to model values.

    Notes
    -----
    1) Algorithm designed by Bernard J. Rauscher, Principal Scientist, Rauscher
       Scientific LLC. Implementation and documentation assisted by Gemini AI.
    2) This algorithm treats the last RESET frame as frame zero. By default it assumes
       no data is returned. For e.g. JWST NIRCam files having 10 up-the-ramp frames,
       set nz = 11 and leave drop_first = 1 (the default).
    """
    def __init__(self, nz: int, deg: int = 1, drop_first: int = 1, use_gpu: bool = True):

        # Save params
        self.nz = nz
        self.deg = deg
        self.drop_first = drop_first

        # Check for errors
        if drop_first >= nz:
            raise ValueError(f"drop_first ({drop_first}) must be less than total samples nz ({nz}).")

        # Map full sequence onto [-1.0, +1.0] using pure NumPy float64 for precision
        full_x = np.linspace(-1.0, 1.0, nz, dtype=np.float64)

        # Slice off dropped frames, retaining domain coordinates for active samples
        x_cpu = full_x[drop_first:]

        # Evaluate Legendre polynomials on CPU (legvander is a NumPy function)
        B_cpu = np.polynomial.legendre.legvander(x_cpu, deg)

        # The Moore-Penrose pseudoinverse does fitting
        F_cpu = np.linalg.pinv(B_cpu)

        # Also make a matrix that builds models
        M_cpu = B_cpu @ F_cpu

        # Cast to float32 for pipeline execution
        x_cpu = x_cpu.astype(np.float32)
        B_cpu = B_cpu.astype(np.float32)
        F_cpu = F_cpu.astype(np.float32)
        M_cpu = M_cpu.astype(np.float32)

        # Store arrays on requested target device (GPU or CPU)
        target = to_gpu if use_gpu else to_cpu
        self.x = target(x_cpu)
        self.B = target(B_cpu)
        self.F = target(F_cpu)
        self.M = target(M_cpu)

    def to_cpu(self):
        """Move all operator matrices to host system RAM (NumPy) in-place."""
        self.x = to_cpu(self.x)
        self.B = to_cpu(self.B)
        self.F = to_cpu(self.F)
        self.M = to_cpu(self.M)
        return self

    def to_gpu(self):
        """Move all operator matrices to GPU VRAM (CuPy) in-place."""
        self.x = to_gpu(self.x)
        self.B = to_gpu(self.B)
        self.F = to_gpu(self.F)
        self.M = to_gpu(self.M)
        return self

    @property
    def backend(self):
        """Returns the module (cupy or numpy) where the matrices currently reside."""
        return get_backend(self.B)