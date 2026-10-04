import numpy as np
import jax.numpy as jnp

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
    def __init__(self, nz: int, deg: int = 1, drop_first: int = 1):

        # Save params
        self.nz = nz
        self.deg = deg
        self.drop_first = drop_first

        # Check for errors
        if drop_first >= nz:
            raise ValueError(f"drop_first ({drop_first}) must be less than total samples nz ({nz}).")

        # Map the full sequence of nz samples (including dropped samples) onto the closed interval [-1.0, +1.0]
        full_x = np.linspace(-1.0, 1.0, nz, dtype=np.float64)

        # Slice off dropped frames, retaining domain coordinates for active samples
        x = full_x[drop_first:]

        # Evaluate Legendre polynomials over active sample locations
        B_cpu = np.polynomial.legendre.legvander(x, deg)

        # The Moore-Penrose pseudoinverse does fitting
        F_cpu = np.linalg.pinv(B_cpu)

        # Also make a matrix that builds models
        M_cpu = np.matmul(B_cpu, F_cpu)

        # Convert everything to JAX, 32-bit
        self.x = x
        self.B = jnp.array(B_cpu, dtype=jnp.float32) # Basis matrix
        self.F = jnp.array(F_cpu, dtype=jnp.float32) # Fitting matrix
        self.M = jnp.array(M_cpu, dtype=jnp.float32) # Modeling matrix

    # ------------------------------------------------------------------
    # JAX PyTree Registration Hooks
    # ------------------------------------------------------------------
    def tree_flatten(self):
        # Dynamic JAX arrays traced on CPU/GPU
        children = (self.x, self.B, self.F, self.M)
        # Static metadata (determines shape/structure)
        aux_data = (self.nz, self.deg, self.drop_first)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = object.__new__(cls)
        obj.x, obj.B, obj.F, obj.M = children
        obj.nz, obj.deg, obj.drop_first = aux_data
        return obj