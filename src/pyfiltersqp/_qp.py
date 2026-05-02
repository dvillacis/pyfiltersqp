"""QP subproblem via OSQP."""
from __future__ import annotations

import contextlib
import ctypes
import ctypes.util
import os
import sys

import numpy as np
import scipy.sparse as sp
import osqp


# libc fflush — used by _silence_c_stdout to flush libc's line-buffered
# stdout before restoring fd=1, so prints emitted just before solve()
# returns don't leak past the redirection window.
_LIBC = None
try:
    _libname = ctypes.util.find_library("c") or "libc.so.6"
    _LIBC = ctypes.CDLL(_libname)
    _LIBC.fflush.argtypes = [ctypes.c_void_p]
    _LIBC.fflush.restype  = ctypes.c_int
except (OSError, AttributeError):
    _LIBC = None


@contextlib.contextmanager
def _silence_c_stdout():
    """
    Silence prints emitted from C extensions (e.g. OSQP 1.1.1 polish).

    OSQP's polish module emits "Polishing not needed - no active set..."
    directly to fd=1 regardless of its ``verbose`` setting.  Redirect fd=1
    to /dev/null around the call.  Python-level prints are unaffected
    because they are buffered in ``sys.stdout`` separately.
    """
    try:
        sys.stdout.flush()
        saved_fd = os.dup(1)
    except (AttributeError, OSError):
        yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
        # Flush libc's stdout buffer before restoring fd=1 so any printf
        # the C extension queued just before returning is sunk to devnull.
        if _LIBC is not None:
            _LIBC.fflush(None)
    finally:
        os.dup2(saved_fd, 1)
        os.close(devnull)
        os.close(saved_fd)


def _wrap_osqp(solver: "osqp.OSQP") -> "osqp.OSQP":
    """Replace solver.setup/solve/update with silenced wrappers."""
    for name in ("setup", "solve", "update"):
        original = getattr(solver, name)

        def make_wrapper(orig):
            def wrapper(*args, **kwargs):
                with _silence_c_stdout():
                    return orig(*args, **kwargs)
            return wrapper

        setattr(solver, name, make_wrapper(original))
    return solver


class QPSubproblem:
    """
    *Deprecated path.*  Wraps OSQP to solve the SQP quadratic subproblem.

    Selected by ``SQPSolver(backend="osqp")``.  Materialises the dense n×n
    L-BFGS Hessian on every iteration and hands it to OSQP, which itself
    factors a dense KKT matrix per ADMM step.  This was the only path in
    v0.1; v0.4 added :class:`ImplicitLBFGSQP` and :class:`WoodburyADMM`,
    both of which avoid the dense B_k entirely and are 10–100× faster on
    every problem size we have measured.

    Retained as a fallback for the rare problem where the new backends
    fail (none observed in our test suites as of 2026-05-02).  New code
    should let ``backend="auto"`` pick implicit or woodbury automatically.

    Wraps OSQP to solve::

        min   (1/2) pᵀ B p + gᵀ p
        s.t.  J_eq   p = -c_eq           (linearised equalities)
              J_ineq p ≤ -c_ineq         (linearised inequalities)
              xl - x ≤ p ≤ xu - x        (step bounds)

    OSQP solves the equivalent canonical form::

        min   (1/2) pᵀ P p + qᵀ p
        s.t.  l ≤ A p ≤ u

    The mapping is:
        P = B,  q = g
        A = [ J_eq   ]    l = [ -c_eq              ]    u = [ -c_eq    ]
            [ J_ineq ]        [ -inf * ones(m_ineq) ]        [ -c_ineq  ]
            [ I_n    ]        [ xl - x              ]        [ xu - x   ]

    Parameters
    ----------
    n, m_eq, m_ineq : int
        Problem dimensions.
    osqp_settings : dict, optional
        Override OSQP solver settings.

    Notes
    -----
    **Iteration cost (after first setup)**

    P and A are rebuilt from scratch only on the very first call.  For all
    subsequent calls a two-level caching strategy avoids all scipy.sparse
    overhead:

    * **P** (Hessian): the CSC column-major data indices are computed once
      and stored as ``_P_csc_row / _P_csc_col``.  Subsequent calls compute
      ``P.data = B[_P_csc_row, _P_csc_col]`` directly — no ``triu_indices``
      call and no ``sp.csc_matrix`` construction.

    * **A** (constraints, dense Jacobians): on the first call A is built via
      ``sp.vstack`` and its ``data`` array is retained.  Subsequent calls
      update the Jacobian values in-place via a reshape-view stride trick::

          A_view = A.data.reshape(n, m_eq + m_ineq + 1)
          A_view[:, :m_eq]         = J_eq.T
          A_view[:, m_eq:m_eq+m_ineq] = J_ineq.T
          # identity column (A_view[:, -1]) stays 1.0 — unchanged

      This eliminates the COO→CSR→CSC conversion chain that otherwise
      accounts for ~20–45 % of ``QPSubproblem.solve()`` wall time.

    * **A** (sparse Jacobians): falls back to ``sp.vstack`` with a sparsity-
      pattern check; if the pattern changes a full ``osqp.setup()`` is issued.
    """

    _DEFAULT_OSQP = dict(
        warm_starting=True,
        eps_abs=1e-8,
        eps_rel=1e-8,
        max_iter=4000,
        # OSQP 1.1.x's polish step writes "Polishing not needed..." to fd=1
        # regardless of `verbose`, even with fd-level redirection (the C
        # library appears to capture fd=1 at import).  Disable polish for the
        # OSQP path; the implicit and Woodbury backends each apply their own
        # Schur-complement polish for multiplier accuracy.
        polishing=False,
        verbose=False,
    )

    def __init__(self, n: int, m_eq: int, m_ineq: int,
                 osqp_settings: dict | None = None):
        self.n = n
        self.m_eq = m_eq
        self.m_ineq = m_ineq
        self._solver: osqp.OSQP | None = None
        self._settings = {**self._DEFAULT_OSQP, **(osqp_settings or {})}

        # --- P-matrix cache ---
        # CSC column-major row/col indices for the upper-triangular P.
        # P.data[k] = B[_P_csc_row[k], _P_csc_col[k]]  for k in 0..nnz-1.
        # Computed once on the first solve() call.
        self._P_csc_row: np.ndarray | None = None
        self._P_csc_col: np.ndarray | None = None

        # --- A-matrix cache (dense Jacobians) ---
        # Direct reference to A.data so we can update Jacobian values in-place.
        # None until the first solve(), or when Jacobians are sparse.
        self._A_dense_data: np.ndarray | None = None
        self._A_dense_stride: int = 0   # m_eq + m_ineq + 1

        # --- A-matrix cache (sparse Jacobians) ---
        # Stored pattern of A from the most recent setup() call.
        # Used to detect structure changes that require a full re-setup.
        self._A_ref_nnz: int = 0
        self._A_ref_indptr: np.ndarray | None = None
        self._A_ref_indices: np.ndarray | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _build_P_csc(self, B: np.ndarray) -> sp.csc_matrix:
        """
        Build the upper-triangular P matrix in CSC format.

        On the very first call this also caches ``_P_csc_row / _P_csc_col``
        so that subsequent calls can bypass the scipy.sparse construction.
        """
        n = self.n
        if self._P_csc_row is None:
            # CSC upper-triangular: column j contains rows 0 .. j.
            # _P_csc_col = [0, 1,1, 2,2,2, …]
            # _P_csc_row = [0, 0,1, 0,1,2, …]
            self._P_csc_col = np.repeat(np.arange(n), np.arange(1, n + 1))
            self._P_csc_row = np.concatenate([np.arange(j + 1) for j in range(n)])

        P_data = B[self._P_csc_row, self._P_csc_col]
        return sp.csc_matrix(
            (P_data, (self._P_csc_row, self._P_csc_col)), shape=(n, n)
        )

    @staticmethod
    def _dense_csr(mat: np.ndarray, nrows: int, ncols: int) -> sp.csr_matrix:
        """CSR block with every entry as a structural non-zero (constant pattern)."""
        rows = np.repeat(np.arange(nrows), ncols)
        cols = np.tile(np.arange(ncols), nrows)
        return sp.csr_matrix((mat.ravel(), (rows, cols)), shape=(nrows, ncols))

    def _build_A_csc(self, J_eq, J_ineq) -> sp.csc_matrix:
        """Build A from scratch via sp.vstack (used on first call and sparse path)."""
        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq
        blocks = []
        if m_eq > 0:
            blocks.append(J_eq.tocsr() if sp.issparse(J_eq)
                          else self._dense_csr(J_eq, m_eq, n))
        if m_ineq > 0:
            blocks.append(J_ineq.tocsr() if sp.issparse(J_ineq)
                          else self._dense_csr(J_ineq, m_ineq, n))
        blocks.append(sp.eye(n, format="csr"))
        return sp.vstack(blocks, format="csc")

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def solve(
        self,
        B: np.ndarray,
        g: np.ndarray,
        c_eq: np.ndarray,
        J_eq,
        c_ineq: np.ndarray,
        J_ineq,
        x: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """
        Solve the QP subproblem.

        Returns
        -------
        p : ndarray, shape (n,)
            Primal step (zero vector on failure).
        lam_eq : ndarray, shape (m_eq,)
            Equality multipliers from QP.
        lam_ineq : ndarray, shape (m_ineq,)
            Inequality multipliers from QP (≥ 0 at solution).
        lam_lb : ndarray, shape (n,)
            Lower-bound multipliers (≥ 0).
        lam_ub : ndarray, shape (n,)
            Upper-bound multipliers (≥ 0).
        success : bool
            False if OSQP reported infeasible or error.
        """
        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq
        INF = 1e30

        jac_is_dense = (
            (m_eq   == 0 or not sp.issparse(J_eq))
            and (m_ineq == 0 or not sp.issparse(J_ineq))
        )

        # --- Build bounds l, u (always fast — just NumPy concatenation) ---
        l_parts, u_parts = [], []
        if m_eq > 0:
            l_parts.append(-c_eq);           u_parts.append(-c_eq)
        if m_ineq > 0:
            l_parts.append(np.full(m_ineq, -INF)); u_parts.append(-c_ineq)
        l_parts.append(xl - x);              u_parts.append(xu - x)
        l = np.concatenate(l_parts)
        u = np.concatenate(u_parts)

        # ---------------------------------------------------------------
        # FIRST CALL — build P and A from scratch, set up OSQP
        # ---------------------------------------------------------------
        if self._solver is None:
            P = self._build_P_csc(B)
            A = self._build_A_csc(J_eq, J_ineq)

            # Cache A for fast in-place updates (dense path only)
            if jac_is_dense:
                self._A_dense_data   = A.data                 # retain buffer
                self._A_dense_stride = m_eq + m_ineq + 1

            # Cache A pattern for sparse path
            self._A_ref_nnz     = A.nnz
            self._A_ref_indptr  = A.indptr.copy()
            self._A_ref_indices = A.indices.copy()

            self._solver = _wrap_osqp(osqp.OSQP())
            self._solver.setup(P, g, A, l, u, **self._settings)

        # ---------------------------------------------------------------
        # SUBSEQUENT CALLS — fast in-place update (dense Jacobians)
        # ---------------------------------------------------------------
        elif jac_is_dense and self._A_dense_data is not None:
            # P.data: compute CSC-ordered values without building sp.csc_matrix
            P_data = B[self._P_csc_row, self._P_csc_col]

            # A.data: update Jacobian values in-place via stride view.
            # A_dv[j, :] = [J_eq[:,j], J_ineq[:,j], 1.0] for each column j.
            # Updating A_dv modifies A.data (shared buffer) → OSQP sees new values.
            stride = self._A_dense_stride
            A_dv = self._A_dense_data.reshape(n, stride)
            if m_eq > 0:
                A_dv[:, :m_eq] = J_eq.T
            if m_ineq > 0:
                A_dv[:, m_eq:m_eq + m_ineq] = J_ineq.T
            # A_dv[:, -1] stays 1.0 (identity — never changes)

            self._solver.update(q=g, l=l, u=u,
                                Px=P_data,
                                Ax=self._A_dense_data)

        # ---------------------------------------------------------------
        # SUBSEQUENT CALLS — sparse Jacobians (or type switch from dense)
        # ---------------------------------------------------------------
        else:
            P_data = B[self._P_csc_row, self._P_csc_col]
            A = self._build_A_csc(J_eq, J_ineq)

            same_pattern = (
                A.nnz == self._A_ref_nnz
                and np.array_equal(A.indptr,  self._A_ref_indptr)
                and np.array_equal(A.indices, self._A_ref_indices)
            )
            if same_pattern:
                self._solver.update(q=g, l=l, u=u, Px=P_data, Ax=A.data)
            else:
                # Pattern changed — re-setup OSQP from scratch
                self._A_ref_nnz     = A.nnz
                self._A_ref_indptr  = A.indptr.copy()
                self._A_ref_indices = A.indices.copy()
                P = sp.csc_matrix(
                    (P_data, (self._P_csc_row, self._P_csc_col)), shape=(n, n)
                )
                self._solver = _wrap_osqp(osqp.OSQP())
                self._solver.setup(P, g, A, l, u, **self._settings)

        # ---------------------------------------------------------------
        # Extract solution
        # ---------------------------------------------------------------
        res = self._solver.solve()

        # OSQP v1.x uses space-separated strings ("solved inaccurate", etc.)
        # Accept both spellings for compatibility across versions.
        _ok = {"solved", "solved_inaccurate", "solved inaccurate",
               "maximum_iterations_reached", "maximum iterations reached"}
        if res.info.status in _ok:
            p = np.asarray(res.x)
            y = np.asarray(res.y)
            # OSQP convention: Bp + g + Aᵀ y = 0  →  y IS the NLP multiplier
            # Dual variable layout: [y_eq | y_ineq | y_bounds]
            lam_eq   = y[:m_eq]
            lam_ineq = y[m_eq:m_eq + m_ineq]
            y_bounds = y[m_eq + m_ineq:]
            # NLP: g + J_eqᵀ λ_eq + J_ineqᵀ λ_ineq − λ_lb + λ_ub = 0
            # y_bounds = λ_ub − λ_lb  → split into non-negative components
            lam_lb = np.maximum(0.0, -y_bounds)
            lam_ub = np.maximum(0.0,  y_bounds)
            return p, lam_eq, lam_ineq, lam_lb, lam_ub, True
        else:
            return (np.zeros(n), np.zeros(m_eq), np.zeros(m_ineq),
                    np.zeros(n), np.zeros(n), False)

    def solve_soc(
        self,
        c_eq_trial: np.ndarray,
        c_ineq_trial: np.ndarray,
        x: np.ndarray,
        p: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """
        Second-order correction (SOC) step.

        Re-solves the same QP (same Hessian B, gradient g, and constraint
        Jacobians A) but with constraint values from the trial point x+p.
        The SOC step d approximates:

            J_eq   d = -c_eq(x+p)
            J_ineq d ≤ -c_ineq(x+p)
            xl-(x+p) ≤ d ≤ xu-(x+p)

        Must be called immediately after a successful ``solve()`` while the
        OSQP instance still holds the correct P (Hessian) and A (Jacobians).

        Returns
        -------
        d : ndarray, shape (n,)
            SOC correction vector (zero on failure).
        success : bool
        """
        if self._solver is None:
            return np.zeros(self.n), False

        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq
        INF = 1e30

        # Update only l, u — P and A are already correct from the last solve().
        l_parts, u_parts = [], []
        if m_eq > 0:
            l_parts.append(-c_eq_trial);              u_parts.append(-c_eq_trial)
        if m_ineq > 0:
            l_parts.append(np.full(m_ineq, -INF));    u_parts.append(-c_ineq_trial)
        l_parts.append(xl - (x + p));                 u_parts.append(xu - (x + p))

        self._solver.update(l=np.concatenate(l_parts), u=np.concatenate(u_parts))
        res = self._solver.solve()

        _ok = {"solved", "solved_inaccurate", "solved inaccurate",
               "maximum_iterations_reached", "maximum iterations reached"}
        if res.info.status in _ok:
            return np.asarray(res.x), True
        return np.zeros(n), False
