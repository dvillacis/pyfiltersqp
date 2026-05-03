"""L-BFGS Hessian approximation with Powell damping."""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Optional Numba JIT for the BFGS forward recursion (_precompute_Bs).
#
# Profile (broyden_tridiagonal n=1000, post Phase B): ``_precompute_Bs`` is
# 40 % of total wall time — pure interpreter overhead from the m² Python
# loop, not numpy work.  Numba lifts that loop into compiled code for a
# 5–10× speedup on the function and ~30 % on the outer SQP wall time.
#
# Numba is an OPTIONAL dependency.  Install with:
#     pip install "pyfiltersqp[numba]"
# When unavailable (or import fails for any reason), ``_precompute_Bs``
# falls back to the pure-NumPy implementation below.  No behavioural
# difference — only speed.
# ---------------------------------------------------------------------------

try:                                              # pragma: no cover - import path
    import warnings as _warnings
    from numba import njit
    from numba.core.errors import NumbaPerformanceWarning

    # Numba's type inference for ``F_array[:, j]`` is conservative — it
    # marks the slice as 'any-layout' and emits a NumbaPerformanceWarning
    # for ``np.dot``, even though at runtime the slice is contiguous and
    # routes to BLAS.  Silence the false positive once at import.
    _warnings.filterwarnings(
        "ignore",
        message=r"np\.dot\(\) is faster on contiguous arrays.*",
        category=NumbaPerformanceWarning,
    )

    @njit(cache=True, fastmath=True)
    def _precompute_Bs_jit(S, Y, sigma):
        """
        JIT kernel: compute (Bs, beta) from stacked S, Y of shape (n, m).

        Bs[:, i] = B_i @ s_i        (i = 0..m-1)
        beta[i]  = s_iᵀ Bs[:, i]

        Long-vector dot products go through ``np.dot``, which numba
        dispatches to BLAS — important for large n where hand-rolled
        scalar loops can't beat OpenBLAS SIMD.  The outer i,j loop is
        what the Python interpreter was paying for; numba lifts that.

        Pre-extracts column slices into 1-D arrays via ``.copy()`` so
        ``np.dot`` sees contiguous inputs (numba's type inference
        otherwise marks the slice 'any-layout' and falls back to a
        slower path with a NumbaPerformanceWarning).
        """
        n, m = S.shape
        Bs   = np.empty((n, m))
        beta = np.empty(m)
        sy   = np.empty(m)

        # Pre-extract contiguous columns once (m × n, m ≤ 10).
        S_cols = [S[:, j].copy() for j in range(m)]
        Y_cols = [Y[:, j].copy() for j in range(m)]

        for j in range(m):
            sy[j] = np.dot(Y_cols[j], S_cols[j])

        YtS = Y.T @ S        # (m, m)

        for i in range(m):
            r = sigma * S_cols[i].copy()
            for j in range(i):
                alpha = np.dot(S_cols[j], r) / beta[j]
                gamma = YtS[j, i] / sy[j]
                r += -alpha * Bs[:, j] + gamma * Y_cols[j]
            Bs[:, i] = r
            beta[i]  = np.dot(S_cols[i], r)

        return Bs, beta

    _HAS_NUMBA = True
except ImportError:                               # pragma: no cover
    _precompute_Bs_jit = None
    _HAS_NUMBA = False


class LBFGSMemory:
    """
    Stores the last *memory* (s, y) pairs and computes the L-BFGS
    Hessian B_k and its inverse H_k for the QP subproblem.

    The L-BFGS update approximates the Lagrangian Hessian:
        B_{k+1} ≈ ∇²_xx L(x_{k+1}, λ_{k+1})

    Powell damping is applied to guarantee y_kᵀ s_k > 0 even when the
    true Lagrangian Hessian is indefinite (common near MPCC solutions).

    Parameters
    ----------
    n : int
        Number of variables.
    memory : int
        Number of (s, y) pairs to retain (default 10).
    init_scale : float
        Initial diagonal scaling H_0 = scale * I (default 1.0).
        After the first update, the scale is set to sᵀy / yᵀy  (N&W eq. 7.20).
    """

    def __init__(self, n: int, memory: int = 10, init_scale: float = 1.0):
        self.n = n
        self.memory = memory
        self._s: list[np.ndarray] = []   # s_k = x_{k+1} - x_k
        self._y: list[np.ndarray] = []   # y_k = ∇L_{k+1} - ∇L_k (damped)
        self._scale = init_scale

    def reset(self) -> None:
        """Discard all stored pairs (e.g. between outer MPCC iterations)."""
        self._s.clear()
        self._y.clear()
        self._scale = 1.0

    def update(self, s: np.ndarray, y_raw: np.ndarray) -> None:
        """
        Add a new (s, y) pair with Powell damping.

        Parameters
        ----------
        s : ndarray, shape (n,)
            Step: x_{k+1} - x_k.
        y_raw : ndarray, shape (n,)
            Lagrangian gradient difference BEFORE damping:
            ∇_x L(x_{k+1}, λ_{k+1}) - ∇_x L(x_k, λ_{k+1}).
        """
        s_norm = np.linalg.norm(s)
        if s_norm < 1e-14:
            return  # degenerate step — skip update

        # Powell damping: ensure y_dampedᵀ s > 0.
        # Use matvec(s) = B_k s for the curvature estimate (not _matvec which
        # gives H_k s = B_k^{-1} s — the wrong direction for the damping check).
        Bs = self.matvec(s)
        sBs = float(s @ Bs)
        sy = float(s @ y_raw)

        theta = 1.0
        if sy < 0.2 * sBs:
            theta = 0.8 * sBs / (sBs - sy)

        y = theta * y_raw + (1.0 - theta) * Bs

        sy_damped = float(s @ y)
        if sy_damped <= 0:
            return  # should not happen after damping, but guard anyway

        # Skip curvature-deficient pairs (near-orthogonal s and y).
        # When the curvature ratio is tiny the L-BFGS scale diverges,
        # causing oscillating Hessian estimates near constrained optima.
        # Threshold 1e-8 relative to ||s||*||y|| (standard practice).
        y_norm = np.linalg.norm(y)
        if sy_damped < 1e-8 * s_norm * y_norm:
            return

        # Update scale estimate: γ_k = sᵀy / yᵀy  (Nocedal & Wright eq. 7.20).
        # This sets H_0 = γ_k * I for the next two-loop recursion.
        # When curvature is high (‖y‖ >> ‖s‖), γ_k is small → conservative step.
        # Use EMA to smooth noisy curvature estimates near constrained optima.
        # Clamped to [1e-6, 1e6] to prevent degenerate extremes.
        new_scale = float(np.clip(sy_damped / (y @ y), 1e-6, 1e6))
        self._scale = 0.7 * self._scale + 0.3 * new_scale

        # Store pair, evicting oldest if at capacity
        if len(self._s) == self.memory:
            self._s.pop(0)
            self._y.pop(0)
        self._s.append(s.copy())
        self._y.append(y.copy())

    # ------------------------------------------------------------------
    # Hessian matvec (B_k v) — O(n · m²) precompute, O(n · m) apply
    # ------------------------------------------------------------------

    def _precompute_Bs(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Precompute Bs[:, i] = B_i @ s_i and beta[i] = s_iᵀ Bs[:, i] for i = 0..m-1.

        Used by ``matvec``, ``rank2_factors``, and ``materialise``.
        Cost: O(n · m²).

        Returns
        -------
        Bs   : ndarray, shape (n, m).  Column i is B_i s_i.
        beta : ndarray, shape (m,).
        """
        m = len(self._s)
        if m == 0:
            return np.empty((self.n, 0)), np.empty(0)

        sigma = 1.0 / self._scale   # B_0 = σI
        # Fortran order so S[:, j] / Y[:, j] are contiguous — np.dot inside
        # the JIT kernel hits BLAS only on contiguous vectors (else falls
        # back to a slower path with a NumbaPerformanceWarning).
        S = np.asfortranarray(np.column_stack(self._s))   # (n, m)
        Y = np.asfortranarray(np.column_stack(self._y))   # (n, m)

        if _HAS_NUMBA:
            return _precompute_Bs_jit(S, Y, sigma)

        n = self.n
        Bs   = np.empty((n, m))
        beta = np.empty(m)
        sy   = np.einsum("ij,ij->j", Y, S)         # (m,)

        for i in range(m):
            r = sigma * S[:, i]
            for j in range(i):
                alpha = (S[:, j] @ r) / beta[j]
                gamma = (Y[:, j] @ S[:, i]) / sy[j]
                r = r - alpha * Bs[:, j] + gamma * Y[:, j]
            Bs[:, i] = r
            beta[i]  = S[:, i] @ r

        return Bs, beta

    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Compute B_k @ v using the BFGS forward recursion.

        Applies the sequence of BFGS rank-2 updates starting from B_0 = σI
        (σ = 1 / scale, so H_0 = scale · I → B_0 = σ · I)::

            B_{i+1} v = B_i v
                      − (B_i sᵢ)(B_i sᵢ)ᵀ v / (sᵢᵀ B_i sᵢ)
                      + yᵢ yᵢᵀ v              / (yᵢᵀ sᵢ)

        The term yᵢᵀ v always uses the original v; sᵢᵀ r uses the evolving
        r = B_i v at step i, which equals (B_i sᵢ)ᵀ v = Bs[i]ᵀ v by symmetry.

        Cost: O(n · memory²) setup + O(n · memory) per call.
        """
        m = len(self._s)
        sigma = 1.0 / self._scale

        if m == 0:
            return sigma * v

        Bs, beta = self._precompute_Bs()

        r = sigma * v
        for i in range(m):
            sy_i = float(self._y[i] @ self._s[i])
            r = (r
                 - Bs[:, i] * float(self._s[i] @ r) / beta[i]
                 + self._y[i] * float(self._y[i] @ v) / sy_i)

        return r

    # ------------------------------------------------------------------
    # Inverse-Hessian matvec (H_k v = B_k^{-1} v) — two-loop recursion
    # ------------------------------------------------------------------

    def _matvec(self, v: np.ndarray) -> np.ndarray:
        """
        Compute H_k @ v = B_k^{-1} @ v using the L-BFGS two-loop recursion
        (N&W Algorithm 7.4).

        This is the INVERSE Hessian applied to v.  Use ``matvec`` for the
        forward Hessian B_k @ v.

        This is O(n * memory) — cheap even for large n.
        """
        m = len(self._s)
        if m == 0:
            return self._scale * v

        alpha = np.empty(m)
        rho = np.empty(m)
        q = v.copy()

        for i in range(m - 1, -1, -1):
            rho[i] = 1.0 / (self._y[i] @ self._s[i])
            alpha[i] = rho[i] * (self._s[i] @ q)
            q -= alpha[i] * self._y[i]

        r = self._scale * q

        for i in range(m):
            beta = rho[i] * (self._y[i] @ r)
            r += (alpha[i] - beta) * self._s[i]

        return r

    def _matvec_batch(self, V: np.ndarray) -> np.ndarray:
        """
        Apply H_k = B_k⁻¹ to each column of V in O(n · m · k).

        Equivalent to ``np.column_stack([_matvec(V[:, j]) for j in range(k)])``
        but executes the L-BFGS two-loop recursion once with NumPy
        matrix-vector ops over all k columns simultaneously.  This is
        ~m × faster than k separate ``_matvec`` calls when m is small
        (the typical L-BFGS regime, m ≤ 10).

        Parameters
        ----------
        V : ndarray, shape (n,) or (n, k)
            Input vector(s).  A 1-D input is returned 1-D for convenience.
        """
        squeeze = V.ndim == 1
        if squeeze:
            Q = V.reshape(-1, 1).astype(float, copy=True)
        else:
            Q = V.astype(float, copy=True)

        m = len(self._s)
        if m == 0:
            R = self._scale * Q
            return R.ravel() if squeeze else R

        S = np.column_stack(self._s)              # n × m
        Y = np.column_stack(self._y)              # n × m
        rho_arr = 1.0 / np.einsum("ij,ij->j", Y, S)   # m
        k = Q.shape[1]
        alpha = np.empty((m, k))

        # Backward pass (m × n × k flops, vectorised over k)
        for i in range(m - 1, -1, -1):
            alpha[i] = rho_arr[i] * (S[:, i] @ Q)         # k
            Q -= np.outer(Y[:, i], alpha[i])              # n × k

        R = self._scale * Q

        # Forward pass
        for i in range(m):
            beta = rho_arr[i] * (Y[:, i] @ R)             # k
            R += np.outer(S[:, i], alpha[i] - beta)

        return R.ravel() if squeeze else R

    # ------------------------------------------------------------------
    # Compact representation (σ, Ψ, M) — for downstream structured solvers
    # ------------------------------------------------------------------

    def factors(self) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Return the compact L-BFGS Hessian representation B_k = σI + Ψ M Ψᵀ.

        With Ψ = [Y_k, σ S_k] ∈ ℝ^{n × 2m} and M ∈ ℝ^{2m × 2m},
        this is the outer-product form of Byrd, Lu, Nocedal & Zhu (1994).

        M is recovered numerically from::

            G M G = Ψᵀ B_k Ψ − σ G,   G = Ψᵀ Ψ ∈ ℝ^{2m × 2m}

        using ``matvec`` to compute B_k Ψ column by column.

        Returns
        -------
        sigma : float
            Diagonal scale σ = 1 / scale (B_0 = σI).
        Psi : ndarray, shape (n, 2*m)
            Low-rank factor.  Empty (n, 0) when m == 0.
        M : ndarray, shape (2*m, 2*m)
            Dense coefficient matrix.  Empty (0, 0) when m == 0.
        """
        m = len(self._s)
        sigma = 1.0 / self._scale

        if m == 0:
            return sigma, np.empty((self.n, 0)), np.empty((0, 0))

        S   = np.column_stack(self._s)           # n × m
        Y   = np.column_stack(self._y)           # n × m
        Psi = np.hstack([Y, sigma * S])           # n × 2m

        # B_k Ψ: apply matvec to each column of Ψ
        BPsi = np.column_stack([self.matvec(Psi[:, j]) for j in range(2 * m)])

        G    = Psi.T @ Psi          # 2m × 2m,  Ψᵀ Ψ
        PtBP = Psi.T @ BPsi         # 2m × 2m,  Ψᵀ B_k Ψ
        X    = PtBP - sigma * G     # G M G = X
        # M = G^{-1} X G^{-1}
        A    = np.linalg.solve(G, X)
        M    = np.linalg.solve(G, A.T).T

        return sigma, Psi, M

    def rank2_factors(self) -> tuple[float, np.ndarray, np.ndarray]:
        """
        Return the explicit rank-2 decomposition B_k = σI + W D Wᵀ.

        Unlike ``factors()``, this does NOT go through the compact M matrix
        (which requires inverting G = ΨᵀΨ and can be numerically unstable when
        the stored pairs are nearly linearly dependent).  Instead, the columns
        of W are the explicit scaled rank-2 update vectors::

            W[:, i]   = Bs[i] / sqrt(|beta[i]|),  D[i]   = -1   (subtract step)
            W[:, m+i] = y[i]  / sqrt(sy[i]),       D[m+i] = +1   (add step)

        so that B_k = σI + W @ diag(D) @ Wᵀ exactly.

        Returns
        -------
        sigma : float
            B_0 = σI diagonal scale.
        W : ndarray, shape (n, 2*m)
            Column matrix.  Empty (n, 0) when m == 0.
        d : ndarray, shape (2*m,)
            Diagonal signs (+1 / -1).  Empty (0,) when m == 0.
        """
        m = len(self._s)
        sigma = 1.0 / self._scale

        if m == 0:
            return sigma, np.empty((self.n, 0)), np.empty(0)

        Bs, beta = self._precompute_Bs()
        W = np.empty((self.n, 2 * m))
        d = np.empty(2 * m)

        for i in range(m):
            W[:, i]     = Bs[:, i] / np.sqrt(abs(beta[i]))
            d[i]        = -1.0
            sy_i = float(self._y[i] @ self._s[i])
            W[:, m + i] = self._y[i] / np.sqrt(sy_i)
            d[m + i]    = +1.0

        return sigma, W, d

    # ------------------------------------------------------------------
    # Dense materialisation — O(n² · memory)
    # ------------------------------------------------------------------

    def materialise(self) -> np.ndarray:
        """
        Build and return the dense (n, n) Hessian approximation B_k.

        Applies the BFGS rank-2 update formula sequentially to B_0 = σI::

            B_{i+1} = B_i − (B_i sᵢ)(B_i sᵢ)ᵀ / (sᵢᵀ B_i sᵢ)
                           + yᵢ yᵢᵀ               / (yᵢᵀ sᵢ)

        using the precomputed Bs[i] = B_i sᵢ vectors.

        Cost: O(n² · memory).  The result is passed directly to OSQP as the
        QP Hessian P = B_k (no inversion needed).
        """
        m = len(self._s)
        sigma = 1.0 / self._scale
        B = sigma * np.eye(self.n)
        if m == 0:
            return B

        Bs, beta = self._precompute_Bs()

        for i in range(m):
            sy_i = float(self._y[i] @ self._s[i])
            B -= np.outer(Bs[:, i], Bs[:, i]) / beta[i]
            B += np.outer(self._y[i], self._y[i]) / sy_i

        return B

    def diag(self) -> np.ndarray:
        """
        Return ``diag(B_k)``.  Cost: O(n · memory) — uses the same
        ``_precompute_Bs`` factors as ``matvec`` / ``materialise``.

        Used as a Jacobi preconditioner for projected-CG.  Each stored
        rank-2 BFGS update contributes ``-(B_i s_i)[k]² / β_i + y_i[k]² / sy_i``
        to ``diag(B)[k]``.

        Always positive: starts at ``σ > 0`` and Powell damping keeps every
        rank-2 update PD-preserving.  Caller can still clip near zero for
        numerical safety.
        """
        n = self.n
        m = len(self._s)
        sigma = 1.0 / self._scale
        d = np.full(n, sigma)
        if m == 0:
            return d

        Bs, beta = self._precompute_Bs()
        for i in range(m):
            sy_i = float(self._y[i] @ self._s[i])
            d -= Bs[:, i] ** 2 / beta[i]
            d += self._y[i] ** 2 / sy_i
        return d
