"""Sparse augmented-KKT QP backend with L-BFGS Woodbury overlay.

Designed for problems where ``m_eq + m_ineq`` is comparable to ``n`` and the
constraint Jacobian is sparse — the regime where projected-CG (rank-deficient
``A Aᵀ``) and Woodbury-ADMM (dense ``O(m_constr · m_lbfgs)`` operations)
both stumble.  This is the IPOPT-with-MA57 problem class: PDE-constrained
NLPs, image-bilevel MPCCs (TV-MCP), structured QP transcriptions of OCPs.

Algorithm
---------
The L-BFGS Hessian decomposes as :math:`B = \\sigma I + W D W^T` (rank 2·memory)
via :meth:`pyfiltersqp._lbfgs.LBFGSMemory.rank2_factors`.  The augmented
KKT system for the equality-constrained inner solve is

.. math::

    \\begin{bmatrix} B + \\delta I & A_{\\text{act}}^T \\\\
                     A_{\\text{act}} & 0 \\end{bmatrix}
    \\begin{bmatrix} p \\\\ \\lambda \\end{bmatrix}
    = \\begin{bmatrix} -g \\\\ b \\end{bmatrix}

Direct factorisation of :math:`B` costs :math:`O(n^2)` for the :math:`W D W^T`
part — prohibitive at :math:`n \\gg 100`.  Instead, factor only the *sparse*
base

.. math::

    M_{\\text{base}} = \\begin{bmatrix} (\\sigma + \\delta) I & A_{\\text{act}}^T \\\\
                                       A_{\\text{act}} & 0 \\end{bmatrix}

via ``scipy.sparse.linalg.splu``, then apply Sherman-Morrison-Woodbury to
recover the full Hessian in :math:`O(\\text{memory})` extra triangular solves:

.. math::

    (M_{\\text{base}} + \\widehat W D \\widehat W^T) z = \\text{rhs}

    u = M_{\\text{base}}^{-1} \\text{rhs}

    U = M_{\\text{base}}^{-1} \\widehat W \\quad\\text{(one solve per column)}

    S = D^{-1} + \\widehat W^T U \\quad\\text{(2m × 2m dense)}

    z = u - U S^{-1} (\\widehat W^T u)

where :math:`\\widehat W = [W; 0]` embeds :math:`W` into the augmented space
(zeros in the constraint rows) and :math:`D^{-1} = \\text{diag}(d)` since
:math:`d \\in \\{-1, +1\\}`.

Cost
----
* One sparse LU per active-set iter: :math:`O(\\text{nnz\\_LU}(M_{\\text{base}}))`.
  For :math:`n \\sim m_{\\text{eq}}` and sparse :math:`J` this scales as
  :math:`O((n + m_{\\text{act}}) \\cdot \\text{avg\\_nnz\\_per\\_row} \\cdot \\text{fill})`.
* Per RHS: ``2m + 1`` sparse triangular solves plus a small ``(2m × 2m)``
  dense linear solve plus dense matrix-vector ops.

When to pick this backend
-------------------------
The auto-router selects ``kkt`` when ``m_eq`` is large *and* the Jacobian
is sparse — i.e. cases where PCG's inner Cholesky on ``A Aᵀ`` blows up
into a rank-deficient cycle.  See ``SQPSolver._select_auto_backend``.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import scipy.linalg
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ._lbfgs import LBFGSMemory


class KKTSparseQP:
    """
    Drop-in equivalent of :class:`ProjectedCGQP` and
    :class:`ImplicitLBFGSQP` for sparse, equality-heavy problems.

    Parameters
    ----------
    n, m_eq, m_ineq : int
        Problem dimensions.
    tol_active : float
        Initial-active-set threshold (constraint counted as active when
        ``c_ineq[j] > -tol_active`` at x0).
    tol_kkt : float
        Multiplier threshold for the active-set remove phase.
    max_as_iter : int or None
        Active-set iteration cap.  Default: ``2 n + m_eq + m_ineq + 20``.
    delta_reg : float
        Tikhonov shift on the (1,1) primal block of ``M_base`` (default
        1e-8).  Grows on splu numerical failure (capped at
        ``delta_reg_max``).
    delta_reg_max : float
        Upper bound on the Tikhonov shift.  ``solve()`` returns
        ``success=False`` if delta climbs above this threshold without
        producing a usable factorisation.
    dual_reg_ratio : float
        Ratio ``δ_d / δ_p`` for the (2,2) dual-block shift (default
        ``1e-4``).  The (2,2) shift biases the constraint residual by
        ``δ_d · |λ|``, so when |λ| is large (TV-MCP-class problems
        routinely see |λ| ~ 1e7 from over-committed active sets), a
        full ``δ_d = δ_p = 1e-8`` injects a residual of magnitude 1.0
        into the constraint equations — enough to forbid the SQP outer
        loop from ever driving the constraint violation below ~1.0.
        Setting ``δ_d = 1e-12`` (i.e. ratio 1e-4) drops the bias to
        ~1e-4 even at |λ| = 1e8, freeing the merit framework to make
        progress.  Both shifts grow in lock-step on splu failure.
    block_active_set : bool
        Bulk add/remove per AS iter (default ``True``).  Same semantics
        as :class:`ProjectedCGQP`.
    warm_start : bool
        Carry the previous solve()'s active set across SQP iterations
        (default ``True``).
    rank_filter : bool
        Detect linearly dependent active rows via pivoted QR on
        ``A_actᵀ`` and drop them from the KKT system (default
        ``True``).  Only runs when ``m_act > n`` (when redundancy is
        provable by counting); skipped in the common
        ``m_act ≤ n`` case where the regular splu + dual-reg path
        suffices.  Eliminates the over-commit pathology that
        regularisation hacks paper over: bias on λ vanishes, mu
        runaway disappears, line search regains progress on
        equality-saturated MPCC/PDE problems.  Cost: one dense
        ``scipy.linalg.qr(A.T, mode='r', pivoting=True)`` per
        rank-checking AS iter — O(n · m_act · min(n, m_act)).
    rank_tol : float
        Relative tolerance for rank determination (default 1e-10).
        Row pivots with ``|R[i,i]| ≤ rank_tol · |R[0,0]|`` are
        considered redundant.
    inertia_check : bool
        After each Woodbury-overlay solve, verify the resulting step
        ``p`` satisfies the descent-curvature condition
        ``pᵀ(B + δI)p > 0`` (default ``True``).  When violated, the
        L-BFGS Hessian is non-convex along ``p`` (typical when
        accumulated y-vectors include non-descent steps from a
        watchdog regime), so the QP step is *not* a descent direction
        regardless of merit choice.  The check bumps ``δ_p`` by 10×
        and re-solves until either the condition holds or
        ``δ_p > delta_reg_max``.  This is a directional surrogate
        for IPOPT-style inertia control (which needs symmetric LDLᵀ
        with Bunch-Kaufman pivoting; scipy.sparse provides only LU).
    inertia_curvature_tol : float
        Tolerance for the curvature check (default 1e-10).  Step
        ``p`` is accepted when ``pᵀBp ≥ -inertia_curvature_tol·‖p‖²``,
        i.e. ``B`` may be slightly indefinite along ``p`` but bounded
        below by the existing primal regularisation.
    """

    def __init__(
        self,
        n: int,
        m_eq: int,
        m_ineq: int,
        tol_active: float = 1e-8,
        tol_kkt: float = 1e-10,
        max_as_iter: int | None = None,
        delta_reg: float = 1e-8,
        delta_reg_max: float = 1.0,
        dual_reg_ratio: float = 1e-4,
        block_active_set: bool = True,
        warm_start: bool = True,
        rank_filter: bool = True,
        rank_tol: float = 1e-10,
        inertia_check: bool = True,
        inertia_curvature_tol: float = 1e-10,
    ):
        self.n = n
        self.m_eq = m_eq
        self.m_ineq = m_ineq
        self.tol_active = tol_active
        self.tol_kkt = tol_kkt
        self.max_as_iter = max_as_iter or (2 * n + m_eq + m_ineq + 20)
        self.delta_reg = delta_reg
        self.delta_reg_max = delta_reg_max
        self.dual_reg_ratio = dual_reg_ratio
        self.block_active_set = block_active_set
        self.warm_start = warm_start
        self.rank_filter = rank_filter
        self.rank_tol = rank_tol
        self.inertia_check = inertia_check
        self.inertia_curvature_tol = inertia_curvature_tol

        # SOC cache: factor + Woodbury overlay reused for the SOC RHS
        self._lbfgs:  LBFGSMemory | None = None
        self._g:      np.ndarray  | None = None
        self._J_eq                       = None
        self._J_ineq                     = None
        self._lb:     np.ndarray  | None = None
        self._ub:     np.ndarray  | None = None
        self._lu_soc                     = None
        self._U_soc:  np.ndarray  | None = None
        self._S_soc:  np.ndarray  | None = None
        self._W_soc:  np.ndarray  | None = None
        self._d_soc:  np.ndarray  | None = None
        self._sigma_soc: float           = 0.0
        self._delta_soc: float           = 0.0
        self._A_act_soc                  = None
        self._basis_idx_soc              = None
        self._active_ineq_soc: list[int] = []
        self._active_lb_soc:   list[int] = []
        self._active_ub_soc:   list[int] = []

        # Cross-call warm-start state (see ImplicitLBFGSQP for rationale).
        self._warm_init: bool = False
        self._warm_active_ineq: list[int] = []
        self._warm_active_lb:   list[int] = []
        self._warm_active_ub:   list[int] = []

    # ------------------------------------------------------------------
    # A_act assembly (sparse) — mirrors ProjectedCGQP._build_sparse
    # ------------------------------------------------------------------

    def _build(
        self,
        c_eq, J_eq, c_ineq, J_ineq, lb, ub,
        active_ineq, active_lb, active_ub,
    ):
        """Build A_act (m_act × n) and b_act (m_act,) as a sparse CSR.

        Row order: equalities, active inequalities, active lower bounds,
        active upper bounds — matches the multiplier-extraction order in
        :meth:`solve`.
        """
        n, m_eq = self.n, self.m_eq
        blocks: list = []
        b_parts: list = []

        if m_eq > 0:
            J_eq_csr = J_eq.tocsr() if sp.issparse(J_eq) else sp.csr_matrix(J_eq)
            blocks.append(J_eq_csr)
            b_parts.append(-np.asarray(c_eq, dtype=float))

        if active_ineq:
            J_ineq_csr = J_ineq.tocsr() if sp.issparse(J_ineq) else sp.csr_matrix(J_ineq)
            blocks.append(J_ineq_csr[active_ineq, :])
            b_parts.append(-np.asarray(c_ineq, dtype=float)[active_ineq])

        n_lb, n_ub = len(active_lb), len(active_ub)
        if n_lb + n_ub > 0:
            data = np.empty(n_lb + n_ub, dtype=float)
            rows = np.empty(n_lb + n_ub, dtype=np.int64)
            cols = np.empty(n_lb + n_ub, dtype=np.int64)
            if n_lb:
                data[:n_lb] = -1.0
                rows[:n_lb] = np.arange(n_lb)
                cols[:n_lb] = np.asarray(active_lb, dtype=np.int64)
            if n_ub:
                data[n_lb:] = 1.0
                rows[n_lb:] = n_lb + np.arange(n_ub)
                cols[n_lb:] = np.asarray(active_ub, dtype=np.int64)
            bnd = sp.csr_matrix((data, (rows, cols)), shape=(n_lb + n_ub, n))
            blocks.append(bnd)
            b_parts.append(
                np.concatenate([
                    -np.asarray(lb, dtype=float)[active_lb] if n_lb else np.empty(0),
                    np.asarray(ub,  dtype=float)[active_ub] if n_ub else np.empty(0),
                ])
            )

        if blocks:
            A_act = sp.vstack(blocks, format="csr")
            b_act = np.concatenate(b_parts)
            return A_act, b_act
        return sp.csr_matrix((0, n), dtype=float), np.empty(0, dtype=float)

    # ------------------------------------------------------------------
    # Rank-revealing row filter (item 6b.5)
    # ------------------------------------------------------------------

    @staticmethod
    def _find_basis_rows(A_act, tol_rel: float = 1e-10):
        """Identify a maximal linearly-independent subset of A_act rows
        via pivoted QR on ``A_actᵀ``.

        Returns ``basis_idx`` (sorted ndarray of row indices to keep)
        or ``None`` when ``A_act`` already has full row rank.

        Cost: dense pivoted QR on ``n × m_act`` — i.e.
        ``O(n · m_act · min(n, m_act))`` flops.  Caller should gate
        this on ``m_act > n`` so the QR fires only when redundancy
        is provable by counting.

        Tolerance: a row pivot is considered redundant when
        ``|R[i,i]| ≤ tol_rel · |R[0,0]|`` (relative to the largest
        pivot, not an absolute floor — this is invariant under uniform
        scaling of the constraint Jacobian).
        """
        m_act = A_act.shape[0]
        # Densify A_actᵀ.  scipy.linalg.qr requires a dense matrix;
        # for typical sparse Jacobians the dense conversion is the
        # dominant cost line below, but unavoidable without SuiteSparse.
        A_dense_T = (A_act.T.toarray() if sp.issparse(A_act)
                     else np.asarray(A_act).T)
        # mode='r' skips Q computation (~30 % saving); we only need R
        # diagonal for rank determination and P for the row indices.
        R, P = scipy.linalg.qr(A_dense_T, mode="r", pivoting=True)
        diag_R = np.abs(np.diag(R))
        if diag_R.size == 0 or diag_R[0] < 1e-14:
            return np.empty(0, dtype=np.int64)
        rank = int(np.sum(diag_R > tol_rel * diag_R[0]))
        if rank >= m_act:
            return None
        # Sort to preserve KKT row order (eq → ineq → lb → ub) so the
        # multiplier-extraction logic in ``solve()`` still works
        # without remapping.
        return np.sort(P[:rank])

    # ------------------------------------------------------------------
    # Sparse LU + Woodbury overlay
    # ------------------------------------------------------------------

    def _factor_kkt(self, sigma: float, A_act, force_dual_reg: bool = False,
                    delta_init: float | None = None):
        """Build and factor

            M_base = [ (σ+δ_p) I   A_actᵀ ]
                     [ A_act       −δ_d I ]

        Three-level regularisation strategy:

        1. **Try δ_d = 0 first**: the unbiased saddle-point.  Works for
           well-posed active sets where ``A_act`` has full row rank.
           The bias ``δ_p · λ`` from the (1,1) shift is negligible at
           δ_p ~ 1e-8 even when |λ| is large (e.g. HS15's λ ≈ 1750).
        2. **On splu failure, enable asymmetric dual regularisation**
           ``δ_d = self.dual_reg_ratio · δ_p`` (default ratio 1e-4 →
           δ_d = 1e-12 when δ_p = 1e-8).  Keeps the system non-singular
           when ``A_act`` has zero or near-zero rows but minimises the
           ``δ_d · λ`` bias on the constraint residual.

           **Why asymmetric matters**: with symmetric ``δ_d = δ_p =
           1e-8`` and TV-MCP-class problems where |λ| ~ 1e7 from
           over-committed active sets, the constraint-side bias is
           1e-8 · 1e7 = 0.1 — large enough to forbid the outer SQP
           from driving the constraint violation below ~0.1.  At ratio
           1e-4 the bias drops to 1e-3, freeing the merit framework
           to make progress.
        3. **On continued failure, grow both δ_p and δ_d in lock-step**
           by 10× per attempt, capped at ``self.delta_reg_max``.

        Pre-emptive dual reg fires when ``m_act > n`` because SuperLU's
        internal pivoting can hit invalid BLAS calls (cblas_dtrsv,
        cblas_dgemv) on rank-deficient saddle-point matrices — those
        *print to stderr but do not raise Python exceptions*, so the
        try/except below can't catch them and the lu factor is silently
        corrupted.  m_act > n is a proof of rank deficiency, so we skip
        straight to the asymmetric-regularised system.

        Crucially for the Woodbury overlay: the L-BFGS rank correction
        ``W D Wᵀ`` enters only the (1,1) block, so SMW math is identical
        regardless of whether δ_d > 0.

        Returns ``(lu, delta)`` or ``(None, delta)`` if factorisation
        fails even at ``self.delta_reg_max``.
        """
        n = self.n
        m_act = A_act.shape[0]
        delta = self.delta_reg if delta_init is None else delta_init
        # Pre-emptive dual regularisation when over-committed (see above).
        # Pre-emptive dual regularisation: directly when m_act > n (the
        # system is rank-deficient by counting), or when the caller flags
        # ``force_dual_reg`` (used when the *original* pre-rank-filter
        # active set was over-committed even though A_red is now square
        # — SuperLU's pivoting on perfectly-square saddle-points hits
        # silent BLAS errors for some structures).
        dual_reg = (m_act > n) or force_dual_reg

        for _ in range(20):  # at most 20 attempts; covers 1e-8 → 1e+4
            sigma_eff = sigma + delta
            delta_d   = delta * self.dual_reg_ratio   # asymmetric: ≪ δ_p
            top_left  = sigma_eff * sp.eye(n, format="csc")
            if m_act > 0:
                if dual_reg:
                    bottom_right = -delta_d * sp.eye(m_act, format="csc")
                else:
                    bottom_right = sp.csc_matrix((m_act, m_act))
                M = sp.bmat(
                    [[top_left, A_act.T],
                     [A_act,    bottom_right]],
                    format="csc",
                )
            else:
                M = top_left
            try:
                lu = spla.splu(M)
                # A successful splu can still produce a numerically
                # singular factor (zero pivot).  Catch via solve with a
                # unit RHS and inspect for inf/nan.
                test = lu.solve(np.ones(n + m_act))
                if not np.all(np.isfinite(test)):
                    raise RuntimeError("non-finite splu solve")
                return lu, delta
            except (RuntimeError, np.linalg.LinAlgError):
                if not dual_reg:
                    # First failure: turn on (2,2) regularisation, keep δ.
                    dual_reg = True
                    continue
                delta = max(delta * 10.0, 1e-12)
                if delta > self.delta_reg_max:
                    return None, delta
        return None, delta

    def _solve_with_woodbury(
        self,
        lu,
        sigma: float,
        delta: float,
        A_act,
        W: np.ndarray,
        d: np.ndarray,
        rhs_p: np.ndarray,
        rhs_lam: np.ndarray,
    ):
        """Apply (M_base + Ŵ D Ŵᵀ)⁻¹ rhs via Sherman-Morrison-Woodbury.

        Returns ``(p, lam, U, S)`` where ``U = M_base⁻¹ Ŵ`` and
        ``S = D⁻¹ + Ŵᵀ U`` are cached for solve_soc reuse.
        """
        n = self.n
        m_act = A_act.shape[0]
        rhs = np.concatenate([rhs_p, rhs_lam]) if m_act > 0 else rhs_p

        u = lu.solve(rhs)

        k2 = W.shape[1] if W is not None else 0
        if k2 == 0:
            p = u[:n]
            lam = u[n:] if m_act > 0 else np.empty(0)
            return p, lam, None, None

        # Embed W into the augmented space: Ŵ = [W; 0] of shape (n+m_act, 2m).
        # Solve M_base⁻¹ Ŵ column-by-column.
        U = np.empty((n + m_act, k2)) if m_act > 0 else np.empty((n, k2))
        for j in range(k2):
            if m_act > 0:
                rhs_w = np.concatenate([W[:, j], np.zeros(m_act)])
            else:
                rhs_w = W[:, j]
            U[:, j] = lu.solve(rhs_w)

        # Inner system: S = D⁻¹ + Ŵᵀ U.  D⁻¹ = diag(d) since d ∈ {-1, +1}.
        # ŴᵀU only sees the top n rows of U (last m_act rows of Ŵ are zero).
        S = np.diag(d) + W.T @ U[:n, :]

        # Wᵀu uses only the top n rows of u for the same reason.
        Wt_u = W.T @ u[:n]
        try:
            y = np.linalg.solve(S, Wt_u)
        except np.linalg.LinAlgError:
            y, *_ = np.linalg.lstsq(S, Wt_u, rcond=None)

        z = u - U @ y
        p = z[:n]
        lam = z[n:] if m_act > 0 else np.empty(0)
        return p, lam, U, S

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(
        self,
        lbfgs: LBFGSMemory,
        g: np.ndarray,
        c_eq: np.ndarray,
        J_eq,
        c_ineq: np.ndarray,
        J_ineq,
        x: np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
        """Solve the QP subproblem.  Same signature as
        :meth:`ProjectedCGQP.solve`."""
        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq
        tol = self.tol_active

        # Env-var debug hook (matches PCG's instrumentation)
        import os as _os, sys as _sys, time as _time
        _DBG = bool(_os.environ.get("PYFILTERSQP_DEBUG"))
        _t_qp = _time.perf_counter() if _DBG else 0.0
        _as_iter_count = [0]

        lb = xl - x
        ub = xu - x

        # SOC cache
        self._lbfgs  = lbfgs
        self._g      = g
        self._J_eq   = J_eq
        self._J_ineq = J_ineq
        self._lb     = lb
        self._ub     = ub

        # --- Initial active set (same logic as PCG) ---
        active_ineq: list[int] = []
        active_lb:   list[int] = []
        active_ub:   list[int] = []

        if self.warm_start and self._warm_init:
            active_ineq = [j for j in self._warm_active_ineq if 0 <= j < m_ineq]
            active_lb   = [k for k in self._warm_active_lb   if 0 <= k < n]
            active_ub   = [k for k in self._warm_active_ub   if 0 <= k < n]
            in_lb = set(active_lb)
            in_ub = set(active_ub)
            for k in range(n):
                if ub[k] - lb[k] < tol and k not in in_lb and k not in in_ub:
                    active_lb.append(k)
        else:
            active_ineq = [j for j in range(m_ineq) if c_ineq[j] > -tol]
            for k in range(n):
                if ub[k] - lb[k] < tol:
                    active_lb.append(k)

        if _DBG:
            print(
                f"[KKT] enter solve  m_eq={m_eq}  m_ineq={m_ineq}  "
                f"|active_ineq0|={len(active_ineq)}  |active_lb0|={len(active_lb)}  "
                f"|active_ub0|={len(active_ub)}  warm={self._warm_init}",
                flush=True, file=_sys.stderr,
            )

        lam_eq   = np.zeros(m_eq)
        lam_ineq = np.zeros(m_ineq)
        lam_lb   = np.zeros(n)
        lam_ub   = np.zeros(n)
        p        = np.zeros(n)

        alb_mask = np.zeros(n,      dtype=bool)
        aub_mask = np.zeros(n,      dtype=bool)
        aiq_mask = np.zeros(m_ineq, dtype=bool)
        for k in active_lb:   alb_mask[k] = True
        for k in active_ub:   aub_mask[k] = True
        for j in active_ineq: aiq_mask[j] = True

        visited: set = set()
        bland = False
        last_added   = None
        last_removed = None

        # Watchdogs (same as PCG)
        _BLAND_K          = 30
        _BLAND_AMPLITUDE  = 5
        _bland_window     = deque(maxlen=_BLAND_K)
        _OVERCAP_K        = 3
        _overcap_window   = deque(maxlen=_OVERCAP_K)
        # Fast-fire over-commit detector: once we're in bland mode AND
        # m_act > n, count consecutive iters.  At m_act > n the augmented
        # KKT is rank-deficient by counting alone, single-pivot adds
        # find new "violations" each iter from the regularised solve's
        # residuals, and SuperLU's internal panel routines can hit
        # invalid BLAS calls if we let m_act drift further.  K_FAST=10
        # is small enough to dodge the BLAS landmine but large enough
        # not to false-trip on transient overshoots.
        _BLAND_OVERCAP_FAST_K = 10
        _bland_overcap_count = 0

        # Cached factor + Woodbury overlay state (latest valid solve)
        last_lu     = None
        last_U      = None
        last_S      = None
        last_W      = None
        last_d      = None
        last_sigma  = 0.0
        last_delta  = 0.0
        last_A_act  = None
        last_basis_idx = None

        # QR cache for the rank filter.  Re-running pivoted QR every AS
        # iter costs O(n · m_act · min(n, m_act)) ≈ 400 ms on TV-MCP-
        # class problems — dominates the entire solve at ~15× slowdown.
        #
        # Caching is *correctness-sensitive*: a new row added by the
        # single-pivot phase is in the active set because the previous
        # iterate violated it; treating it as "redundant relative to the
        # cached basis" drops it from the KKT system, the new step
        # ignores the violation, and the line search rejects → SQP
        # collapses early.  We therefore refresh the QR after at most
        # ``_QR_REFRESH_K`` AS iters and on every remove.
        _qr_cache_basis    = None
        _qr_iters_since    = 0
        _QR_REFRESH_K      = 5

        for _ in range(self.max_as_iter):
            key = (frozenset(active_lb), frozenset(active_ub),
                   frozenset(active_ineq))
            if key in visited:
                bland = True
            else:
                visited.add(key)

            m_act_now = m_eq + len(active_ineq) + len(active_lb) + len(active_ub)
            _overcap_window.append(m_act_now > n)
            if (not bland
                    and len(_overcap_window) == _OVERCAP_K
                    and all(_overcap_window)):
                bland = True
                if _DBG:
                    print(
                        f"[KKT] FORCE-BLAND  m_act={m_act_now} > n={n} for "
                        f"{_OVERCAP_K} iters; falling to single-pivot",
                        flush=True, file=_sys.stderr,
                    )

            if bland:
                # Fast over-commit detector (10 consecutive iters >n).
                if m_act_now > n:
                    _bland_overcap_count += 1
                    if _bland_overcap_count >= _BLAND_OVERCAP_FAST_K:
                        if _DBG:
                            print(
                                f"[KKT] WATCHDOG  fast over-commit: "
                                f"m_act={m_act_now} > n={n} for "
                                f"{_BLAND_OVERCAP_FAST_K} consecutive bland "
                                f"iters; exit with best-effort p",
                                flush=True, file=_sys.stderr,
                            )
                        break
                else:
                    _bland_overcap_count = 0

                _bland_window.append(m_act_now)
                if len(_bland_window) == _BLAND_K:
                    win_min = min(_bland_window)
                    win_max = max(_bland_window)
                    amp = win_max - win_min
                    # Two exit conditions:
                    # 1. Small-amplitude oscillation: m_act flip-flops in a
                    #    narrow band (the classic Bland-degenerate cycle).
                    # 2. Persistent over-commit: m_act has stayed above n
                    #    for the entire window — single-pivot adds keep
                    #    finding "new" violations because the rank-
                    #    deficient KKT solve perturbs inactive constraints
                    #    into apparent violation each iter, and we'd grow
                    #    m_act forever otherwise.
                    if amp <= _BLAND_AMPLITUDE or win_min > n:
                        if _DBG:
                            reason = ("oscillation" if amp <= _BLAND_AMPLITUDE
                                      else "over-commit")
                            print(
                                f"[KKT] WATCHDOG  {reason}: m_act in "
                                f"[{win_min}, {win_max}] for {_BLAND_K} "
                                f"bland iters (n={n}); exit with best-effort p",
                                flush=True, file=_sys.stderr,
                            )
                        break
            elif _bland_window:
                _bland_window.clear()

            if _DBG:
                _t_as = _time.perf_counter()
            A_act, b_act = self._build(
                c_eq, J_eq, c_ineq, J_ineq, lb, ub,
                active_ineq, active_lb, active_ub,
            )
            if _DBG:
                _t_build = _time.perf_counter() - _t_as
                _t_qr0   = _time.perf_counter()

            # --- Rank-revealing filter (item 6b.5) ---
            # When ``m_act > n`` the active rows are linearly dependent
            # by counting alone.  Identify a maximal linearly-independent
            # subset via pivoted QR on ``A_actᵀ`` and drop the redundant
            # rows from the KKT system.  Multipliers for dropped rows
            # are zero in any unique KKT solution; the active-set
            # bookkeeping (active_ineq, active_lb, active_ub) is
            # unchanged so subsequent add/remove logic still sees them.
            #
            # Without this, the over-committed augmented KKT relies on
            # the asymmetric dual-reg shift (-δ_d I in the (2,2) block)
            # which biases λ by δ_d·|λ|; on TV-MCP-class problems |λ|
            # ~ 1e8 swamps the merit framework.  Rank filter eliminates
            # the bias entirely by solving a full-rank reduced system.
            basis_idx = None
            qr_ran    = False
            if self.rank_filter and A_act.shape[0] > n:
                # Refresh QR every ``_QR_REFRESH_K`` AS iters; reuse the
                # cached basis in between.  Bounded-staleness trade-off:
                # K iters of slightly biased basis selection in exchange
                # for K× fewer QR calls.
                if (_qr_cache_basis is not None
                        and _qr_iters_since < _QR_REFRESH_K):
                    basis_idx = _qr_cache_basis
                    _qr_iters_since += 1
                else:
                    basis_idx = self._find_basis_rows(A_act, self.rank_tol)
                    qr_ran = True
                    _qr_cache_basis = basis_idx
                    _qr_iters_since = 0
            if basis_idx is not None:
                # Indices may be stale after a single-pivot add (which
                # appends to active_*, growing m_act by 1).  The
                # *existing* indices still correctly point to original
                # rows, so subsetting A_act[basis_idx] stays valid even
                # though A_act now has more rows than the basis was
                # computed on.  New rows are simply treated as redundant
                # until the next QR refresh (correct under saturation;
                # bounded-stale under refresh-K).
                A_red = A_act[basis_idx, :]
                b_red = b_act[basis_idx]
            else:
                A_red, b_red = A_act, b_act

            if _DBG:
                _t_qr   = _time.perf_counter() - _t_qr0
                _t_kkt0 = _time.perf_counter()

            # --- Inner solve: factor M_base, apply Woodbury overlay ---
            sigma, W, d = lbfgs.rank2_factors()
            # Force dual reg when the rank filter produced a square
            # A_red but the *original* active set was over-committed.
            # SuperLU's saddle-point pivoting silently BLAS-errors on
            # some perfectly-square indefinite matrices; the dual shift
            # makes the matrix non-singular by construction.
            force_dr = (basis_idx is not None
                        and A_red.shape[0] >= n)

            # Inertia-aware factor + curvature-check loop.  If the
            # post-Woodbury step ``p`` has negative curvature against
            # the L-BFGS Hessian (``pᵀBp ≤ -tol·‖p‖²``), the model is
            # indefinite along ``p`` and the step is not a descent
            # direction regardless of merit choice.  Bump δ_p by 10×
            # and re-factor.  Cap retries; on cap, accept the step
            # anyway (best effort) to keep the AS loop alive.  This
            # is a directional surrogate for IPOPT-style symmetric-LDLᵀ
            # inertia control (scipy.sparse provides only LU).
            _inertia_delta_start = self.delta_reg
            _MAX_INERTIA_BUMPS   = 8
            for _inertia_attempt in range(_MAX_INERTIA_BUMPS + 1):
                lu, delta = self._factor_kkt(
                    sigma, A_red,
                    force_dual_reg=force_dr,
                    delta_init=_inertia_delta_start,
                )

                if lu is None:
                    if _DBG:
                        print(
                            f"[KKT] FACTOR-FAIL  delta={delta:.1e} > max; "
                            f"returning best-effort p so far",
                            flush=True, file=_sys.stderr,
                        )
                    return (p, lam_eq, lam_ineq, lam_lb, lam_ub, False)

                p, lam_red, U, S = self._solve_with_woodbury(
                    lu, sigma, delta, A_red, W, d, -g, b_red,
                )

                if (not self.inertia_check
                        or _inertia_attempt >= _MAX_INERTIA_BUMPS):
                    break

                p_norm2 = float(p @ p) if p.size > 0 else 0.0
                if p_norm2 < 1e-30:
                    break  # near-zero step; curvature undefined
                pBp = float(p @ lbfgs.matvec(p))
                # Descent-curvature condition: pᵀBp ≥ -tol·‖p‖²
                if pBp >= -self.inertia_curvature_tol * p_norm2:
                    break  # OK — convex (or near-convex) along p

                # Negative curvature detected: bump δ and refactor.
                _inertia_delta_start = max(_inertia_delta_start * 10.0, 1e-10)
                if _DBG:
                    print(
                        f"[KKT] INERTIA-BUMP  iter={_as_iter_count[0]+1}  "
                        f"pᵀBp/‖p‖² = {pBp/p_norm2:.2e}  "
                        f"δ_p → {_inertia_delta_start:.1e}",
                        flush=True, file=_sys.stderr,
                    )
                if _inertia_delta_start > self.delta_reg_max:
                    if _DBG:
                        print(
                            f"[KKT] INERTIA-CAP  δ_p exceeded delta_reg_max; "
                            f"accept best-effort step",
                            flush=True, file=_sys.stderr,
                        )
                    break

            # Map reduced multipliers back to the full active-set vector.
            # Dropped (redundant) rows get λ = 0.
            if basis_idx is not None:
                lam = np.zeros(A_act.shape[0])
                lam[basis_idx] = lam_red
            else:
                lam = lam_red

            # Cache for SOC reuse (cache the *reduced* system; basis_idx
            # not cached because SOC reuses the same active set anyway —
            # we'd need to re-run QR or re-solve the reduced system if
            # the SOC RHS changes.  Fall back to the full system in SOC
            # below for safety.)
            last_lu, last_U, last_S, last_W, last_d = lu, U, S, W, d
            last_sigma, last_delta, last_A_act = sigma, delta, A_red
            last_basis_idx = basis_idx

            if _DBG:
                _as_iter_count[0] += 1
                _t_kkt = _time.perf_counter() - _t_kkt0
                m_act_now = A_act.shape[0]
                rank = A_red.shape[0]
                qr_msg = (f"  qr={_t_qr*1000:5.1f}ms  rank={rank}/{m_act_now}"
                          if basis_idx is not None else "")
                print(
                    f"[KKT] AS_iter={_as_iter_count[0]:3d}  "
                    f"m_act={m_act_now:5d}  build={_t_build*1000:7.1f}ms  "
                    f"kkt={_t_kkt*1000:7.1f}ms  delta={delta:.1e}  bland={bland}{qr_msg}",
                    flush=True, file=_sys.stderr,
                )

            # Map lam entries back to typed arrays (same row order as _build)
            lam_eq[:]   = 0.0
            lam_ineq[:] = 0.0
            lam_lb[:]   = 0.0
            lam_ub[:]   = 0.0

            row_ptr = 0
            for i in range(m_eq):
                lam_eq[i] = lam[row_ptr]; row_ptr += 1
            for j in active_ineq:
                lam_ineq[j] = lam[row_ptr]; row_ptr += 1
            for k in active_lb:
                lam_lb[k] = lam[row_ptr]; row_ptr += 1
            for k in active_ub:
                lam_ub[k] = lam[row_ptr]; row_ptr += 1

            # ---- Block mode (default): bulk add / bulk remove per iter ----
            # Identical machinery to ProjectedCGQP — see _qp_pcg.py for the
            # full rationale on block updates and the visited-set / bland /
            # tabu cycle defences.
            if self.block_active_set and not bland:
                added_any = False

                if m_ineq > 0 and J_ineq is not None:
                    Jp = np.asarray(J_ineq @ p).ravel()
                    iq_viol_mask = (~aiq_mask) & (Jp + c_ineq > tol)
                    new_iq = np.where(iq_viol_mask)[0]
                    for j in new_iq:
                        active_ineq.append(int(j))
                        aiq_mask[j] = True
                    if new_iq.size > 0:
                        added_any = True

                lb_viol_mask = (~alb_mask) & (lb - p > tol)
                new_lb = np.where(lb_viol_mask)[0]
                for k in new_lb:
                    active_lb.append(int(k))
                    alb_mask[k] = True
                if new_lb.size > 0:
                    added_any = True

                ub_viol_mask = (~aub_mask) & (p - ub > tol)
                new_ub = np.where(ub_viol_mask)[0]
                for k in new_ub:
                    active_ub.append(int(k))
                    aub_mask[k] = True
                if new_ub.size > 0:
                    added_any = True

                if added_any:
                    last_added = None
                    last_removed = None
                    continue

                removed_any = False

                rm_iq = [j for j in active_ineq if lam_ineq[j] < -self.tol_kkt]
                if rm_iq:
                    aiq_set = set(rm_iq)
                    active_ineq = [j for j in active_ineq if j not in aiq_set]
                    for j in rm_iq:
                        aiq_mask[j] = False
                    removed_any = True

                rm_lb = [k for k in active_lb
                         if lam_lb[k] < -self.tol_kkt
                         and not (ub[k] - lb[k] < tol)]
                if rm_lb:
                    alb_set = set(rm_lb)
                    active_lb = [k for k in active_lb if k not in alb_set]
                    for k in rm_lb:
                        alb_mask[k] = False
                    removed_any = True

                rm_ub = [k for k in active_ub
                         if lam_ub[k] < -self.tol_kkt
                         and not (ub[k] - lb[k] < tol)]
                if rm_ub:
                    aub_set = set(rm_ub)
                    active_ub = [k for k in active_ub if k not in aub_set]
                    for k in rm_ub:
                        aub_mask[k] = False
                    removed_any = True

                if removed_any:
                    last_added = None
                    last_removed = None
                    # Remove invalidates the QR cache: dropped rows may
                    # have been in the basis, so the cached indices are
                    # no longer reliable.
                    _qr_cache_basis = None
                    continue

                break  # KKT satisfied

            # ---- Single-pivot Bland fallback ----
            added = False

            lb_viol = np.where(alb_mask, -np.inf, lb - p)
            if bland and last_removed is not None and last_removed[0] == "lb":
                lb_viol[last_removed[1]] = -np.inf
            if bland:
                wk = int(np.argmax(lb_viol > 0.0)) if (lb_viol > 0.0).any() else -1
            else:
                wk = int(np.argmax(lb_viol))
                if lb_viol[wk] <= 0.0: wk = -1
            if wk >= 0:
                active_lb.append(wk); alb_mask[wk] = True; added = True
                last_added = ("lb", wk)

            if not added:
                ub_viol = np.where(aub_mask, -np.inf, p - ub)
                if bland and last_removed is not None and last_removed[0] == "ub":
                    ub_viol[last_removed[1]] = -np.inf
                if bland:
                    wk = int(np.argmax(ub_viol > 0.0)) if (ub_viol > 0.0).any() else -1
                else:
                    wk = int(np.argmax(ub_viol))
                    if ub_viol[wk] <= 0.0: wk = -1
                if wk >= 0:
                    active_ub.append(wk); aub_mask[wk] = True; added = True
                    last_added = ("ub", wk)

            if not added and m_ineq > 0 and J_ineq is not None:
                Jp = np.asarray(J_ineq @ p).ravel()
                iq_viol = np.where(aiq_mask, -np.inf, Jp + c_ineq)
                if bland and last_removed is not None and last_removed[0] == "ineq":
                    iq_viol[last_removed[1]] = -np.inf
                if bland:
                    wj = int(np.argmax(iq_viol > 0.0)) if (iq_viol > 0.0).any() else -1
                else:
                    wj = int(np.argmax(iq_viol))
                    if iq_viol[wj] <= 0.0: wj = -1
                if wj >= 0:
                    active_ineq.append(wj); aiq_mask[wj] = True; added = True
                    last_added = ("ineq", wj)

            if added:
                last_removed = None
                continue

            removed = False

            tabu_ineq = (last_added[1]
                         if bland and last_added is not None and last_added[0] == "ineq"
                         else None)
            if bland:
                cand = [j for j in active_ineq
                        if lam_ineq[j] < -self.tol_kkt and j != tabu_ineq]
                wj = min(cand) if cand else -1
            else:
                wv, wj = -self.tol_kkt, -1
                for j in active_ineq:
                    if j == tabu_ineq:
                        continue
                    if lam_ineq[j] < wv:
                        wv, wj = lam_ineq[j], j
            if wj >= 0:
                active_ineq.remove(wj); aiq_mask[wj] = False; removed = True
                last_removed = ("ineq", wj)

            if not removed:
                tabu_lb = (last_added[1]
                           if bland and last_added is not None and last_added[0] == "lb"
                           else None)
                if bland:
                    cand = [k for k in active_lb
                            if lam_lb[k] < -self.tol_kkt and k != tabu_lb]
                    wk = min(cand) if cand else -1
                else:
                    wv, wk = -self.tol_kkt, -1
                    for k in active_lb:
                        if k == tabu_lb:
                            continue
                        if lam_lb[k] < wv:
                            wv, wk = lam_lb[k], k
                if wk >= 0:
                    active_lb.remove(wk); alb_mask[wk] = False; removed = True
                    last_removed = ("lb", wk)

            if not removed:
                tabu_ub = (last_added[1]
                           if bland and last_added is not None and last_added[0] == "ub"
                           else None)
                if bland:
                    cand = [k for k in active_ub
                            if lam_ub[k] < -self.tol_kkt and k != tabu_ub]
                    wk = min(cand) if cand else -1
                else:
                    wv, wk = -self.tol_kkt, -1
                    for k in active_ub:
                        if k == tabu_ub:
                            continue
                        if lam_ub[k] < wv:
                            wv, wk = lam_ub[k], k
                if wk >= 0:
                    active_ub.remove(wk); aub_mask[wk] = False; removed = True
                    last_removed = ("ub", wk)

            if removed:
                last_added = None
                # Remove invalidates the QR cache (see block-remove note).
                _qr_cache_basis = None
            else:
                break  # KKT satisfied

        if _DBG:
            print(
                f"[KKT] EXIT solve  AS_iters={_as_iter_count[0]}  "
                f"total={(_time.perf_counter()-_t_qp)*1000:7.1f}ms",
                flush=True, file=_sys.stderr,
            )

        p = np.clip(p, lb, ub)

        # Cache final state for SOC reuse
        self._lu_soc    = last_lu
        self._U_soc     = last_U
        self._S_soc     = last_S
        self._W_soc     = last_W
        self._d_soc     = last_d
        self._sigma_soc = last_sigma
        self._delta_soc = last_delta
        self._A_act_soc = last_A_act
        self._basis_idx_soc = last_basis_idx
        self._active_ineq_soc = list(active_ineq)
        self._active_lb_soc   = list(active_lb)
        self._active_ub_soc   = list(active_ub)

        if self.warm_start:
            # Only save warm state when m_act is *not* over-committed.
            # Inheriting an over-committed active set into the next QP
            # poisons it: the new solve starts at m_act > n, FORCE-BLAND
            # fires, the fast over-commit watchdog needs ≥10 bland iters
            # to trigger, and SuperLU's internal BLAS panel can fail
            # before the watchdog gets there.  When the current solve
            # ended over-committed (likely the watchdog itself fired),
            # clear warm state so the next call cold-starts.
            m_act_final = (m_eq + len(active_ineq)
                           + len(active_lb) + len(active_ub))
            if m_act_final <= n:
                self._warm_active_ineq = list(active_ineq)
                self._warm_active_lb   = list(active_lb)
                self._warm_active_ub   = list(active_ub)
                self._warm_init = True
            else:
                if _DBG:
                    print(
                        f"[KKT] warm save SKIPPED: m_act_final={m_act_final} > "
                        f"n={n}; next solve cold-starts",
                        flush=True, file=_sys.stderr,
                    )
                self._warm_init = False

        return p, lam_eq, lam_ineq, lam_lb, lam_ub, True

    def solve_soc(
        self,
        c_eq_trial:   np.ndarray,
        c_ineq_trial: np.ndarray,
        x:  np.ndarray,
        p:  np.ndarray,
        xl: np.ndarray,
        xu: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """Second-order correction step.  Reuses the cached LU + Woodbury
        overlay from the most recent ``solve()`` call; only the RHS
        changes.  Same interface as :meth:`ProjectedCGQP.solve_soc`."""
        if self._lu_soc is None or self._A_act_soc is None:
            return np.zeros(self.n), False

        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq
        active_ineq = self._active_ineq_soc
        active_lb   = self._active_lb_soc
        active_ub   = self._active_ub_soc

        lb_soc = xl - (x + p)
        ub_soc = xu - (x + p)

        # Rebuild b_act with trial-point constraint values; A_act is
        # unchanged (active set + Jacobians are fixed for SOC).
        b_parts: list = []
        if m_eq > 0:
            b_parts.append(-np.asarray(c_eq_trial, dtype=float))
        if active_ineq:
            b_parts.append(-np.asarray(c_ineq_trial, dtype=float)[active_ineq])
        n_lb, n_ub = len(active_lb), len(active_ub)
        if n_lb + n_ub > 0:
            b_parts.append(
                np.concatenate([
                    -np.asarray(lb_soc, dtype=float)[active_lb] if n_lb else np.empty(0),
                    np.asarray(ub_soc,  dtype=float)[active_ub] if n_ub else np.empty(0),
                ])
            )
        b_act = np.concatenate(b_parts) if b_parts else np.empty(0)

        # If the cached factor was built on a rank-filtered (basis-only)
        # A_act, project the trial-point b onto the same basis rows.
        # Cached ``_A_act_soc`` is already the reduced matrix.
        if self._basis_idx_soc is not None:
            b_act = b_act[self._basis_idx_soc]

        try:
            d_step, _, _, _ = self._solve_with_woodbury(
                self._lu_soc,
                self._sigma_soc,
                self._delta_soc,
                self._A_act_soc,
                self._W_soc,
                self._d_soc,
                -self._g,
                b_act,
            )
        except Exception:
            return np.zeros(n), False

        d_step = np.clip(d_step, lb_soc, ub_soc)
        return d_step, True
