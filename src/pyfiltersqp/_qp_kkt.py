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
        Tikhonov shift on the (1,1) block of ``M_base`` (default 1e-8).
        Grows on splu numerical failure (capped at ``delta_reg_max``).
    delta_reg_max : float
        Upper bound on the Tikhonov shift.  ``solve()`` returns
        ``success=False`` if delta climbs above this threshold without
        producing a usable factorisation.
    block_active_set : bool
        Bulk add/remove per AS iter (default ``True``).  Same semantics
        as :class:`ProjectedCGQP`.
    warm_start : bool
        Carry the previous solve()'s active set across SQP iterations
        (default ``True``).
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
        block_active_set: bool = True,
        warm_start: bool = True,
    ):
        self.n = n
        self.m_eq = m_eq
        self.m_ineq = m_ineq
        self.tol_active = tol_active
        self.tol_kkt = tol_kkt
        self.max_as_iter = max_as_iter or (2 * n + m_eq + m_ineq + 20)
        self.delta_reg = delta_reg
        self.delta_reg_max = delta_reg_max
        self.block_active_set = block_active_set
        self.warm_start = warm_start

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
    # Sparse LU + Woodbury overlay
    # ------------------------------------------------------------------

    def _factor_kkt(self, sigma: float, A_act):
        """Build and factor

            M_base = [ (σ+δ_p) I   A_actᵀ ]
                     [ A_act       −δ_d I ]

        Two-level regularisation strategy:

        1. **Try δ_d = 0 first**: the unbiased saddle-point.  Works for
           well-posed active sets where ``A_act`` has full row rank.
           The bias ``δ_p · λ`` from the (1,1) shift is negligible at
           δ_p ~ 1e-8 even when |λ| is large (e.g. HS15's λ ≈ 1750).
        2. **On splu failure, enable dual regularisation** ``δ_d = δ_p``:
           IPOPT-style primal-dual shift.  Keeps the system non-singular
           when ``A_act`` has zero or near-zero rows (common in MPCC /
           Scholtes formulations where some constraint Jacobians vanish
           at certain x).  Adds a ``δ_d · λ`` bias to the constraint
           residual; harmless for SOC / line search use.
        3. **On continued failure, grow δ_p (and δ_d in lock-step)** by
           10× per attempt, capped at ``self.delta_reg_max``.

        Crucially for the Woodbury overlay: the L-BFGS rank correction
        ``W D Wᵀ`` enters only the (1,1) block, so SMW math is identical
        regardless of whether δ_d > 0.

        Returns ``(lu, delta)`` or ``(None, delta)`` if factorisation
        fails even at ``self.delta_reg_max``.
        """
        n = self.n
        m_act = A_act.shape[0]
        delta = self.delta_reg
        dual_reg = False     # try unregularised (2,2)=0 first

        for _ in range(20):  # at most 20 attempts; covers 1e-8 → 1e+4
            sigma_eff = sigma + delta
            top_left  = sigma_eff * sp.eye(n, format="csc")
            if m_act > 0:
                if dual_reg:
                    bottom_right = -delta * sp.eye(m_act, format="csc")
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

        # Cached factor + Woodbury overlay state (latest valid solve)
        last_lu     = None
        last_U      = None
        last_S      = None
        last_W      = None
        last_d      = None
        last_sigma  = 0.0
        last_delta  = 0.0
        last_A_act  = None

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
                _bland_window.append(m_act_now)
                if (len(_bland_window) == _BLAND_K
                        and (max(_bland_window) - min(_bland_window))
                        <= _BLAND_AMPLITUDE):
                    if _DBG:
                        print(
                            f"[KKT] WATCHDOG  m_act in [{min(_bland_window)}, "
                            f"{max(_bland_window)}] for {_BLAND_K} bland iters; "
                            f"exit AS loop with best-effort p",
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
                _t_kkt0  = _time.perf_counter()

            # --- Inner solve: factor M_base, apply Woodbury overlay ---
            sigma, W, d = lbfgs.rank2_factors()
            lu, delta = self._factor_kkt(sigma, A_act)

            if lu is None:
                # Numerically singular at maximum delta.  Bail.
                if _DBG:
                    print(
                        f"[KKT] FACTOR-FAIL  delta={delta:.1e} > max; "
                        f"returning best-effort p so far",
                        flush=True, file=_sys.stderr,
                    )
                return (p, lam_eq, lam_ineq, lam_lb, lam_ub, False)

            p, lam, U, S = self._solve_with_woodbury(
                lu, sigma, delta, A_act, W, d, -g, b_act,
            )

            # Cache for SOC reuse
            last_lu, last_U, last_S, last_W, last_d = lu, U, S, W, d
            last_sigma, last_delta, last_A_act = sigma, delta, A_act

            if _DBG:
                _as_iter_count[0] += 1
                _t_kkt = _time.perf_counter() - _t_kkt0
                print(
                    f"[KKT] AS_iter={_as_iter_count[0]:3d}  "
                    f"m_act={A_act.shape[0]:5d}  build={_t_build*1000:7.1f}ms  "
                    f"kkt={_t_kkt*1000:7.1f}ms  delta={delta:.1e}  bland={bland}",
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
        self._active_ineq_soc = list(active_ineq)
        self._active_lb_soc   = list(active_lb)
        self._active_ub_soc   = list(active_ub)

        if self.warm_start:
            self._warm_active_ineq = list(active_ineq)
            self._warm_active_lb   = list(active_lb)
            self._warm_active_ub   = list(active_ub)
            self._warm_init = True

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
