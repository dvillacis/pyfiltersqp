"""Projected-CG QP subproblem solver — for very large n where direct
Schur factorisation of the active-set KKT matrix becomes too expensive.

Algorithm
---------
Solves the SQP QP subproblem::

    min   (1/2) pᵀ B_k p + gᵀ p
    s.t.  J_eq p = -c_eq             (m_eq equalities)
          J_ineq p ≤ -c_ineq         (m_ineq inequalities)
          xl - x ≤ p ≤ xu - x        (step bounds)

via a primal-dual *active-set* loop (mirrors :class:`ImplicitLBFGSQP`)
whose inner equality-constrained QP is solved by **projected conjugate
gradient** (Nocedal & Wright Algorithm 16.2) instead of a direct Schur
complement.

Why
---
At each active-set iteration ImplicitLBFGSQP forms the Schur matrix
``S = A_act H A_actᵀ`` (cost: ``m_act × n × m`` matvecs) and solves an
``m_act × m_act`` dense system (``np.linalg.lstsq``: O(m_act³)).  When
the active set is large (active-bound-heavy problems at n > 10 000),
the cubic factorisation dominates.

PCG solves the inner KKT iteratively in O(cg_iters × (n·m + m_act·n))
operations and only needs ``B·v`` (the ``LBFGSMemory.matvec`` callable
we already have).  No dense ``B_k`` is ever formed; no n×n or m_act×m_act
factorisation is ever computed.

Per-CG-iteration cost breakdown (for typical m_act ≪ n):

* ``matvec(d)`` for B_k @ d              : O(n · memory)        ← dominant
* one project: solve (A Aᵀ) v = A r       : O(m_act² + m_act · n)
* axpy + dot products                    : O(n + m_act)

The (A Aᵀ) factorisation is computed *once* per active-set iteration
and re-used across all CG iters of that AS step.

Active set
----------
Same primal-dual active-set machinery as :class:`ImplicitLBFGSQP`:
add-most-violated, remove-most-negative-multiplier, with cycle detection
(visited active-set hashing → switch to Bland's rule), one-step tabu
memory, and the row-encoding convention that makes all non-equality
multipliers ≥ 0.  See _implicit_qp.py for the rationale of these
defences; we ship the same set here.

When to pick this backend
-------------------------
The auto-router does *not* yet pick PCG (opt-in via ``backend="pcg"``).
Empirically:

* PCG is better than ImplicitLBFGSQP when m_act is large (≳ 100) AND
  the L-BFGS Hessian is well-conditioned enough that PCG converges in
  far fewer iters than m_act.
* PCG is *worse* on small/HS-sized problems (m_act < 10): the active-set
  iteration count dominates and Schur's direct solve is essentially free.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._lbfgs import LBFGSMemory


class ProjectedCGQP:
    """
    Active-set + projected-CG QP solver, drop-in compatible with
    :class:`ImplicitLBFGSQP` and :class:`WoodburyADMM`.

    Parameters
    ----------
    n, m_eq, m_ineq : int
        Problem dimensions.
    tol_active : float
        Initial-active-set threshold (constraint counted as active when
        ``c_ineq[j] > -tol_active`` at x0).
    tol_kkt : float
        Multiplier threshold for the remove phase.
    max_as_iter : int or None
        Maximum active-set iterations.  Default: ``2 n + m_eq + m_ineq + 20``.
    cg_tol : float
        Relative residual tolerance for the inner PCG solve (default 1e-4).
        Loose by design — this is *inexact-Newton* style, and SQP's outer
        loop tolerates a partial inner solve.  Tightening to 1e-6 doubles
        the inner iter count without measurably improving outer convergence.
    cg_max_iter : int or None
        Inner PCG iteration cap.  Default ``min(2 (m_eq + 50), 500)``.
        For large unconstrained or sparsely-constrained problems, n CG
        iters is overkill: an L-BFGS Hessian carries only O(memory) useful
        information, so CG converges (or doesn't) in a small multiple of
        memory once the active set is right.  The cap keeps a single SQP
        step bounded even when CG would otherwise wander.
    block_active_set : bool
        Bulk add/remove per AS iter (default ``True``).  See
        :class:`ImplicitLBFGSQP` for the full rationale; the same
        block-mode + Bland fallback applies here.  Especially valuable
        for PCG since the per-AS-iter Cholesky on ``A_act A_actᵀ``
        scales as O(m_act³) — fewer AS iters cuts that cubic cost.
    warm_start : bool
        Carry the previous solve()'s active set across SQP iterations
        (default ``True``).
    """

    def __init__(
        self,
        n: int,
        m_eq: int,
        m_ineq: int,
        tol_active:  float = 1e-8,
        tol_kkt:     float = 1e-10,
        max_as_iter: int | None = None,
        cg_tol:      float = 1e-4,
        cg_max_iter: int | None = None,
        block_active_set: bool = True,
        warm_start: bool = True,
    ):
        self.n          = n
        self.m_eq       = m_eq
        self.m_ineq     = m_ineq
        self.tol_active = tol_active
        self.tol_kkt    = tol_kkt
        self.max_as_iter = max_as_iter or (2 * n + m_eq + m_ineq + 20)
        self.cg_tol      = cg_tol
        self.cg_max_iter = cg_max_iter or min(2 * (m_eq + 50), 500)
        self.block_active_set = block_active_set
        self.warm_start = warm_start

        # State cached between solve() and solve_soc()
        self._lbfgs:  LBFGSMemory | None = None
        self._g:      np.ndarray  | None = None
        self._J_eq                       = None
        self._J_ineq                     = None
        self._lb:     np.ndarray  | None = None
        self._ub:     np.ndarray  | None = None
        self._active_ineq_soc: list[int] = []
        self._active_lb_soc:   list[int] = []
        self._active_ub_soc:   list[int] = []

        # Cross-call warm-start state (see ImplicitLBFGSQP for rationale).
        self._warm_init: bool = False
        self._warm_active_ineq: list[int] = []
        self._warm_active_lb:   list[int] = []
        self._warm_active_ub:   list[int] = []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row(J, i: int, n: int) -> np.ndarray:
        if sp.issparse(J):
            return np.asarray(J[i, :].todense()).ravel()
        return np.asarray(J[i, :]).ravel()

    def _build(
        self,
        c_eq, J_eq, c_ineq, J_ineq, lb, ub,
        active_ineq, active_lb, active_ub,
    ):
        """Build A_act (m_act × n) and b_act (m_act,) for the current active set.
        Same row encoding as ImplicitLBFGSQP.  Returns a sparse CSR matrix
        when either Jacobian is sparse, dense ndarray otherwise."""
        if sp.issparse(J_eq) or sp.issparse(J_ineq):
            return self._build_sparse(
                c_eq, J_eq, c_ineq, J_ineq, lb, ub,
                active_ineq, active_lb, active_ub,
            )

        n, m_eq = self.n, self.m_eq
        rows: list[np.ndarray] = []
        b:    list[float]      = []

        for i in range(m_eq):
            rows.append(self._row(J_eq, i, n))
            b.append(-c_eq[i])
        for j in active_ineq:
            rows.append(self._row(J_ineq, j, n))
            b.append(-c_ineq[j])
        e = np.zeros(n)
        for k in active_lb:
            e[k] = -1.0
            rows.append(e.copy())
            b.append(float(-lb[k]))
            e[k] = 0.0
        for k in active_ub:
            e[k] = 1.0
            rows.append(e.copy())
            b.append(float(ub[k]))
            e[k] = 0.0

        if rows:
            return np.array(rows, dtype=float), np.array(b, dtype=float)
        return np.empty((0, n), dtype=float), np.empty(0, dtype=float)

    def _build_sparse(
        self,
        c_eq, J_eq, c_ineq, J_ineq, lb, ub,
        active_ineq, active_lb, active_ub,
    ):
        """Sparse fast path — see :meth:`ImplicitLBFGSQP._build_sparse`."""
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
    # Inner: projected CG for equality-constrained QP
    # ------------------------------------------------------------------

    def _pcg(self, lbfgs, g, A, b):
        """
        Solve  min 0.5 pᵀ B p + gᵀ p   s.t.  A p = b   via projected CG.

        Implementation follows Nocedal & Wright (2006) Algorithm 16.2.
        With m_act = 0 (unconstrained) this degenerates to standard CG.

        Returns
        -------
        p   : ndarray (n,)
        lam : ndarray (m_act,)
            Multipliers for the equality system, recovered from the final
            KKT residual via  A Aᵀ λ = -A (g + B p).
        """
        n     = self.n
        m_act = A.shape[0]
        cg_max_iter = self.cg_max_iter
        cg_tol      = self.cg_tol

        sparse_A = sp.issparse(A)

        # ---- initial primal iterate satisfying A p = b ----
        if m_act > 0:
            # AAT must be dense for the np.linalg.cholesky / solve path.
            # Sparse @ sparse.T returns sparse; densify for the m_act ×
            # m_act factorisation.  When m_act ≪ n this dense block is
            # tiny relative to the sparse Jacobian itself.
            AAT = A @ A.T
            if sp.issparse(AAT):
                AAT = AAT.toarray()
            else:
                AAT = np.asarray(AAT)
            try:
                AAT_inv_b = np.linalg.solve(AAT, b)
            except np.linalg.LinAlgError:
                AAT_inv_b, *_ = np.linalg.lstsq(AAT, b, rcond=None)
            p = np.asarray(A.T @ AAT_inv_b).ravel()
        else:
            AAT = None
            p = np.zeros(n)

        # Convenience: project onto null(A) using the cached factor.
        # When AAT is rank-deficient (m_act > n, or numerically singular for
        # redundant constraint rows), Cholesky fails.  Falling back to lstsq
        # per project()-call is catastrophic — lstsq is O(m_act³) and
        # project() runs once per CG inner iter, so a single AS iter on a
        # rank-deficient system can burn O(cg_iters · m_act³) flops.  Instead,
        # add a tiny Tikhonov shift to AAT so Cholesky succeeds, and keep
        # every project() at O(m_act²) triangular-solve cost.  The shift
        # corresponds to solving the regularised system
        #   (AAT + εI) λ = A v
        # in place of the pseudoinverse — for the projection use here this
        # is equivalent to projecting onto a slightly-shrunk null(A) and
        # is harmless since the outer SQP only needs descent, not exactness.
        if m_act > 0:
            try:
                AAT_lu = np.linalg.cholesky(AAT)
            except np.linalg.LinAlgError:
                # Diagonal shift scaled to AAT magnitude; 1e-10 relative is
                # enough to clear redundancy from active-set duplicates while
                # being numerically negligible relative to the data.
                trace = float(np.trace(AAT))
                eps = max(1e-12, 1e-10 * trace / max(m_act, 1))
                AAT = AAT + eps * np.eye(m_act)
                AAT_lu = np.linalg.cholesky(AAT)

            def _solve_AAT(rhs, _L=AAT_lu):
                y = np.linalg.solve(_L, rhs)
                return np.linalg.solve(_L.T, y)

            if sparse_A:
                def project(v):
                    Av = np.asarray(A @ v).ravel()
                    return v - np.asarray(A.T @ _solve_AAT(Av)).ravel()
            else:
                def project(v):
                    return v - A.T @ _solve_AAT(A @ v)
        else:
            _solve_AAT = None
            def project(v):
                return v

        # ---- CG main loop ----
        # Note: we tried a Jacobi preconditioner ``M = diag(B_k)`` (computed
        # via :meth:`LBFGSMemory.diag` in O(n·m)) and found it consistently
        # *slower* than no preconditioner on every problem class we
        # benchmarked.  The reason: L-BFGS ``B = σI + Σ rank-2 updates``
        # already has its dominant component (σI) perfectly conditioned;
        # the rank-2 contributions to ``diag(B)`` are localised and dividing
        # by them redistributes more than it conditions, increasing CG
        # iter count.  The diag method is kept as a public utility because
        # it's useful in other contexts (e.g. v0.6a Woodbury-KKT
        # preconditioning) but PCG runs unpreconditioned by design.
        r       = lbfgs.matvec(p) + g                # full gradient at p
        g_proj  = project(r)
        d       = -g_proj
        rs_old  = float(g_proj @ g_proj)
        rs_init = max(rs_old, 1e-30)

        for k in range(cg_max_iter):
            if rs_old <= cg_tol * cg_tol * rs_init:
                break

            Bd  = lbfgs.matvec(d)
            dBd = float(d @ Bd)
            if dBd <= 1e-14:
                break

            alpha = rs_old / dBd
            p     = p + alpha * d
            r     = r + alpha * Bd
            g_proj_new = project(r)
            rs_new = float(g_proj_new @ g_proj_new)
            beta   = rs_new / rs_old
            d      = -g_proj_new + beta * d
            g_proj = g_proj_new
            rs_old = rs_new

        # ---- recover multipliers from KKT residual ----
        # ∇L = B p + g + Aᵀ λ = 0  ⇒  Aᵀ λ = -(B p + g)
        # Least-norm: λ = -(A Aᵀ)⁻¹ A (B p + g)
        if m_act > 0:
            final_grad = lbfgs.matvec(p) + g
            lam = -_solve_AAT(np.asarray(A @ final_grad).ravel())
        else:
            lam = np.empty(0)

        return p, lam

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
        """Drop-in equivalent of :meth:`ImplicitLBFGSQP.solve`.  See module
        docstring for the algorithm; the active-set machinery is identical,
        only the inner solve differs."""
        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq
        tol = self.tol_active

        # Env-var-triggered timing.  Reports per-AS-iter cost so we can tell
        # whether a slow QP is one big AS iter or many small ones.
        import os as _os, sys as _sys, time as _time
        _DBG = bool(_os.environ.get("PYFILTERSQP_DEBUG"))
        _t_qp = _time.perf_counter() if _DBG else 0.0
        _as_iter_count = [0]

        lb = xl - x
        ub = xu - x

        # Cache for solve_soc
        self._lbfgs  = lbfgs
        self._g      = g
        self._J_eq   = J_eq
        self._J_ineq = J_ineq
        self._lb     = lb
        self._ub     = ub

        # Initial active set (mirrors ImplicitLBFGSQP).  Warm-start from
        # the previous solve()'s final active set when available; cold-
        # start from currently-violated inequalities otherwise.
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
                if ub[k] - lb[k] < tol:        # truly fixed variable
                    active_lb.append(k)

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

        if _DBG:
            print(
                f"[PCG] enter solve  m_eq={m_eq}  m_ineq={m_ineq}  "
                f"|active_ineq0|={len(active_ineq)}  |active_lb0|={len(active_lb)}  "
                f"|active_ub0|={len(active_ub)}  warm={self._warm_init}",
                flush=True, file=_sys.stderr,
            )

        visited: set = set()
        bland = False
        last_added   = None
        last_removed = None

        # Cycle watchdog (degenerate-active-set safety net).
        #
        # The visited-set above flips ``bland=True`` when an active-set state
        # repeats — which proves Bland's-rule convergence under the standard
        # non-degenerate hypothesis.  When ``A_act`` is rank-deficient
        # (m_act > n minus equality redundancy), multipliers aren't unique
        # and the "remove most-negative-multiplier" pivot can flip-flop
        # between equivalent KKT solutions even under Bland's rule.  In the
        # field this manifests as m_act bouncing inside a small set
        # (e.g. {599, 600}) for thousands of AS iters.
        #
        # The watchdog tracks |active set| over the K most recent iters
        # spent in bland mode; if those values collapse to ≤ 2 distinct
        # numbers we declare the cycle and exit with the current iterate
        # as a best-effort step.  The outer SQP's line search will reject
        # it if it's not a descent direction.
        #
        # K is large enough that genuine Bland convergence on a 100-active-
        # set problem (~10²–10³ iters) doesn't trip the watchdog.
        from collections import deque as _deque
        # Watchdog: if m_act stays within ``_BLAND_AMPLITUDE`` of itself
        # over ``_BLAND_K`` consecutive bland-mode iters, the active set
        # is oscillating between equivalent rank-deficient KKT solutions
        # and Bland's rule alone won't break out.  Amplitude criterion
        # (vs distinct-set) handles cycles of any period — observed:
        # period-2 {599,600}, period-3 {544,545,546}, etc.
        _BLAND_K          = 30
        _BLAND_AMPLITUDE  = 5
        _bland_window     = _deque(maxlen=_BLAND_K)
        # Persistence buffer for the m_act > n force-bland trigger.  A
        # transient block-add spike past n is fine — block-remove will
        # drop the surplus next iter.  We only force bland when m_act
        # exceeds n for K_OVER consecutive iters, indicating the active
        # set has genuinely overcommitted.
        _OVERCAP_K          = 3
        _overcap_window     = _deque(maxlen=_OVERCAP_K)
        # Fast-fire over-commit detector under bland mode (see KKTSparseQP
        # for the full rationale; same logic mirrored here).
        _BLAND_OVERCAP_FAST_K = 10
        _bland_overcap_count = 0

        for _ in range(self.max_as_iter):
            key = (frozenset(active_lb), frozenset(active_ub),
                   frozenset(active_ineq))
            if key in visited:
                bland = True
            else:
                visited.add(key)

            # Force single-pivot mode once m_act exceeds n.  Block-add can
            # over-commit by tens or hundreds of rows in one iter, which
            # makes A_act guaranteed-rank-deficient (≥ 1 redundancy per
            # row over n).  In that regime block-add and block-remove
            # toggle large groups of constraints forever — see the TV-MCP
            # trace where m_act ping-pongs between ~430 and ~600 in block
            # mode without the visited-set ever repeating, so bland never
            # flips on its own.  Single-pivot caps growth at +1 per iter
            # and lets the bland-window watchdog (above) trigger cleanly.
            m_act_now = m_eq + len(active_ineq) + len(active_lb) + len(active_ub)
            _overcap_window.append(m_act_now > n)
            if (not bland
                    and len(_overcap_window) == _OVERCAP_K
                    and all(_overcap_window)):
                bland = True
                if _DBG:
                    print(
                        f"[PCG] FORCE-BLAND  m_act={m_act_now} > n={n} for "
                        f"{_OVERCAP_K} iters; falling to single-pivot",
                        flush=True, file=_sys.stderr,
                    )

            # Watchdog: count consecutive bland iters where m_act sits in
            # a small set.  Reset the window whenever bland flips back off.
            if bland:
                # Fast over-commit detector (10 consecutive iters >n).
                if m_act_now > n:
                    _bland_overcap_count += 1
                    if _bland_overcap_count >= _BLAND_OVERCAP_FAST_K:
                        if _DBG:
                            print(
                                f"[PCG] WATCHDOG  fast over-commit: "
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
                    # Two exit conditions: small-amplitude oscillation
                    # (Bland-degenerate cycle) OR persistent over-commit
                    # (m_act > n throughout the window — rank-deficient
                    # solve perturbs inactive constraints each iter so
                    # single-pivot adds find "new" violations forever).
                    if amp <= _BLAND_AMPLITUDE or win_min > n:
                        if _DBG:
                            reason = ("oscillation" if amp <= _BLAND_AMPLITUDE
                                      else "over-commit")
                            print(
                                f"[PCG] WATCHDOG  {reason}: m_act in "
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
                _t_pcg0  = _time.perf_counter()
            p, lam = self._pcg(lbfgs, g, A_act, b_act)
            if _DBG:
                _as_iter_count[0] += 1
                _t_pcg = _time.perf_counter() - _t_pcg0
                m_act = A_act.shape[0]
                print(
                    f"[PCG] AS_iter={_as_iter_count[0]:3d}  "
                    f"m_act={m_act:5d}  build={_t_build*1000:7.1f}ms  "
                    f"pcg={_t_pcg*1000:7.1f}ms  bland={bland}",
                    flush=True, file=_sys.stderr,
                )

            # Map lam entries back to typed arrays
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
            # Mirrors ImplicitLBFGSQP.  See _implicit_qp.py for the full
            # rationale on why block updates dominate at scale and why the
            # visited-set cycle detector is the safety net that drops back
            # to single-pivot Bland's rule.
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

                break

            # ---- Phase 1: add the most-violated inactive constraint ----
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

            # ---- Phase 2: remove most-negative-multiplier active constraint ----
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
                break   # KKT satisfied

        if _DBG:
            print(
                f"[PCG] EXIT solve  AS_iters={_as_iter_count[0]}  "
                f"total={(_time.perf_counter()-_t_qp)*1000:7.1f}ms",
                flush=True, file=_sys.stderr,
            )

        p = np.clip(p, lb, ub)
        self._active_ineq_soc = list(active_ineq)
        self._active_lb_soc   = list(active_lb)
        self._active_ub_soc   = list(active_ub)

        if self.warm_start:
            # Don't save over-committed state (see KKTSparseQP for full
            # rationale).  When the current solve ended with m_act > n
            # the watchdog likely fired; cold-start next time.
            m_act_final = (m_eq + len(active_ineq)
                           + len(active_lb) + len(active_ub))
            if m_act_final <= n:
                self._warm_active_ineq = list(active_ineq)
                self._warm_active_lb   = list(active_lb)
                self._warm_active_ub   = list(active_ub)
                self._warm_init = True
            else:
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
        """Second-order correction.  Reuse the cached active set; one PCG
        call with the trial-point constraint values on the RHS."""
        if self._lbfgs is None:
            return np.zeros(self.n), False

        lb_soc = xl - (x + p)
        ub_soc = xu - (x + p)

        A_act, b_act = self._build(
            c_eq_trial, self._J_eq,
            c_ineq_trial, self._J_ineq,
            lb_soc, ub_soc,
            self._active_ineq_soc,
            self._active_lb_soc,
            self._active_ub_soc,
        )
        d, _ = self._pcg(self._lbfgs, self._g, A_act, b_act)
        d = np.clip(d, lb_soc, ub_soc)
        return d, True
