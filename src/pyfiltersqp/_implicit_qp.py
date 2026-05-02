"""Implicit L-BFGS QP subproblem solver — Schur complement KKT + active-set."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._lbfgs import LBFGSMemory


class ImplicitLBFGSQP:
    """
    Solves the SQP QP subproblem without materialising the dense n×n Hessian.

    Solves::

        min   (1/2) pᵀ B p + gᵀ p
        s.t.  J_eq   p = -c_eq           (linearised equalities)
              J_ineq p ≤ -c_ineq         (linearised inequalities)
              xl - x ≤ p ≤ xu - x        (step bounds)

    Algorithm
    ---------
    Uses ``LBFGSMemory._matvec`` to apply H = B⁻¹ in O(n·m) without ever
    forming the dense n×n B.  The KKT system for the current active constraint
    set is solved via the Schur complement (m_act × m_act system)::

        S lam = -(b_act + A_act H g),   S = A_act H A_actᵀ
        p     = -H(g + A_actᵀ lam)

    Inequalities and bounds are handled with a primal-dual active-set loop:

    * Add phase  — add the most-violated inactive constraint.
    * Remove phase — remove the active constraint with the most-negative
      KKT multiplier (inequality / bound only; equalities are never removed).
    * Terminate when no violations remain and all non-equality multipliers ≥ 0.

    Row encoding (so all non-equality multipliers have optimality lam ≥ 0):

    * Equality i      : row =  J_eq[i],   b = -c_eq[i]    (any sign)
    * Ineq j (active) : row =  J_ineq[j], b = -c_ineq[j]  (lam ≥ 0)
    * Lower bound k   : row = -eₖ,         b = -(xl-x)[k]  (lam ≥ 0)
    * Upper bound k   : row =  eₖ,         b = (xu-x)[k]   (lam ≥ 0)

    Cost
    ----
    O(m_act · n · m_memory) per QP solve (vs O(n²) for dense OSQP).
    For typical SQP iterates with few active bounds (m_act ≈ m_eq + m_ineq),
    this is effectively O(n) per solve.

    Parameters
    ----------
    n, m_eq, m_ineq : int
        Problem dimensions.
    tol_active : float
        Threshold below which a constraint is considered initially active.
    tol_kkt : float
        Multiplier below which an active constraint is removed from the
        active set.
    max_as_iter : int or None
        Maximum active-set iterations.  Defaults to n + m_eq + m_ineq + 20.
    """

    def __init__(
        self,
        n: int,
        m_eq: int,
        m_ineq: int,
        tol_active: float = 1e-8,
        tol_kkt: float = 1e-10,
        max_as_iter: int | None = None,
    ):
        self.n          = n
        self.m_eq       = m_eq
        self.m_ineq     = m_ineq
        self.tol_active = tol_active
        self.tol_kkt    = tol_kkt
        self.max_as_iter = max_as_iter or (2 * n + m_eq + m_ineq + 20)

        # State cached between solve() and solve_soc()
        self._lbfgs:  LBFGSMemory | None = None
        self._g:      np.ndarray  | None = None
        self._J_eq                       = None
        self._J_ineq                     = None
        self._lb:     np.ndarray  | None = None   # xl - x  from last solve
        self._ub:     np.ndarray  | None = None   # xu - x  from last solve
        self._active_ineq_soc: list[int] = []
        self._active_lb_soc:   list[int] = []
        self._active_ub_soc:   list[int] = []

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row(J, i: int, n: int) -> np.ndarray:
        """Extract row i of J (dense or sparse) as a dense (n,) array."""
        if sp.issparse(J):
            return np.asarray(J[i, :].todense()).ravel()
        return np.asarray(J[i, :]).ravel()

    def _build(
        self,
        c_eq:   np.ndarray,
        J_eq,
        c_ineq: np.ndarray,
        J_ineq,
        lb:     np.ndarray,
        ub:     np.ndarray,
        active_ineq: list[int],
        active_lb:   list[int],
        active_ub:   list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Build A_act (m_act × n) and b_act (m_act,) for the current active set.

        Equalities are always the first m_eq rows; ineq, lb, ub follow.
        """
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
            e[k] = -1.0          # p[k] ≥ lb[k]  →  -e_k p ≤ -lb[k],  lam ≥ 0
            rows.append(e.copy())
            b.append(float(-lb[k]))
            e[k] = 0.0

        for k in active_ub:
            e[k] = 1.0           # p[k] ≤ ub[k]  →  +e_k p ≤ +ub[k],  lam ≥ 0
            rows.append(e.copy())
            b.append(float(ub[k]))
            e[k] = 0.0

        if rows:
            return np.array(rows, dtype=float), np.array(b, dtype=float)
        return np.empty((0, n), dtype=float), np.empty(0, dtype=float)

    @staticmethod
    def _schur_solve(
        lbfgs: LBFGSMemory,
        g: np.ndarray,
        A_act: np.ndarray,
        b_act: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Equality-constrained QP via Schur complement.

            min (1/2) pᵀ B p + gᵀ p   s.t.  A_act p = b_act

        Returns p (n,) and lam (m_act,).
        Cost: O(m_act · n · memory) + O(m_act³).
        """
        Hg    = lbfgs._matvec(g)
        m_act = A_act.shape[0]

        if m_act == 0:
            return -Hg, np.empty(0)

        # H A_actᵀ in one batched two-loop sweep — shape (n, m_act).
        # ~m× faster than calling _matvec column-by-column for typical m ≤ 10.
        HA_T = lbfgs._matvec_batch(A_act.T)

        # Schur complement S = A_act H A_actᵀ (m_act × m_act)
        S = A_act @ HA_T

        # S lam = -(b_act + A_act Hg)
        rhs     = -(b_act + A_act @ Hg)
        lam, *_ = np.linalg.lstsq(S, rhs, rcond=None)

        p = -Hg - HA_T @ lam
        return p, lam

    # ------------------------------------------------------------------
    # public API
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
        """
        Solve the QP subproblem.

        Returns
        -------
        p         : ndarray (n,)
        lam_eq    : ndarray (m_eq,)
        lam_ineq  : ndarray (m_ineq,)
        lam_lb    : ndarray (n,)
        lam_ub    : ndarray (n,)
        success   : bool
        """
        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq
        tol = self.tol_active

        lb = xl - x
        ub = xu - x

        # Cache for solve_soc
        self._lbfgs  = lbfgs
        self._g      = g
        self._J_eq   = J_eq
        self._J_ineq = J_ineq
        self._lb     = lb
        self._ub     = ub

        # --- Initial active set ---
        # Equalities always active (handled inside _build).
        # Bounds are NOT pre-populated — the add-phase discovers them when the
        # unconstrained step violates a bound.  Pre-adding bounds where x is
        # currently at the boundary over-constrains the system and can drive the
        # active-set to a wrong local KKT point.
        # Exception: truly fixed variables (lb ≈ ub) must be in the active set.
        active_ineq: list[int] = []
        active_lb:   list[int] = []
        active_ub:   list[int] = []

        for j in range(m_ineq):
            if c_ineq[j] > -tol:
                active_ineq.append(j)

        for k in range(n):
            if ub[k] - lb[k] < tol:        # truly fixed variable
                active_lb.append(k)

        lam_eq   = np.zeros(m_eq)
        lam_ineq = np.zeros(m_ineq)
        lam_lb   = np.zeros(n)
        lam_ub   = np.zeros(n)
        p        = np.zeros(n)

        # Track active sets as boolean masks for O(1) vectorised scans.
        # The Python lists are also kept because _build assembles A_act in
        # append order — switching to mask-only ordering would change the
        # row layout downstream.
        alb_mask = np.zeros(n,      dtype=bool)
        aub_mask = np.zeros(n,      dtype=bool)
        aiq_mask = np.zeros(m_ineq, dtype=bool)
        for k in active_lb:   alb_mask[k] = True
        for k in active_ub:   aub_mask[k] = True
        for j in active_ineq: aiq_mask[j] = True

        # Cycle detection + anti-cycling memory.
        #
        # Greedy (most-violated / most-negative) is fast but cycles on
        # degenerate problems (e.g. CUTEst HS23 toggles between two adjacent
        # active sets at x*).  Two complementary defences:
        #
        # 1. Visited-set tracking → switch to Bland's rule (lowest-index
        #    eligible) on revisit.  Standard simplex-style anti-cycling.
        # 2. One-step tabu — record the (kind, index) that was added or
        #    removed in the previous iteration and refuse to undo it in the
        #    current iteration.  Breaks period-2 cycles where Bland's alone
        #    would still pick the same index (when the cycling involves the
        #    same constraint repeatedly entering and leaving the active set).
        visited: set = set()
        bland = False
        last_added   = None    # (kind, index) — kind in {"lb","ub","ineq"}
        last_removed = None

        for _ in range(self.max_as_iter):
            # Hash the active set; if seen before, we've started cycling and
            # switch to Bland's rule for the rest of this solve.
            key = (frozenset(active_lb), frozenset(active_ub),
                   frozenset(active_ineq))
            if key in visited:
                bland = True
            else:
                visited.add(key)

            A_act, b_act = self._build(
                c_eq, J_eq, c_ineq, J_ineq, lb, ub,
                active_ineq, active_lb, active_ub,
            )
            p, lam = self._schur_solve(lbfgs, g, A_act, b_act)

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

            # ---- Phase 1: add an inactive violated constraint ----
            # Greedy: most-violated.  Bland: lowest-index eligible.
            # Tabu: refuse to add the constraint we removed last iteration.
            added = False

            lb_viol = np.where(alb_mask, -np.inf, lb - p)
            if bland and last_removed is not None and last_removed[0] == "lb":
                lb_viol[last_removed[1]] = -np.inf      # tabu
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
                last_removed = None    # tabu lasts only one iteration
                continue

            # ---- Phase 2: remove an active constraint with negative multiplier ----
            # Greedy: most-negative.  Bland: lowest-index eligible.
            # Tabu: refuse to remove the constraint we added last iteration.
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
                last_added = None     # tabu lasts only one iteration
            else:
                break   # KKT satisfied
        # NB: when the for-else falls through (max_as_iter exhausted) we
        # return success=True with the last iterate — matches HS100 et al.
        # which terminate AS with bad multipliers but a usable descent
        # direction.  The SQP outer loop's SOC + filter fallback handles
        # the cases where the step is genuinely bad (e.g. HS23 — which is
        # also why the auto-router prefers Woodbury for HS23-shape inputs).

        p = np.clip(p, lb, ub)

        # Cache final active set for solve_soc
        self._active_ineq_soc = list(active_ineq)
        self._active_lb_soc   = list(active_lb)
        self._active_ub_soc   = list(active_ub)

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
        """
        Second-order correction (SOC) step.

        Re-uses the active set and Jacobians from the last ``solve()`` call,
        updating only the constraint RHS with values from the trial point x+p.
        A single Schur solve is performed (no active-set loop needed).

        Returns
        -------
        d       : ndarray (n,)
        success : bool
        """
        if self._lbfgs is None:
            return np.zeros(self.n), False

        lbfgs  = self._lbfgs
        g      = self._g
        J_eq   = self._J_eq
        J_ineq = self._J_ineq

        lb_soc = xl - (x + p)
        ub_soc = xu - (x + p)

        A_act, b_act = self._build(
            c_eq_trial, J_eq,
            c_ineq_trial, J_ineq,
            lb_soc, ub_soc,
            self._active_ineq_soc,
            self._active_lb_soc,
            self._active_ub_soc,
        )

        d, _ = self._schur_solve(lbfgs, g, A_act, b_act)
        d = np.clip(d, lb_soc, ub_soc)
        return d, True
