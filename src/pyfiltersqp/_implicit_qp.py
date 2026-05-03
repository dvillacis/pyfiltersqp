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
    block_active_set : bool
        Add/remove all violated/negative-multiplier constraints per AS iter
        (default ``True``).  Single-pivot Bland's-rule fallback engages
        automatically when the visited-set cycle detector fires, preserving
        the original convergence guarantees.  At m_act ≫ 1 this turns
        ``O(m_act)`` AS iters into ``O(log m_act)`` and is the dominant
        speed-up at scale.
    warm_start : bool
        Reuse the active set from the previous ``solve()`` call as the
        initial set of the next call (default ``True``).  Constraint
        indices refer to fixed problem rows so the warm set is always
        well-defined; the AS loop drops/adds as needed at the new x.
    """

    def __init__(
        self,
        n: int,
        m_eq: int,
        m_ineq: int,
        tol_active: float = 1e-8,
        tol_kkt: float = 1e-10,
        max_as_iter: int | None = None,
        block_active_set: bool = True,
        warm_start: bool = True,
    ):
        self.n          = n
        self.m_eq       = m_eq
        self.m_ineq     = m_ineq
        self.tol_active = tol_active
        self.tol_kkt    = tol_kkt
        self.max_as_iter = max_as_iter or (2 * n + m_eq + m_ineq + 20)
        self.block_active_set = block_active_set
        self.warm_start = warm_start

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

        # Cross-call warm-start state (active set from previous solve()).
        # ``_warm_init`` flips True after the first successful solve(), so
        # the next solve() seeds its initial active set from these lists
        # instead of cold-starting from c_ineq > -tol_active.
        self._warm_init: bool = False
        self._warm_active_ineq: list[int] = []
        self._warm_active_lb:   list[int] = []
        self._warm_active_ub:   list[int] = []

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
    ):
        """
        Build A_act (m_act × n) and b_act (m_act,) for the current active set.

        Equalities are always the first m_eq rows; ineq, lb, ub follow.

        Returns ``A_act`` as a sparse CSR matrix when either input Jacobian
        is sparse, dense ndarray otherwise.  ``_schur_solve`` and the PCG
        inner loop accept both.
        """
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

    def _build_sparse(
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
    ):
        """
        Sparse fast path: build ``A_act`` as a CSR matrix via row slicing
        + ``sp.vstack`` instead of densifying each row.

        Same row layout as :meth:`_build`: equalities, active ineq, active
        lb, active ub — multiplier mapping in :meth:`solve` depends on it.

        Cost: O(nnz(J_eq) + nnz(J_ineq[active_ineq, :]) + |active_lb| +
        |active_ub|).  When the original Jacobians are themselves sparse
        with O(n) non-zeros (e.g. PDE-discretised constraints), this is
        ``m_act ×`` faster than the dense per-row densify.
        """
        n, m_eq = self.n, self.m_eq
        blocks: list = []
        b_parts: list = []

        if m_eq > 0:
            J_eq_csr = J_eq.tocsr() if sp.issparse(J_eq) else sp.csr_matrix(J_eq)
            blocks.append(J_eq_csr)
            b_parts.append(-np.asarray(c_eq, dtype=float))

        if active_ineq:
            if sp.issparse(J_ineq):
                J_ineq_csr = J_ineq.tocsr()
            else:
                J_ineq_csr = sp.csr_matrix(J_ineq)
            blocks.append(J_ineq_csr[active_ineq, :])
            b_parts.append(-np.asarray(c_ineq, dtype=float)[active_ineq])

        n_lb, n_ub = len(active_lb), len(active_ub)
        if n_lb + n_ub > 0:
            # Single COO build for the bound block: row i has ±1 at col k.
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

    @staticmethod
    def _schur_solve(
        lbfgs: LBFGSMemory,
        g: np.ndarray,
        A_act,
        b_act: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Equality-constrained QP via Schur complement.

            min (1/2) pᵀ B p + gᵀ p   s.t.  A_act p = b_act

        ``A_act`` may be a dense ndarray or a sparse CSR matrix; the only
        operation requiring a dense form is the batched ``_matvec_batch``
        on ``A_actᵀ``, where the output is intrinsically dense (n × m_act).

        Returns p (n,) and lam (m_act,).
        Cost: O(m_act · n · memory) + O(m_act³).
        """
        Hg    = lbfgs._matvec(g)
        m_act = A_act.shape[0]

        if m_act == 0:
            return -Hg, np.empty(0)

        # H A_actᵀ in one batched two-loop sweep — shape (n, m_act).
        # ~m× faster than calling _matvec column-by-column for typical m ≤ 10.
        # _matvec_batch needs a dense input; for sparse A_act we densify
        # the transpose here (cost O(n · m_act), unavoidable for L-BFGS).
        if sp.issparse(A_act):
            A_T_dense = np.asarray(A_act.T.toarray())
        else:
            A_T_dense = np.asarray(A_act.T)
        HA_T = lbfgs._matvec_batch(A_T_dense)

        # Schur complement S = A_act H A_actᵀ (m_act × m_act).
        # Sparse @ dense → dense for sparse A_act; dense @ dense for dense.
        S = np.asarray(A_act @ HA_T)

        # S lam = -(b_act + A_act Hg)
        AHg = np.asarray(A_act @ Hg).ravel()
        rhs = -(b_act + AHg)
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
        # Cold start: only currently violated inequalities + truly fixed
        # variables.  Bounds are NOT pre-populated — the add-phase discovers
        # them when the unconstrained step violates a bound.  Pre-adding
        # bounds where x is currently at the boundary over-constrains the
        # system and can drive the active-set to a wrong local KKT point.
        # Warm start: take the previous solve()'s final active set verbatim.
        # Constraint indices are stable across SQP iterations (the problem
        # rows don't change), so warm indices remain meaningful even though
        # the linearisation point moved.  The block-remove phase prunes any
        # constraints that turn out to be inactive at the new x.
        active_ineq: list[int] = []
        active_lb:   list[int] = []
        active_ub:   list[int] = []

        if self.warm_start and self._warm_init:
            active_ineq = [j for j in self._warm_active_ineq if 0 <= j < m_ineq]
            active_lb   = [k for k in self._warm_active_lb   if 0 <= k < n]
            active_ub   = [k for k in self._warm_active_ub   if 0 <= k < n]
            # Always include truly fixed variables (warm set may have lost
            # them if bounds widened then narrowed across SQP iters).
            in_lb = set(active_lb)
            in_ub = set(active_ub)
            for k in range(n):
                if ub[k] - lb[k] < tol and k not in in_lb and k not in in_ub:
                    active_lb.append(k)
        else:
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

        # Cycle watchdog (see :class:`ProjectedCGQP` for the full rationale).
        # Tracks m_act over the K most recent iters spent in bland mode and
        # exits the AS loop when those values collapse to ≤ 2 distinct
        # numbers — i.e. when the active set is flip-flopping between
        # equivalent rank-deficient KKT solutions and Bland's rule alone
        # cannot break the cycle.
        from collections import deque as _deque
        _BLAND_K          = 30
        _BLAND_AMPLITUDE  = 5
        _bland_window     = _deque(maxlen=_BLAND_K)
        _OVERCAP_K        = 3
        _overcap_window   = _deque(maxlen=_OVERCAP_K)
        _BLAND_OVERCAP_FAST_K = 10
        _bland_overcap_count = 0

        for _ in range(self.max_as_iter):
            # Hash the active set; if seen before, we've started cycling and
            # switch to Bland's rule for the rest of this solve.
            key = (frozenset(active_lb), frozenset(active_ub),
                   frozenset(active_ineq))
            if key in visited:
                bland = True
            else:
                visited.add(key)

            # Force single-pivot when m_act exceeds n — block-add can
            # otherwise toggle large groups of rows forever in the rank-
            # deficient regime without ever revisiting the same exact set.
            # See ProjectedCGQP for the full rationale.
            m_act_now = m_eq + len(active_ineq) + len(active_lb) + len(active_ub)
            _overcap_window.append(m_act_now > n)
            if (not bland
                    and len(_overcap_window) == _OVERCAP_K
                    and all(_overcap_window)):
                bland = True

            if bland:
                if m_act_now > n:
                    _bland_overcap_count += 1
                    if _bland_overcap_count >= _BLAND_OVERCAP_FAST_K:
                        break  # fast over-commit exit
                else:
                    _bland_overcap_count = 0

                _bland_window.append(m_act_now)
                if len(_bland_window) == _BLAND_K:
                    win_min = min(_bland_window)
                    amp = max(_bland_window) - win_min
                    # Cycle (small-amp oscillation) OR over-commit (m_act
                    # stuck above n).  See ProjectedCGQP for full rationale.
                    if amp <= _BLAND_AMPLITUDE or win_min > n:
                        break
            elif _bland_window:
                _bland_window.clear()
                _bland_overcap_count = 0

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

            # ---- Block mode (default): bulk add / bulk remove per iter ----
            # When the visited-set cycle detector has not fired we add ALL
            # currently violated inactive constraints (or remove ALL active
            # constraints with negative multipliers) in a single iteration.
            # This converges in O(log m_act) iters at scale instead of
            # O(m_act).  When the visited-set repeats — i.e. block updates
            # have started oscillating between two large sets — ``bland``
            # flips True and we drop into the single-pivot Bland's-rule
            # path below, which preserves the original termination proof.
            if self.block_active_set and not bland:
                # --- block add ---
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
                    # Block-add disables single-pivot tabu since we no
                    # longer track a single (kind, index) addition.
                    last_added = None
                    last_removed = None
                    continue

                # --- block remove ---
                # Remove all active inequality / bound constraints whose
                # KKT multiplier is sufficiently negative.  Equalities are
                # never removed.  Truly fixed variables (lb ≈ ub) keep
                # their bound active regardless of sign — removing it
                # would unfix the variable and re-trigger an add next iter.
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

                # No block adds, no block removes — KKT satisfied.
                break

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

        # Stash for the next solve()'s warm-start.  Indices refer to
        # fixed problem rows so they remain valid across SQP iterations.
        # Skip the save when the current solve ended over-committed
        # (m_act > n) — see KKTSparseQP for full rationale.
        if self.warm_start:
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
