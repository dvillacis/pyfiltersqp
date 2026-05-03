"""SQP outer loop."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .problem import NLPProblem
from .result import IterationInfo, SQPResult
from ._lbfgs import LBFGSMemory
from ._qp import QPSubproblem
from ._implicit_qp import ImplicitLBFGSQP
from ._qp_admm import WoodburyADMM
from ._qp_pcg import ProjectedCGQP
from ._qp_kkt import KKTSparseQP
from ._line_search import (
    l1_merit, directional_deriv, update_penalty, armijo_backtrack,
    filter_backtrack,
)
from ._filter import Filter, constraint_violation
from ._kkt import kkt_residuals


class SQPSolver:
    """
    SQP solver with L-BFGS Hessian approximation and l₁ merit line search.

    See DESIGN.md for a full description of the algorithm.

    Parameters
    ----------
    tol_feas : float
        Primal feasibility tolerance (default 1e-6).
    tol_opt : float
        Dual (stationarity) tolerance (default 1e-6).
    tol_comp : float
        Complementarity tolerance for inequality multipliers (default 1e-6).
    max_iter : int
        Maximum outer SQP iterations (default 200).
    max_ls_iter : int
        Maximum backtracking steps per line search (default 30).
    lbfgs_memory : int
        Number of (s, y) pairs retained in L-BFGS (default 10).
    mu0 : float
        Initial l₁ merit penalty parameter (default 1.0).
    mu_max : float
        Cap on the l₁ merit penalty parameter (default 1e10).  When the
        descent-direction condition asks for ``μ > mu_max``, ``μ`` is
        clamped at the cap and the armijo line search will fail (the
        directional derivative is no longer guaranteed negative).  The
        existing filter-fallback path then engages automatically — the
        intended escape when the merit framework can't make progress
        (TV-MCP-class problems where rank-deficient KKT solves leave a
        residual the merit can't penalize away).  HS-suite problems
        never approach this cap.
    step_max : float or None
        Optional cap on ``‖p‖∞`` for the QP step (default ``None``,
        unbounded).  When set, any step larger than the cap is scaled
        down uniformly before the line search.  Acts as a crude
        trust-region surrogate: the L-BFGS Hessian on
        rank-deficient/over-committed active sets sometimes returns
        wildly oversized steps (TV-MCP saw ``‖p‖`` of 1700+ at iters
        where useful steps were ``≤ 50``).  Clipping bounds the damage
        the line search has to claw back, and lets multiple consecutive
        small-α iters accumulate curvature instead of repeatedly
        triggering the L-BFGS reset watchdog.  Recommended for the
        ``kkt`` backend on equality-saturated problems; HS suite
        unaffected when left ``None``.
    rho : float
        Line search backtracking factor (default 0.5).
    eta : float
        Armijo sufficient decrease constant (default 1e-4).
    verbose : bool
        Print per-iteration diagnostics and populate result.history (default False).
    osqp_options : dict, optional
        Passed directly to QPSubproblem / OSQP.  Only consulted when
        ``backend="osqp"`` (deprecated path — see below).
    backend : {"auto", "implicit", "woodbury", "pcg", "kkt", "osqp"}
        QP backend selection (default "auto").

        * **auto** — pick "implicit", "pcg", or "woodbury" from problem
          shape.  See :meth:`SQPSolver._select_auto_backend` for the
          three-tier heuristic.  ``"kkt"`` is *not* auto-selected; it is
          opt-in only.
        * **implicit** — :class:`ImplicitLBFGSQP`, primal-dual active-set +
          Schur complement on ``LBFGSMemory._matvec``.  O(m_act · n · m).
        * **woodbury** — :class:`WoodburyADMM`, Woodbury-identity ADMM on
          the compact L-BFGS factors with a Schur-complement polish.  Cost
          per ADMM step is independent of the active set: O(n·m + m_constr²·m).
        * **pcg** — :class:`ProjectedCGQP`, active-set + projected CG on
          the L-BFGS Hessian.  O(cg_iters · n · m) per AS iter.
        * **kkt** — :class:`KKTSparseQP`, sparse augmented-KKT direct
          solve via ``scipy.sparse.linalg.splu`` plus a Sherman-Morrison-
          Woodbury overlay for the L-BFGS rank-2k Hessian correction.
          *Opt-in*: pass ``backend="kkt"`` explicitly.  Best for problems
          where ``m_eq + m_ineq ≈ n`` and the constraint Jacobian is
          sparse (PDE-constrained NLPs, image-bilevel MPCCs, structured
          OCP transcriptions) — exactly the regime where PCG hits
          rank-deficient ``A Aᵀ`` and Woodbury-ADMM hits its dense
          ``O(m_constr²·m)`` ``inv`` bottleneck.
        * **osqp** — *deprecated*.  :class:`QPSubproblem` materialises the
          dense n×n Hessian and feeds it to OSQP.  Slower than the implicit
          and Woodbury paths at every problem size in our benchmarks
          (≥10× slower at n≥1000).  Retained as a fallback for problems
          where the new backends unexpectedly fail; new code should not
          select it.
    use_implicit_qp : bool, deprecated
        Backward-compatibility alias for ``backend="implicit"``.  Prefer
        ``backend=`` for new code.
    use_woodbury_admm : bool, deprecated
        Backward-compatibility alias for ``backend="woodbury"``.  Prefer
        ``backend=`` for new code.
    """

    def __init__(
        self,
        tol_feas:     float = 1e-6,
        tol_opt:      float = 1e-6,
        tol_comp:     float = 1e-6,
        max_iter:     int   = 500,
        max_ls_iter:  int   = 30,
        lbfgs_memory: int   = 10,
        mu0:          float = 1.0,
        mu_max:       float = 1e10,
        step_max:     float | None = None,
        rho:          float = 0.5,
        eta:          float = 1e-4,
        verbose:      bool  = False,
        osqp_options:     dict | None = None,
        use_filter:       bool  = False,
        h_max:            float = 1e4,
        gamma_f:          float = 1e-5,
        gamma_h:          float = 1e-5,
        backend:           str  = "auto",
        use_implicit_qp:   bool = False,
        use_woodbury_admm: bool = False,
    ):
        self.tol_feas        = tol_feas
        self.tol_opt         = tol_opt
        self.tol_comp        = tol_comp
        self.max_iter        = max_iter
        self.max_ls_iter     = max_ls_iter
        self.lbfgs_memory    = lbfgs_memory
        self.mu0             = mu0
        self.mu_max          = mu_max
        self.step_max        = step_max
        self.rho             = rho
        self.eta             = eta
        self.verbose         = verbose
        self.osqp_options    = osqp_options or {}
        self.use_filter        = use_filter
        self.h_max             = h_max
        self.gamma_f           = gamma_f
        self.gamma_h           = gamma_h

        # Resolve QP backend.  The boolean flags remain as overrides for
        # backward compatibility; explicit ``backend=`` wins if both are set.
        if backend not in ("auto", "implicit", "woodbury", "pcg", "kkt", "osqp"):
            raise ValueError(
                f"backend must be one of 'auto', 'implicit', 'woodbury', "
                f"'pcg', 'kkt', 'osqp'; got {backend!r}"
            )
        if backend == "auto":
            if use_woodbury_admm:
                backend = "woodbury"
            elif use_implicit_qp:
                backend = "implicit"
        self.backend = backend
        self.use_implicit_qp   = use_implicit_qp   or backend == "implicit"
        self.use_woodbury_admm = use_woodbury_admm or backend == "woodbury"

    @staticmethod
    def _select_auto_backend(problem: NLPProblem) -> str:
        """
        Pick "implicit", "pcg", or "woodbury" based on the expected
        active-set size and constraint structure.

        Three QP backends, each with a different sweet spot:

        * **ImplicitLBFGSQP** — primal-dual active-set + direct Schur
          (lstsq on m_act × m_act).  O(m_act · n · memory) per AS step.
          Wins when m_act stays small relative to n.
        * **ProjectedCGQP** — same active-set wrapper but the inner
          equality solve is projected CG.  Inexact-Newton style: avoids
          the m_act³ Schur factorisation.  Wins on equality-heavy
          large-n problems where direct Schur is the bottleneck.
        * **WoodburyADMM** — ADMM with Woodbury identity on the compact
          L-BFGS factors.  Cost per step is independent of the active
          set; has fixed ~2 ms setup overhead.  Wins when bounds /
          inequalities saturate and the active-set methods churn.

        Three-tier heuristic, validated against CUTEst + large-scale
        Python benchmarks (2026-05-02):

        Small n (n ≤ 50):
            Bounds are usually mostly inactive, Woodbury setup dwarfs
            per-step work — pick implicit unless the inequality count
            alone is > 2n (KISSING2-like saturated active set).

        Large n (n > 50):
            * Active set provably small → implicit
            * Equality-heavy without dominant bounds → PCG
            * Otherwise (bounds- or inequality-heavy) → Woodbury

        Empirical wins:
        * unconstrained MGH → implicit wins by 5–250× (active set empty)
        * HS-family (n ≤ 10, all-bound) → implicit
        * eq_quadratic at n=3000–5000 → PCG by 2× over implicit, 5–6× over woodbury
        * Rosenbrock-chain at n=1000 (bound-heavy) → Woodbury by 28×
        * ineq_quadratic at n=1000 (m_ineq=200) → Woodbury by 22×
        * KISSING2 (n=94, m_ineq=625) → Woodbury (only one that converges)
        """
        n = problem.n
        m_eq, m_ineq = problem.m_eq, problem.m_ineq

        if n <= 50:
            # Saturated-inequality regime (e.g. CUTEst KISSING2): the active
            # set is dense and ImplicitLBFGSQP's Schur lstsq grows as O(m³).
            if m_ineq > 2 * n:
                return "woodbury"
            return "implicit"

        # --- Large n -----------------------------------------------------
        n_bound = int(np.sum(np.isfinite(problem.xl))
                      + np.sum(np.isfinite(problem.xu)))
        active_max = m_eq + m_ineq + n_bound

        # Active set provably small → direct Schur via Implicit.
        # Strict ``<`` so boundary cases (active_max == n/10, e.g.
        # eq_quadratic with default sizing) fall through to the PCG
        # check rather than getting stuck on direct Schur.
        if active_max < max(20, n // 10):
            return "implicit"

        # Equality-dominant and not bound-heavy → PCG.  ``m_eq > max(10, n/20)``
        # filters out problems where m_eq is incidental; ``n_bound < n/5``
        # excludes Rosenbrock-chain-style bound-heavy cases (PCG's
        # active-set churn explodes the same way ImplicitLBFGSQP's does);
        # ``m_ineq < m_eq`` excludes inequality-saturated cases (Woodbury
        # handles those better).
        if (m_eq > max(10, n // 20)
                and n_bound < n // 5
                and m_ineq < m_eq):
            return "pcg"

        # Default for large n: Woodbury.
        return "woodbury"

    def solve(
        self,
        problem: NLPProblem,
        x0: np.ndarray,
        lam_eq0:   np.ndarray | None = None,
        lam_ineq0: np.ndarray | None = None,
    ) -> SQPResult:
        """
        Solve the NLP.

        Parameters
        ----------
        problem : NLPProblem
        x0 : ndarray, shape (n,)
            Initial primal point.
        lam_eq0 : ndarray, shape (m_eq,), optional
            Initial equality multipliers (warm-start).
        lam_ineq0 : ndarray, shape (m_ineq,), optional
            Initial inequality multipliers (warm-start).

        Returns
        -------
        SQPResult
        """
        p = problem
        n, m_eq, m_ineq = p.n, p.m_eq, p.m_ineq

        # Optional unconditional entry print (env-var triggered).  Lets the
        # caller distinguish "stuck in pyfiltersqp" from "stuck before
        # pyfiltersqp" without needing verbose=True forwarded through
        # multi-layer adapters.  Set PYFILTERSQP_DEBUG=1 to enable.
        import os as _os
        if _os.environ.get("PYFILTERSQP_DEBUG"):
            import sys as _sys
            print(
                f"[PYFSQP] ENTER solve  n={n}  m_eq={m_eq}  m_ineq={m_ineq}  "
                f"backend={self.backend}",
                flush=True, file=_sys.stderr,
            )

        x       = np.asarray(x0, dtype=float).copy()
        lam_eq  = np.zeros(m_eq)  if lam_eq0  is None else np.asarray(lam_eq0,  dtype=float).copy()
        lam_ineq = np.zeros(m_ineq) if lam_ineq0 is None else np.asarray(lam_ineq0, dtype=float).copy()

        # Resolve "auto" once we have the problem in hand.
        backend = self.backend
        if backend == "auto":
            backend = self._select_auto_backend(p)
            self.use_implicit_qp   = (backend == "implicit")
            self.use_woodbury_admm = (backend == "woodbury")

        lbfgs = LBFGSMemory(n, memory=self.lbfgs_memory)
        if backend == "woodbury":
            qp = WoodburyADMM(n, m_eq, m_ineq)
        elif backend == "implicit":
            qp = ImplicitLBFGSQP(n, m_eq, m_ineq)
        elif backend == "pcg":
            qp = ProjectedCGQP(n, m_eq, m_ineq)
        elif backend == "kkt":
            qp = KKTSparseQP(n, m_eq, m_ineq)
        else:
            qp = QPSubproblem(n, m_eq, m_ineq, self.osqp_options)
        mu    = self.mu0
        history: list[IterationInfo] = []
        lam_lb = np.zeros(n)
        lam_ub = np.zeros(n)
        filt            = Filter(h_max=self.h_max, gamma_f=self.gamma_f, gamma_h=self.gamma_h)
        resto_count     = 0             # consecutive feasibility-restoration steps
        # Local mutable copy: the l₁ merit branch can promote this to True
        # mid-solve when it stalls (see "filter fallback" below) — sticky
        # once flipped, since the filter handles non-monotone iterates that
        # the l₁ merit refuses near degenerate active sets.
        use_filter_local = self.use_filter

        # Watchdog: count consecutive accepted iterations whose line-search
        # alpha collapsed below ALPHA_TINY.  When this hits ALPHA_TINY_K, the
        # L-BFGS curvature is almost certainly contaminated (tiny alpha →
        # tiny s → tiny but noisy y → next B is wronger → next alpha tinier).
        # Reset memory and let the next QP use a clean σI Hessian.
        ALPHA_TINY   = 1e-2
        ALPHA_TINY_K = 3
        tiny_alpha_count = 0

        # Helper: evaluate all problem functions at x.
        # Jacobians are preserved as sparse when the callable returns a scipy
        # sparse matrix; otherwise they are densified to float64 ndarray.
        def _eval(x):
            f = float(p.objective(x))
            g = np.asarray(p.gradient(x), dtype=float)
            if m_eq > 0:
                c_eq  = np.asarray(p.eq_constraints(x), dtype=float)
                _Jeq  = p.eq_jacobian(x)
                J_eq  = _Jeq if sp.issparse(_Jeq) else np.asarray(_Jeq, dtype=float)
            else:
                c_eq, J_eq = np.empty(0), None
            if m_ineq > 0:
                c_ineq  = np.asarray(p.ineq_constraints(x), dtype=float)
                _Jineq  = p.ineq_jacobian(x)
                J_ineq  = _Jineq if sp.issparse(_Jineq) else np.asarray(_Jineq, dtype=float)
            else:
                c_ineq, J_ineq = np.empty(0), None
            return f, g, c_eq, J_eq, c_ineq, J_ineq

        # Lagrangian gradient at x given multipliers (PLUS convention)
        # ∇_x L = g + J_eq^T λ_eq + J_ineq^T λ_ineq
        # np.asarray().ravel() handles both dense (no-op) and sparse (flattens
        # the result of sparse.T @ vec which may be a matrix row).
        def _grad_lag(g, J_eq, J_ineq, lam_eq, lam_ineq):
            gl = g.copy()
            if J_eq   is not None: gl += np.asarray(J_eq.T   @ lam_eq).ravel()
            if J_ineq is not None: gl += np.asarray(J_ineq.T @ lam_ineq).ravel()
            return gl

        f, g, c_eq, J_eq, c_ineq, J_ineq = _eval(x)

        # Gradient-aware L-BFGS H_0 initialisation for badly-scaled problems.
        # When ‖g‖ at x0 is huge (e.g. MGH variably_dimensioned at the
        # canonical x0 has ‖g‖ ≈ 10¹¹) the default H_0 = I gives a first
        # Newton step of magnitude ‖g‖, which collapses the line search at
        # iter 0 before L-BFGS can build a curvature estimate.  Only kick in
        # for extreme scales — otherwise leave the well-tuned H_0 = I path
        # untouched on regular problems where the OSQP QP is sensitive to
        # Hessian conditioning.
        g_inf = float(np.max(np.abs(g))) if g.size > 0 else 1.0
        if g_inf > 1e4:
            lbfgs._scale = 1.0 / g_inf

        if self.verbose:
            import sys, time as _time
            print(
                f"  pyfsqp ENTER solve  backend={backend}  n={n}  m_eq={m_eq}  "
                f"m_ineq={m_ineq}  ‖g0‖∞={g_inf:.2e}",
                flush=True,
            )
            _t_qp_total = [0.0]
            _t_qp_last  = [0.0]

        for k in range(self.max_iter):
            # 1. Solve QP subproblem
            if backend in ("woodbury", "implicit", "pcg", "kkt"):
                # Implicit / Woodbury / PCG / KKT paths: no dense B_k needed.
                # `update_penalty` uses lbfgs.matvec for pᵀ B p.
                B = lbfgs.matvec
                if self.verbose:
                    import time as _time
                    _t0 = _time.perf_counter()
                step, lam_eq_new, lam_ineq_new, lam_lb_new, lam_ub_new, qp_ok = qp.solve(
                    lbfgs, g, c_eq, J_eq, c_ineq, J_ineq, x, p.xl, p.xu
                )
                if self.verbose:
                    _t_qp_last[0] = _time.perf_counter() - _t0
                    _t_qp_total[0] += _t_qp_last[0]
                    print(
                        f"  pyfsqp iter={k:4d}  qp_solve={_t_qp_last[0]*1000:7.1f} ms  "
                        f"(cum={_t_qp_total[0]:.1f}s)  qp_ok={qp_ok}  |step|={float(np.linalg.norm(step)):.2e}",
                        flush=True,
                    )
            else:
                # Dense OSQP path: O(n²) materialise + OSQP ADMM (deprecated).
                B = lbfgs.materialise()
                step, lam_eq_new, lam_ineq_new, lam_lb_new, lam_ub_new, qp_ok = qp.solve(
                    B, g, c_eq, J_eq, c_ineq, J_ineq, x, p.xl, p.xu
                )

            # Step-size cap (trust-region surrogate).  When the QP
            # returns an oversized step — common when the L-BFGS Hessian
            # is essentially σI on a rank-deficient/over-committed active
            # set — uniformly scale it down to ‖p‖∞ ≤ step_max.  Multipliers
            # are left unscaled (they still represent the active-set
            # KKT solution); the line search will explore α·step, so any
            # KKT-quality scaling is handled there.
            if qp_ok and self.step_max is not None:
                step_inf = float(np.max(np.abs(step))) if step.size > 0 else 0.0
                if step_inf > self.step_max:
                    scale = self.step_max / step_inf
                    step = step * scale
                    if __import__("os").environ.get("PYFILTERSQP_DEBUG"):
                        import sys as _sys
                        print(
                            f"[PYFSQP] STEP-CAP  iter={k}  "
                            f"|p|_inf {step_inf:.2e} → {self.step_max:.2e}  "
                            f"(scale={scale:.2e})",
                            flush=True, file=_sys.stderr,
                        )

            if not qp_ok:
                if use_filter_local:
                    # Feasibility restoration: steepest descent on h = ‖c_eq‖₁ + ‖max(0,c_ineq)‖₁
                    h_curr = constraint_violation(c_eq, c_ineq)
                    h_grad = np.zeros(n)
                    if m_eq > 0 and J_eq is not None:
                        h_grad += np.asarray(J_eq.T @ np.sign(c_eq)).ravel()
                    if m_ineq > 0 and J_ineq is not None:
                        active = c_ineq > 0
                        if active.any():
                            idx = np.where(active)[0]
                            if sp.issparse(J_ineq):
                                h_grad += np.asarray(J_ineq[idx, :].sum(axis=0)).ravel()
                            else:
                                h_grad += J_ineq[idx].T @ np.ones(len(idx))
                    h_grad_norm = np.linalg.norm(h_grad)
                    restored = False
                    if h_grad_norm > 1e-14:
                        rest_dir = -h_grad / h_grad_norm
                        alpha_r = 1.0
                        for _ in range(self.max_ls_iter):
                            x_r = np.clip(x + alpha_r * rest_dir, p.xl, p.xu)
                            c_eq_r   = np.asarray(p.eq_constraints(x_r))   if m_eq   > 0 else np.empty(0)
                            c_ineq_r = np.asarray(p.ineq_constraints(x_r)) if m_ineq > 0 else np.empty(0)
                            if constraint_violation(c_eq_r, c_ineq_r) < (1.0 - 1e-4) * h_curr:
                                x = x_r
                                f, g, c_eq, J_eq, c_ineq, J_ineq = _eval(x)
                                restored = True
                                break
                            alpha_r *= 0.5
                    if not restored:
                        return self._make_result(x, f, lam_eq, lam_ineq, lam_lb, lam_ub, p, 3, k, history)
                    continue  # restart iteration with restored x
                else:
                    return self._make_result(x, f, lam_eq, lam_ineq, lam_lb, lam_ub, p, 3, k, history)

            # 2. Check KKT at current x with FRESH QP multipliers.
            # Using fresh multipliers avoids stale-multiplier false divergence: when
            # the solver steps exactly to x* the QP at x* gives p≈0 and the correct
            # KKT multipliers, so the convergence test fires immediately.
            kkt = kkt_residuals(g, c_eq, c_ineq, J_eq, J_ineq,
                                lam_eq_new, lam_ineq_new, lam_lb_new, lam_ub_new)
            if kkt.satisfied(self.tol_feas, self.tol_opt, self.tol_comp):
                return self._make_result(x, f, lam_eq_new, lam_ineq_new,
                                         lam_lb_new, lam_ub_new, p, 0, k, history)

            # Secondary convergence: floating-point step (QP returns p ≈ 0) combined
            # with primal feasibility and complementarity.  Fires when the L-BFGS SQP
            # has reached machine precision — the dual stationarity residual (||B p||)
            # is bounded below by floating-point noise so the primary KKT test may
            # never fire even at a genuine local minimum.
            step_tol = max(self.tol_opt, 1e-12) * max(1.0, float(np.linalg.norm(x)))
            if (float(np.linalg.norm(step)) < step_tol
                    and kkt.primal_eq   < self.tol_feas
                    and kkt.primal_ineq < self.tol_feas
                    and kkt.compl       < self.tol_comp):
                return self._make_result(x, f, lam_eq_new, lam_ineq_new,
                                         lam_lb_new, lam_ub_new, p, 0, k, history)

            # 3. Line search — filter (infeasible iterates) or l₁ merit
            h_curr = constraint_violation(c_eq, c_ineq) if use_filter_local else 0.0

            if use_filter_local and h_curr > 1e-6:
                # ---- Filter line search for infeasible iterates ----
                # Only infeasible points use the filter — feasible points fall
                # through to the l₁ merit below, which converges more cleanly.
                alpha, _, c_eq_new, c_ineq_new = filter_backtrack(
                    x, step, f, h_curr, c_eq, c_ineq,
                    filt,
                    p.objective,
                    p.eq_constraints   if m_eq   > 0 else None,
                    p.ineq_constraints if m_ineq > 0 else None,
                    rho=self.rho, max_iter=self.max_ls_iter,
                )

                if alpha > 0.0:
                    # Add current (f, h_curr) to filter only on f-type accepted
                    # steps.  h-type steps (violation decreases) do NOT augment
                    # the filter, keeping the envelope sparse and avoiding
                    # f-ceiling entries that would block h-reduction paths.
                    # (Fletcher-Leyffer 2002, §4.2)
                    h_trial = constraint_violation(c_eq_new, c_ineq_new)
                    h_type  = h_trial < (1.0 - filt.beta_h) * h_curr
                    if not h_type:
                        filt.add(f, h_curr)

                if alpha == 0.0:
                    # Second-order correction (Maratos fix): re-solve QP with
                    # constraint values at the full step x+p, then check the
                    # combined step p+d against the filter.
                    x_p      = x + step
                    c_eq_p   = np.asarray(p.eq_constraints(x_p))   if m_eq   > 0 else np.empty(0)
                    c_ineq_p = np.asarray(p.ineq_constraints(x_p)) if m_ineq > 0 else np.empty(0)
                    d_soc, soc_ok = qp.solve_soc(c_eq_p, c_ineq_p, x, step, p.xl, p.xu)
                    if soc_ok:
                        step_soc   = step + d_soc
                        x_soc      = x + step_soc
                        f_soc      = float(p.objective(x_soc))
                        c_eq_soc   = np.asarray(p.eq_constraints(x_soc))   if m_eq   > 0 else np.empty(0)
                        c_ineq_soc = np.asarray(p.ineq_constraints(x_soc)) if m_ineq > 0 else np.empty(0)
                        h_soc      = constraint_violation(c_eq_soc, c_ineq_soc)
                        if (filt.is_acceptable(f_soc, h_soc)
                                and filt.has_sufficient_decrease(f, h_curr, f_soc, h_soc)):
                            alpha, step = 1.0, step_soc

                if alpha == 0.0:
                    # Feasibility restoration: steepest descent on h to escape
                    # filter stagnation.  Bypasses the filter — the restored
                    # point re-enters the filter loop on the next iteration.
                    if resto_count < 5:
                        h_grad = np.zeros(n)
                        if m_eq > 0 and J_eq is not None:
                            h_grad += np.asarray(J_eq.T @ np.sign(c_eq)).ravel()
                        if m_ineq > 0 and J_ineq is not None:
                            active = c_ineq > 0
                            if active.any():
                                idx = np.where(active)[0]
                                if sp.issparse(J_ineq):
                                    h_grad += np.asarray(J_ineq[idx, :].sum(axis=0)).ravel()
                                else:
                                    h_grad += J_ineq[idx].T @ np.ones(len(idx))
                        h_grad_norm = np.linalg.norm(h_grad)
                        restored = False
                        if h_grad_norm > 1e-14:
                            rest_dir = -h_grad / h_grad_norm
                            alpha_r = 1.0
                            for _ in range(self.max_ls_iter):
                                x_r = np.clip(x + alpha_r * rest_dir, p.xl, p.xu)
                                c_eq_r   = np.asarray(p.eq_constraints(x_r))   if m_eq   > 0 else np.empty(0)
                                c_ineq_r = np.asarray(p.ineq_constraints(x_r)) if m_ineq > 0 else np.empty(0)
                                if constraint_violation(c_eq_r, c_ineq_r) < (1.0 - 1e-4) * h_curr:
                                    x = x_r
                                    f, g, c_eq, J_eq, c_ineq, J_ineq = _eval(x)
                                    restored = True
                                    break
                                alpha_r *= 0.5
                        if restored:
                            resto_count += 1
                            continue   # restart SQP iteration from restored x
                    return self._make_result(x, f, lam_eq_new, lam_ineq_new,
                                             lam_lb_new, lam_ub_new, p, 2, k, history)

                # Reset restoration counter on successful filter step
                resto_count = 0

            else:
                # ---- l₁ merit line search (v0.1 path AND feasible filter iterates) ----
                mu    = update_penalty(g, step, B, c_eq, c_ineq, mu,
                                       mu_max=self.mu_max)
                D_phi = directional_deriv(g, step, c_eq, c_ineq, mu)
                alpha, _, c_eq_new, c_ineq_new = armijo_backtrack(
                    x, step, f, c_eq, c_ineq, D_phi, mu,
                    p.objective,
                    p.eq_constraints   if m_eq   > 0 else None,
                    p.ineq_constraints if m_ineq > 0 else None,
                    rho=self.rho, eta=self.eta, max_iter=self.max_ls_iter,
                )

                if alpha <= 1e-10:
                    # Maratos correction: the QP step satisfies the *linearised*
                    # constraints at x+p, but the *true* constraints have
                    # quadratic curvature that can sabotage the l₁ merit decrease
                    # near a constrained optimum.  Re-solve the QP with c(x+p)
                    # on the RHS to get a correction d, then test the combined
                    # step x + (p + d).  Costs at most one extra QP solve
                    # (solve_soc reuses cached factors) plus one merit eval.
                    x_p     = x + step
                    c_eq_p  = (np.asarray(p.eq_constraints(x_p))
                               if m_eq   > 0 else np.empty(0))
                    c_ineq_p = (np.asarray(p.ineq_constraints(x_p))
                                if m_ineq > 0 else np.empty(0))
                    d_soc, soc_ok = qp.solve_soc(c_eq_p, c_ineq_p,
                                                 x, step, p.xl, p.xu)
                    if soc_ok:
                        step_soc = step + d_soc
                        x_soc    = x + step_soc
                        f_soc    = float(p.objective(x_soc))
                        c_eq_s   = (np.asarray(p.eq_constraints(x_soc))
                                    if m_eq   > 0 else np.empty(0))
                        c_ineq_s = (np.asarray(p.ineq_constraints(x_soc))
                                    if m_ineq > 0 else np.empty(0))
                        phi_curr = l1_merit(f,     c_eq,   c_ineq,   mu)
                        phi_soc  = l1_merit(f_soc, c_eq_s, c_ineq_s, mu)
                        if phi_soc < phi_curr:
                            alpha      = 1.0
                            step       = step_soc
                            c_eq_new   = c_eq_s
                            c_ineq_new = c_ineq_s

                if alpha <= 1e-10:
                    # Filter fallback: l₁ merit + SOC both refused this step.
                    # The remaining failure mode (CUTEst KISSING2 et al.) is
                    # stagnation near a degenerate KKT point where the strict-
                    # decrease l₁ merit can't accept any further iterate but
                    # the filter — which admits non-monotone (f, h) progress —
                    # can.  Switch line searches sticky for the rest of the
                    # solve and retry the *same* QP step under filter rules.
                    use_filter_local = True
                    h_curr = constraint_violation(c_eq, c_ineq)
                    alpha, _, c_eq_new, c_ineq_new = filter_backtrack(
                        x, step, f, h_curr, c_eq, c_ineq,
                        filt,
                        p.objective,
                        p.eq_constraints   if m_eq   > 0 else None,
                        p.ineq_constraints if m_ineq > 0 else None,
                        rho=self.rho, max_iter=self.max_ls_iter,
                    )
                    if alpha > 0.0:
                        h_trial = constraint_violation(c_eq_new, c_ineq_new)
                        if not (h_trial < (1.0 - filt.beta_h) * h_curr):
                            filt.add(f, h_curr)

                if alpha <= 1e-10:
                    if __import__("os").environ.get("PYFILTERSQP_DEBUG"):
                        import sys as _sys
                        h_now = constraint_violation(c_eq, c_ineq)
                        print(
                            f"[PYFSQP] LINE-SEARCH-FAIL  iter={k}  "
                            f"l1+SOC+filter all refused  alpha<=1e-10  "
                            f"|step|={float(np.linalg.norm(step)):.2e}  "
                            f"f={f:.4e}  h={h_now:.2e}  mu={mu:.2e}",
                            flush=True, file=_sys.stderr,
                        )
                    return self._make_result(x, f, lam_eq_new, lam_ineq_new,
                                             lam_lb_new, lam_ub_new, p, 2, k, history)

            # 5. Accept step
            x_new = x + alpha * step
            f_new, g_new, c_eq_new, J_eq_new, c_ineq_new, J_ineq_new = _eval(x_new)

            # 6. L-BFGS update (use new multipliers for y)
            gl_old = _grad_lag(g,     J_eq,     J_ineq,     lam_eq_new, lam_ineq_new)
            gl_new = _grad_lag(g_new, J_eq_new, J_ineq_new, lam_eq_new, lam_ineq_new)
            lbfgs.update(x_new - x, gl_new - gl_old)

            # Watchdog: tiny-alpha streak ⇒ L-BFGS is misleading the QP.
            # Reset memory so the next iter starts from a clean scaled-I
            # Hessian, breaking the bad-B → tiny-α → bad-y → worse-B loop.
            if alpha < ALPHA_TINY:
                tiny_alpha_count += 1
                if tiny_alpha_count >= ALPHA_TINY_K:
                    lbfgs.reset()
                    g_inf = float(np.max(np.abs(g_new))) if g_new.size > 0 else 1.0
                    if g_inf > 1.0:
                        lbfgs._scale = 1.0 / g_inf
                    tiny_alpha_count = 0
            else:
                tiny_alpha_count = 0

            # Env-var-triggered per-iter debug (mirror of verbose=True).
            # Useful when verbose isn't forwarded through multi-layer
            # adapters (e.g. pympcc → _filtersqp_adapter → SQPSolver).
            if (self.verbose
                    or __import__("os").environ.get("PYFILTERSQP_DEBUG")):
                kkt_new = kkt_residuals(g_new, c_eq_new, c_ineq_new,
                                        J_eq_new, J_ineq_new, lam_eq_new, lam_ineq_new,
                                        lam_lb_new, lam_ub_new)
            if self.verbose:
                history.append(IterationInfo(
                    iteration=k,
                    obj=f_new,
                    primal_eq=kkt_new.primal_eq,
                    primal_ineq=kkt_new.primal_ineq,
                    dual=kkt_new.dual,
                    compl=kkt_new.compl,
                    step_norm=float(np.linalg.norm(alpha * step)),
                    step_size=alpha,
                    qp_iters=0,   # TODO: extract from OSQP result
                ))
            if (self.verbose
                    or __import__("os").environ.get("PYFILTERSQP_DEBUG")):
                import sys
                print(
                    f"  pyfsqp iter={k:4d}  obj={f_new:+.4e}  "
                    f"feas_eq={kkt_new.primal_eq:.2e}  feas_in={kkt_new.primal_ineq:.2e}  "
                    f"dual={kkt_new.dual:.2e}  compl={kkt_new.compl:.2e}  "
                    f"alpha={alpha:.2e}  |p|={float(np.linalg.norm(alpha*step)):.2e}",
                    flush=True,
                )

            # 7. Advance
            x, f, g        = x_new, f_new, g_new
            c_eq, J_eq     = c_eq_new, J_eq_new
            c_ineq, J_ineq = c_ineq_new, J_ineq_new
            lam_eq, lam_ineq = lam_eq_new, lam_ineq_new
            lam_lb, lam_ub   = lam_lb_new, lam_ub_new

        return self._make_result(x, f, lam_eq, lam_ineq, lam_lb, lam_ub, problem, 1, self.max_iter, history)

    @staticmethod
    def _make_result(x, f, lam_eq, lam_ineq, lam_lb, lam_ub, problem, status, iterations, history):
        return SQPResult(
            x=x.copy(),
            obj=f,
            lam_eq=lam_eq.copy(),
            lam_ineq=lam_ineq.copy(),
            lam_lb=lam_lb.copy(),
            lam_ub=lam_ub.copy(),
            status=status,
            message="",           # filled by SQPResult.__post_init__
            success=(status == 0),
            iterations=iterations,
            history=history,
        )


def solve(
    problem: NLPProblem,
    x0: np.ndarray,
    lam_eq0:   np.ndarray | None = None,
    lam_ineq0: np.ndarray | None = None,
    **solver_kwargs,
) -> SQPResult:
    """
    Convenience wrapper: construct an :class:`SQPSolver` and solve.

    All keyword arguments are forwarded to :class:`SQPSolver`.

    Examples
    --------
    >>> result = solve(problem, x0, tol_feas=1e-8, verbose=True)
    """
    return SQPSolver(**solver_kwargs).solve(problem, x0, lam_eq0, lam_ineq0)
