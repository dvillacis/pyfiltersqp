"""Woodbury-ADMM QP subproblem solver — O(n·m) per step, no dense B_k needed."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ._lbfgs import LBFGSMemory


class WoodburyADMM:
    """
    Solves the SQP QP subproblem without materialising B_k::

        min   (1/2) pᵀ B_k p + gᵀ p
        s.t.  J_eq   p = −c_eq           (m_eq equalities)
              J_ineq p ≤ −c_ineq         (m_ineq inequalities)
              xl−x ≤ p ≤ xu−x            (step bounds)

    Algorithm
    ---------
    ADMM on the consensus form A p = z, z ∈ K::

        A = [J_eq; J_ineq; Iₙ],   K = {−c_eq} × (−∞,−c_ineq] × [lb,ub]

    **p-update** — two-step Woodbury applied to (B_k + ρ AᵀA)⁻¹:

    1. H_ρ = (B_k + ρI)⁻¹ via the 2m×2m matrix from ``lbfgs.factors()``::

           C = M⁻¹ + ΨᵀΨ/(σ+ρ)   (2m×2m, precomputed once)
           H_ρ v = v/(σ+ρ) − Ψ (C⁻¹ Ψᵀ v) / (σ+ρ)²

    2. Schur correction using the m_constr×m_constr matrix::

           H_ρ_JᵀC = H_ρ J_constrᵀ   (n × m_constr, precomputed once)
           S = Iₘ + ρ J_constr H_ρ_JᵀC   (m_constr×m_constr)
           (B_k+ρAᵀA)⁻¹ rhs = H_ρ rhs − H_ρ_JᵀC (S⁻¹ (ρ J_constr H_ρ rhs))

    **z-update** (O(n + m_constr)):

    * Equalities:   z_eq   = −c_eq                          (fixed projection)
    * Inequalities: z_ineq = min(J_ineq p + u_ineq, −c_ineq)
    * Bounds:       z_bnd  = clip(p + u_bnd, lb, ub)

    **u-update**: standard scaled ADMM dual u += Ap − z.

    Multipliers extracted at convergence (λ = ρ u, matching OSQP convention)::

        lam_eq   = ρ u[:m_eq]
        lam_ineq = max(0, ρ u[m_eq:m_eq+m_ineq])
        lam_lb   = max(0, −ρ u_bnd)
        lam_ub   = max(0,  ρ u_bnd)

    Cost
    ----
    Setup: O(n·m²) for ``factors()`` + O(n·m·m_constr) for H_ρ_JᵀC.
    Per ADMM step: O(n·(m + m_constr)).  For typical m ≤ 10, m_constr ≤ 20:
    ~60n floating-point operations per step.

    Parameters
    ----------
    n, m_eq, m_ineq : int
        Problem dimensions.
    rho : float
        Initial ADMM penalty parameter (default 1.0).
    max_iter : int
        Maximum ADMM iterations per QP solve (default 500).
    tol_abs, tol_rel : float
        Absolute / relative convergence tolerances (OSQP defaults: 1e-6).
    rho_update_freq : int
        Steps between ρ adaptation checks (default 10).
    rho_balance : float
        Ratio threshold triggering ρ update (default 10.0).
    rho_min, rho_max : float
        Clamp ρ after adaptation.
    """

    def __init__(
        self,
        n: int,
        m_eq: int,
        m_ineq: int,
        rho: float = 1.0,
        max_iter: int = 1000,
        tol_abs: float = 1e-6,
        tol_rel: float = 1e-6,
        rho_min: float = 1e-6,
        rho_max: float = 1e6,
        rho_update_freq: int = 10,
        rho_balance: float = 3.0,
    ):
        self.n = n
        self.m_eq = m_eq
        self.m_ineq = m_ineq
        self.rho = rho
        self.max_iter = max_iter
        self.tol_abs = tol_abs
        self.tol_rel = tol_rel
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.rho_update_freq = rho_update_freq
        self.rho_balance = rho_balance

        # Warm-start state (persists across outer SQP iterations)
        self._u_constr:    np.ndarray | None = None   # (m_constr,)
        self._u_bnd:       np.ndarray | None = None   # (n,)
        self._lb_prev:     np.ndarray | None = None   # step lower bounds from previous QP
        self._ub_prev:     np.ndarray | None = None   # step upper bounds from previous QP
        self._lam_max_prev: float            = 0.0    # max |λ| from previous QP solve
        self._rho_at_save:  float            = 0.0    # ρ used when u was last saved

        # Woodbury factors (rebuilt by _woodbury_setup)
        self._sigma_rho: float             = 0.0
        self._W:         np.ndarray | None = None   # n × 2m rank-2 columns
        self._F_inv:     np.ndarray | None = None   # (2m × 2m) inner inverse
        self._H_rho_JcT: np.ndarray | None = None   # n × m_constr
        self._S_inv:     np.ndarray | None = None   # (m_constr × m_constr)⁻¹

        # Cached for solve_soc
        self._lbfgs_ref:     LBFGSMemory | None  = None
        self._J_constr_soc:                        None  = None
        self._JT_soc                              = None
        self._g_soc:         np.ndarray | None    = None
        self._rho_soc:       float                = 1.0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row(J, i: int, n: int) -> np.ndarray:
        """Extract row i of J (dense or sparse) as a dense (n,) array."""
        if sp.issparse(J):
            return np.asarray(J[i, :].todense()).ravel()
        return np.asarray(J[i, :]).ravel()

    # Dense-path threshold: when both Jacobians are dense and the row count
    # is at most this, stack as a plain ndarray.  scipy.sparse construction
    # and per-step matmul/transpose dominate the ADMM loop for small
    # m_constr — see scripts/profile_solver.py --backend woodbury.  For
    # Rosenbrock-chain (m_constr = 2) the sparse path was 30 % of total
    # solve time; the dense path eliminates that almost entirely.
    _DENSE_M_CONSTR_THRESHOLD = 32

    @classmethod
    def _stack_J(cls, J_eq, J_ineq, n: int, m_eq: int, m_ineq: int):
        """
        Stack [J_eq; J_ineq] as an (m_constr, n) matrix.

        * Both inputs dense and m_constr ≤ ``_DENSE_M_CONSTR_THRESHOLD``
          → return a dense ndarray (faster downstream — no scipy.sparse).
        * Otherwise → CSR sparse matrix.
        * m_eq == m_ineq == 0 → None.
        """
        m_constr = m_eq + m_ineq
        if m_constr == 0:
            return None

        eq_dense   = (m_eq   == 0) or not sp.issparse(J_eq)
        ineq_dense = (m_ineq == 0) or not sp.issparse(J_ineq)
        use_dense  = eq_dense and ineq_dense and m_constr <= cls._DENSE_M_CONSTR_THRESHOLD

        if use_dense:
            blocks = []
            if m_eq > 0:
                blocks.append(np.atleast_2d(np.asarray(J_eq, dtype=float)))
            if m_ineq > 0:
                blocks.append(np.atleast_2d(np.asarray(J_ineq, dtype=float)))
            return np.vstack(blocks)

        blocks = []
        if m_eq > 0:
            blocks.append(J_eq.tocsr() if sp.issparse(J_eq)
                          else sp.csr_matrix(np.atleast_2d(J_eq)))
        if m_ineq > 0:
            blocks.append(J_ineq.tocsr() if sp.issparse(J_ineq)
                          else sp.csr_matrix(np.atleast_2d(J_ineq)))
        return sp.vstack(blocks, format='csr')

    # ------------------------------------------------------------------
    # Woodbury setup (called once per QP solve, and on ρ change)
    # ------------------------------------------------------------------

    def _woodbury_setup(
        self,
        lbfgs: LBFGSMemory,
        J_constr,       # sparse CSR or None
        rho: float,
    ) -> None:
        """
        Precompute the two-step Woodbury inverse for (B_k + ρ AᵀA)⁻¹.

        Uses the explicit rank-2 decomposition B_k = σI + W D Wᵀ (from
        ``rank2_factors``) instead of the compact (σ, Ψ, M) form.  This
        avoids inverting G = ΨᵀΨ, which becomes ill-conditioned when the
        stored (s, y) pairs are nearly linearly dependent.

        Stores: _sigma_rho, _W, _F_inv, _H_rho_JcT, _S_inv.
        """
        sigma, W, d = lbfgs.rank2_factors()
        sigma_rho = sigma + rho
        k = W.shape[1]    # = 2m  (0 if no pairs stored)

        # ----------------------------------------------------------------
        # Step 1: F⁻¹  (2m × 2m)  for H_ρ = (B_k + ρI)⁻¹
        #
        # B_k + ρI = (σ+ρ)I + W D Wᵀ
        # Woodbury: H_ρ = (σ+ρ)⁻¹ I − (σ+ρ)⁻² W F⁻¹ Wᵀ
        # where F = D + WᵀW/(σ+ρ)   (D⁻¹ = D since D has ±1 entries)
        # ----------------------------------------------------------------
        if k > 0:
            WtW = W.T @ W                           # 2m × 2m
            F   = np.diag(d) + WtW / sigma_rho      # 2m × 2m
            try:
                F_inv = np.linalg.inv(F)
            except np.linalg.LinAlgError:
                F_inv = np.linalg.pinv(F)
        else:
            F_inv = None

        self._sigma_rho = sigma_rho
        self._W         = W
        self._F_inv     = F_inv

        # ----------------------------------------------------------------
        # Step 2: H_ρ J_constrᵀ  and  S = I + ρ J H_ρ Jᵀ  (m_constr × m_constr)
        # ----------------------------------------------------------------
        if J_constr is not None:
            m_c = J_constr.shape[0]
            # Densify J_constrᵀ as n × m_c (m_c is small)
            if sp.issparse(J_constr):
                JcT = J_constr.T.toarray()          # n × m_c
            else:
                JcT = np.asarray(J_constr).T        # n × m_c
            H_rho_JcT = self._H_rho_batch(JcT)     # n × m_c  (using current _Psi etc.)

            # S = I + ρ J_constr H_rho_JcT  (m_c × m_c)
            JH = np.asarray(J_constr @ H_rho_JcT)  # m_c × m_c
            S  = np.eye(m_c) + rho * JH
            try:
                S_inv = np.linalg.inv(S)
            except np.linalg.LinAlgError:
                S_inv = np.linalg.pinv(S)

            self._H_rho_JcT = H_rho_JcT
            self._S_inv     = S_inv
        else:
            self._H_rho_JcT = None
            self._S_inv     = None

    # ------------------------------------------------------------------
    # Woodbury apply helpers
    # ------------------------------------------------------------------

    def _H_rho_batch(self, V: np.ndarray) -> np.ndarray:
        """
        Apply H_ρ = (B_k + ρI)⁻¹ to each column of V (n × k_cols) in O(n·m·k_cols).

        Also accepts a 1-D vector (treated as a single column, returned as 1-D).
        Uses the precomputed _sigma_rho, _W, _F_inv.

        H_ρ V = V/(σ+ρ) − W (F⁻¹ Wᵀ V) / (σ+ρ)²
        """
        squeeze = V.ndim == 1
        if squeeze:
            V = V.reshape(-1, 1)

        result = V / self._sigma_rho

        W = self._W
        if (W is not None
                and W.shape[1] > 0
                and self._F_inv is not None):
            WtV   = W.T @ V                             # (2m × k_cols)
            alpha = self._F_inv @ WtV                   # (2m × k_cols)
            result = result - (W @ alpha) / self._sigma_rho**2

        return result.ravel() if squeeze else result

    def _woodbury_apply(self, rhs: np.ndarray, J_constr) -> np.ndarray:
        """
        Apply (B_k + ρ AᵀA)⁻¹ rhs using the two-step Woodbury.

            v = H_ρ rhs
            v −= H_ρ_JᵀC · S⁻¹ · (ρ J_constr v)   [Schur correction]
        """
        v = self._H_rho_batch(rhs)

        if (J_constr is not None
                and self._H_rho_JcT is not None
                and self._S_inv is not None):
            Jv       = np.asarray(J_constr @ v).ravel()   # m_constr
            alpha    = self._S_inv @ (self.rho * Jv)       # m_constr
            v        = v - self._H_rho_JcT @ alpha         # n

        return v

    # ------------------------------------------------------------------
    # Public API — same signature as ImplicitLBFGSQP
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
        Solve the QP subproblem via Woodbury-ADMM.

        Returns
        -------
        p         : ndarray (n,)   primal step
        lam_eq    : ndarray (m_eq,)
        lam_ineq  : ndarray (m_ineq,)
        lam_lb    : ndarray (n,)
        lam_ub    : ndarray (n,)
        success   : bool
        """
        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq
        m_constr = m_eq + m_ineq
        lb  = xl - x
        ub  = xu - x

        # Warm-start ρ towards the previous QP's max multiplier magnitude.
        # Optimal ADMM convergence requires ρ ≈ λ* (the KKT multiplier).
        # lam_max_prev is a reliable estimate of λ* when the previous QP solve
        # used a correct Woodbury (guaranteed by self.rho = rho below).
        if self._lam_max_prev > 0:
            rho = float(np.clip(self._lam_max_prev, self.rho_min, self.rho_max))
        else:
            rho = self.rho

        # CRITICAL: update self.rho BEFORE _woodbury_setup so that
        # _woodbury_apply (which reads self.rho for the Schur correction)
        # always uses the same ρ that _S_inv was built with.
        self.rho = rho

        J_constr = self._stack_J(J_eq, J_ineq, n, m_eq, m_ineq)
        # Compute J_constr.T once and reuse — for sparse CSR this avoids
        # rebuilding the CSC view on every ADMM iteration; for dense it's
        # a free zero-copy stride flip.
        J_constr_T = J_constr.T if J_constr is not None else None
        self._woodbury_setup(lbfgs, J_constr, rho)

        # Cache for solve_soc (rho cached after ADMM loop, see below)
        self._lbfgs_ref    = lbfgs
        self._J_constr_soc = J_constr
        self._JT_soc       = J_constr_T
        self._g_soc        = g

        # --- Initialise z, u (warm-start with safety clipping) ---
        # Dual variables u warm-start well: optimal dual changes smoothly across
        # SQP iterations.  When ρ changes between QP calls rescale u to maintain
        # the λ = ρu invariant: u_new = u_old × (ρ_old / ρ_new).
        # Also clip u_bnd to ±max(10,‖g‖)/ρ so the initial ADMM rhs perturbation
        # ρ(z−u) stays comparable to the objective gradient.
        g_norm = float(np.linalg.norm(g))
        u_bnd_clip = max(10.0, g_norm) / rho

        if (self._u_constr is not None
                and self._u_constr.shape == (m_constr,)):
            u_constr = self._u_constr.copy()
            if self._rho_at_save > 0 and self._rho_at_save != rho:
                u_constr *= self._rho_at_save / rho
        else:
            u_constr = np.zeros(m_constr)

        if self._u_bnd is not None and self._u_bnd.shape == (n,):
            u_bnd_raw = self._u_bnd.copy()
            if self._rho_at_save > 0 and self._rho_at_save != rho:
                u_bnd_raw *= self._rho_at_save / rho
            u_bnd = np.clip(u_bnd_raw, -u_bnd_clip, u_bnd_clip)
        else:
            u_bnd = np.zeros(n)

        # Cold-start u_bnd[i] when a bound TRANSITIONS from inactive to active.
        # A warm-started dual is only problematic when x was interior at the
        # previous QP but has now hit a bound (ub or lb ≈ 0).  If the bound was
        # already active at the previous QP, the warm-start is well-calibrated
        # and should be kept.  Resetting newly-active components to 0 allows
        # ADMM to build the dual from a correct feasible direction.
        tol_cold = 10.0 * self.tol_abs * np.sqrt(n + max(m_constr, 1))
        ub_fin = np.isfinite(ub)
        lb_fin = np.isfinite(lb)
        if self._ub_prev is not None and self._ub_prev.shape == (n,):
            # Upper bound just became active: was far, now ≈0
            newly_ub = ub_fin & (ub < tol_cold) & (self._ub_prev >= tol_cold)
            u_bnd = np.where(newly_ub, 0.0, u_bnd)
        if self._lb_prev is not None and self._lb_prev.shape == (n,):
            # Lower bound just became active: was far, now ≈0
            newly_lb = lb_fin & (-lb < tol_cold) & (-self._lb_prev >= tol_cold)
            u_bnd = np.where(newly_lb, 0.0, u_bnd)

        # z: always re-initialize to a feasible point for the current QP
        z_constr = np.zeros(m_constr)
        if m_eq > 0:
            z_constr[:m_eq] = -c_eq
        z_bnd = np.clip(np.zeros(n), lb, ub)

        Jp = np.empty(0)   # J_constr @ p  (recomputed each iteration)

        rho_init = rho   # never halve below the initial warm-start value

        for k_iter in range(self.max_iter):
            # ---- p-update ----
            rhs = -g.copy()
            if m_constr > 0 and J_constr is not None:
                rhs += rho * np.asarray(J_constr_T @ (z_constr - u_constr)).ravel()
            rhs += rho * (z_bnd - u_bnd)

            p = self._woodbury_apply(rhs, J_constr)

            # ---- z-update (projection onto K) ----
            z_constr_prev = z_constr.copy()
            z_bnd_prev    = z_bnd.copy()

            if m_constr > 0 and J_constr is not None:
                Jp    = np.asarray(J_constr @ p).ravel()   # m_constr
                z_new = Jp + u_constr
                if m_eq   > 0:
                    z_new[:m_eq] = -c_eq                            # equality: fixed
                if m_ineq > 0:
                    z_new[m_eq:] = np.minimum(z_new[m_eq:], -c_ineq)  # ineq: half-space
                z_constr = z_new

            z_bnd = np.clip(p + u_bnd, lb, ub)

            # ---- u-update ----
            if m_constr > 0:
                u_constr += Jp - z_constr
            u_bnd += p - z_bnd

            # ---- Convergence (OSQP-style primal + dual norms) ----
            r_p_bnd  = p - z_bnd                                     # n
            r_p_constr = Jp - z_constr if m_constr > 0 else np.empty(0)
            r_prim_norm = float(np.sqrt(
                np.dot(r_p_constr, r_p_constr) + np.dot(r_p_bnd, r_p_bnd)
            ))

            dz_c = z_constr - z_constr_prev   # m_constr
            dz_b = z_bnd    - z_bnd_prev      # n
            if m_constr > 0 and J_constr is not None:
                r_dual = rho * (np.asarray(J_constr_T @ dz_c).ravel() + dz_b)
            else:
                r_dual = rho * dz_b
            r_dual_norm = float(np.linalg.norm(r_dual))

            Ap_sq = (float(np.dot(Jp, Jp)) if m_constr > 0 else 0.0) + float(np.dot(p, p))
            z_sq  = float(np.dot(z_constr, z_constr)) + float(np.dot(z_bnd, z_bnd))
            Au_sq = float(np.dot(u_constr, u_constr)) + float(np.dot(u_bnd, u_bnd))

            eps_p = (self.tol_abs * np.sqrt(n + m_constr)
                     + self.tol_rel * max(np.sqrt(Ap_sq), np.sqrt(z_sq)))
            eps_d = (self.tol_abs * np.sqrt(n)
                     + self.tol_rel * rho * np.sqrt(Au_sq))

            if r_prim_norm <= eps_p and r_dual_norm <= eps_d:
                break

            # ---- ρ adaptation (OSQP rule) ----
            if (k_iter + 1) % self.rho_update_freq == 0:
                if r_prim_norm > self.rho_balance * r_dual_norm:
                    new_rho = min(rho * 2.0, self.rho_max)
                elif r_dual_norm > self.rho_balance * r_prim_norm:
                    new_rho = max(rho / 2.0, self.rho_min, rho_init)
                else:
                    new_rho = rho
                if new_rho != rho:
                    # Rescale u to maintain λ = ρu invariant: u_new = u_old*(ρ_old/ρ_new)
                    scale = rho / new_rho
                    u_constr *= scale
                    u_bnd    *= scale
                    rho = new_rho
                    self.rho = rho
                    self._woodbury_setup(lbfgs, J_constr, rho)

        # Freeze the final ρ for solve_soc (must match cached _S_inv)
        self._rho_soc = rho

        # Clip final p to bounds (safety)
        p = np.clip(p, lb, ub)

        # Snap p to active bounds within primal tolerance.
        # ADMM converges to p[i] ≈ ub[i] (not exactly) for active upper bounds.
        # A 7e-7 gap causes x[i] = xl[i] + 0.5 - 7e-7 instead of xl[i] + 0.5,
        # leading to ub[i] = 7e-7 at the next SQP step and warm-start failure.
        tol_snap = 10.0 * self.tol_abs * np.sqrt(n + m_constr)
        lb_fin   = np.isfinite(lb)
        ub_fin   = np.isfinite(ub)
        p = np.where(ub_fin & (ub - p < tol_snap), ub, p)
        p = np.where(lb_fin & (p - lb < tol_snap), lb, p)

        # Save warm-start state (lb/ub saved for bound-transition detection)
        self._u_constr    = u_constr.copy()
        self._u_bnd       = u_bnd.copy()
        self._lb_prev     = lb.copy()
        self._ub_prev     = ub.copy()
        self._rho_at_save = rho   # ρ active when u was saved (for rescaling on next call)

        # Extract NLP multipliers from scaled dual λ = ρ u
        lam_eq   = rho * u_constr[:m_eq]                  if m_eq   > 0 else np.zeros(0)
        lam_ineq = np.maximum(0.0, rho * u_constr[m_eq:]) if m_ineq > 0 else np.zeros(0)
        y_b      = rho * u_bnd
        lam_lb   = np.maximum(0.0, -y_b)
        lam_ub   = np.maximum(0.0,  y_b)

        # Polish: replace ADMM's O(ρ·tol) multipliers with KKT-quality
        # multipliers from a single Schur solve on the inferred active set.
        p, lam_eq, lam_ineq, lam_lb, lam_ub = self._polish(
            lbfgs, g, c_eq, J_eq, c_ineq, J_ineq, lb, ub,
            p, lam_eq, lam_ineq, lam_lb, lam_ub,
        )
        p = np.clip(p, lb, ub)

        # Save max |λ| for ρ warm-start at next QP (capped at rho_max to
        # prevent a single bad QP from permanently inflating ρ_init)
        all_lam = np.concatenate([np.abs(lam_eq), lam_ineq, lam_lb, lam_ub])
        raw_max = float(np.max(all_lam)) if len(all_lam) > 0 else 0.0
        self._lam_max_prev = float(np.clip(raw_max, 0.0, self.rho_max))

        return p, lam_eq, lam_ineq, lam_lb, lam_ub, True

    def _polish(
        self,
        lbfgs: LBFGSMemory,
        g: np.ndarray,
        c_eq: np.ndarray,
        J_eq,
        c_ineq: np.ndarray,
        J_ineq,
        lb: np.ndarray,
        ub: np.ndarray,
        p: np.ndarray,
        lam_eq: np.ndarray,
        lam_ineq: np.ndarray,
        lam_lb: np.ndarray,
        lam_ub: np.ndarray,
        tol_active: float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Polish the ADMM solution by solving the equality-constrained KKT
        system on the active set inferred from the ADMM dual variables.

        ADMM converges its multipliers to ~ρ·tol error.  When |λ*| is large
        (e.g. HS15: λ_ub ≈ 1750 at the optimum) this contaminates the L-BFGS
        ``y`` update used by the next outer SQP iteration.  A single Schur
        solve with the active set fixes the multipliers to KKT-quality at
        O(m_act · n · m) cost — the same per-step cost as ImplicitLBFGSQP.

        Falls back to the input ADMM solution if the polish disagrees with
        the active-set guess (negative multiplier on a supposed-active row)
        or the Schur matrix is singular.
        """
        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq

        active_ineq = [j for j in range(m_ineq) if lam_ineq[j] > tol_active]
        active_lb   = [k for k in range(n)      if lam_lb[k]   > tol_active]
        active_ub   = [k for k in range(n)      if lam_ub[k]   > tol_active]

        rows: list[np.ndarray] = []
        bs:   list[float]      = []

        for i in range(m_eq):
            rows.append(self._row(J_eq, i, n))
            bs.append(float(-c_eq[i]))
        for j in active_ineq:
            rows.append(self._row(J_ineq, j, n))
            bs.append(float(-c_ineq[j]))
        e = np.zeros(n)
        for k in active_lb:
            e[k] = -1.0
            rows.append(e.copy())
            bs.append(float(-lb[k]))
            e[k] = 0.0
        for k in active_ub:
            e[k] = 1.0
            rows.append(e.copy())
            bs.append(float(ub[k]))
            e[k] = 0.0

        if not rows:
            return -lbfgs._matvec(g), lam_eq, lam_ineq, lam_lb, lam_ub

        A = np.asarray(rows, dtype=float)
        b = np.asarray(bs,   dtype=float)

        # Schur: S λ = -(b + A H g);  p = -H(g + Aᵀ λ)
        # Batched matvec sweeps the L-BFGS two-loop once over all rows of A.
        Hg   = lbfgs._matvec(g)
        HA_T = lbfgs._matvec_batch(A.T)
        S    = A @ HA_T
        rhs  = -(b + A @ Hg)

        try:
            lam_p = np.linalg.solve(S, rhs)
        except np.linalg.LinAlgError:
            try:
                lam_p, *_ = np.linalg.lstsq(S, rhs, rcond=None)
            except np.linalg.LinAlgError:
                return p, lam_eq, lam_ineq, lam_lb, lam_ub

        # Sanity: non-equality multipliers must be ≥ 0 at a true KKT point.
        idx = m_eq
        for _ in active_ineq:
            if lam_p[idx] < -tol_active:
                return p, lam_eq, lam_ineq, lam_lb, lam_ub
            idx += 1
        for _ in active_lb:
            if lam_p[idx] < -tol_active:
                return p, lam_eq, lam_ineq, lam_lb, lam_ub
            idx += 1
        for _ in active_ub:
            if lam_p[idx] < -tol_active:
                return p, lam_eq, lam_ineq, lam_lb, lam_ub
            idx += 1

        p_new = -Hg - HA_T @ lam_p

        lam_eq_new   = np.zeros(m_eq)
        lam_ineq_new = np.zeros(m_ineq)
        lam_lb_new   = np.zeros(n)
        lam_ub_new   = np.zeros(n)

        idx = 0
        for i in range(m_eq):
            lam_eq_new[i] = lam_p[idx]; idx += 1
        for j in active_ineq:
            lam_ineq_new[j] = max(0.0, lam_p[idx]); idx += 1
        for k in active_lb:
            lam_lb_new[k] = max(0.0, lam_p[idx]); idx += 1
        for k in active_ub:
            lam_ub_new[k] = max(0.0, lam_p[idx]); idx += 1

        return p_new, lam_eq_new, lam_ineq_new, lam_lb_new, lam_ub_new

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

        Re-uses the cached Woodbury factors from the last ``solve()`` call
        (same Hessian and Jacobians).  Runs a short ADMM warm-started from
        the current u to find the SOC direction d at the trial point x+p.
        """
        if self._lbfgs_ref is None:
            return np.zeros(self.n), False

        n, m_eq, m_ineq = self.n, self.m_eq, self.m_ineq
        m_constr = m_eq + m_ineq
        lb_soc   = xl - (x + p)
        ub_soc   = xu - (x + p)
        g          = self._g_soc
        J_constr   = self._J_constr_soc
        J_constr_T = self._JT_soc
        rho        = self._rho_soc

        # Initialise from warm-start dual, zero primal
        z_constr = np.zeros(m_constr)
        if m_eq > 0:
            z_constr[:m_eq] = -c_eq_trial
        z_bnd    = np.clip(np.zeros(n), lb_soc, ub_soc)
        u_constr = (np.zeros(m_constr) if self._u_constr is None
                    else self._u_constr.copy())
        g_norm_soc = float(np.linalg.norm(g))
        u_bnd_clip = max(10.0, g_norm_soc) / rho
        u_bnd = (np.zeros(n) if self._u_bnd is None
                 else np.clip(self._u_bnd, -u_bnd_clip, u_bnd_clip))

        d = np.zeros(n)
        Jd = np.empty(0)
        soc_iters = min(50, self.max_iter)

        for _ in range(soc_iters):
            rhs = -g.copy()
            if m_constr > 0 and J_constr is not None:
                rhs += rho * np.asarray(J_constr_T @ (z_constr - u_constr)).ravel()
            rhs += rho * (z_bnd - u_bnd)

            d = self._woodbury_apply(rhs, J_constr)

            if m_constr > 0 and J_constr is not None:
                Jd    = np.asarray(J_constr @ d).ravel()
                z_new = Jd + u_constr
                if m_eq   > 0:
                    z_new[:m_eq] = -c_eq_trial
                if m_ineq > 0:
                    z_new[m_eq:] = np.minimum(z_new[m_eq:], -c_ineq_trial)
                z_constr  = z_new
                u_constr += Jd - z_constr

            z_bnd  = np.clip(d + u_bnd, lb_soc, ub_soc)
            u_bnd += d - z_bnd

        d = np.clip(d, lb_soc, ub_soc)
        return d, True
