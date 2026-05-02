"""KKT residual computation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KKTResiduals:
    primal_eq:   float   # ||c_eq||_inf
    primal_ineq: float   # ||max(0, c_ineq)||_inf
    dual:        float   # ||g - J_eqᵀ lam_eq - J_ineqᵀ lam_ineq||_inf
    compl:       float   # ||lam_ineq ⊙ c_ineq||_inf

    def satisfied(
        self,
        tol_feas: float = 1e-6,
        tol_opt:  float = 1e-6,
        tol_comp: float = 1e-6,
    ) -> bool:
        return (
            self.primal_eq   <= tol_feas
            and self.primal_ineq <= tol_feas
            and self.dual        <= tol_opt
            and self.compl       <= tol_comp
        )


def kkt_residuals(
    g: np.ndarray,
    c_eq: np.ndarray,
    c_ineq: np.ndarray,
    J_eq: np.ndarray | None,
    J_ineq: np.ndarray | None,
    lam_eq: np.ndarray,
    lam_ineq: np.ndarray,
    lam_lb: np.ndarray | None = None,
    lam_ub: np.ndarray | None = None,
) -> KKTResiduals:
    """
    Compute KKT residuals.

    OSQP uses the PLUS convention: the Lagrangian is
    L = f + λ_eq·c_eq + λ_ineq·c_ineq − λ_lb·(x−xl) + λ_ub·(x−xu),
    giving the stationarity condition::

        g + J_eq(x)ᵀ λ_eq + J_ineq(x)ᵀ λ_ineq − λ_lb + λ_ub = 0

    Parameters
    ----------
    g : ndarray, shape (n,)
    c_eq : ndarray, shape (m_eq,)
    c_ineq : ndarray, shape (m_ineq,)
    J_eq : ndarray, shape (m_eq, n)  or None
    J_ineq : ndarray, shape (m_ineq, n)  or None
    lam_eq : ndarray, shape (m_eq,)
    lam_ineq : ndarray, shape (m_ineq,)
    lam_lb : ndarray, shape (n,), optional   lower-bound multipliers
    lam_ub : ndarray, shape (n,), optional   upper-bound multipliers
    """
    primal_eq   = float(np.max(np.abs(c_eq)))            if len(c_eq)   > 0 else 0.0
    primal_ineq = float(np.max(np.maximum(0.0, c_ineq))) if len(c_ineq) > 0 else 0.0

    dual_vec = g.copy()
    if J_eq   is not None and len(lam_eq)   > 0:
        dual_vec += J_eq.T   @ lam_eq
    if J_ineq is not None and len(lam_ineq) > 0:
        dual_vec += J_ineq.T @ lam_ineq
    if lam_lb is not None:
        dual_vec -= lam_lb
    if lam_ub is not None:
        dual_vec += lam_ub
    dual = float(np.max(np.abs(dual_vec)))

    compl = float(np.max(np.abs(lam_ineq * c_ineq))) if len(c_ineq) > 0 else 0.0

    return KKTResiduals(primal_eq, primal_ineq, dual, compl)
