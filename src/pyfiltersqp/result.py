"""Result dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class IterationInfo:
    """Per-iteration diagnostics."""
    iteration:   int
    obj:         float
    primal_eq:   float   # ||c_eq||_inf
    primal_ineq: float   # ||max(0, c_ineq)||_inf
    dual:        float   # ||∇f - Jᵀλ||_inf
    compl:       float   # ||λ_ineq ⊙ c_ineq||_inf
    step_norm:   float   # ||α * p||
    step_size:   float   # α (line search result)
    qp_iters:    int     # OSQP iterations for this subproblem


@dataclass
class SQPResult:
    """
    Output of :func:`solve` / :meth:`SQPSolver.solve`.

    Attributes
    ----------
    x : ndarray
        Primal solution.
    obj : float
        Objective value f(x).
    lam_eq : ndarray, shape (m_eq,)
        Equality constraint multipliers.
    lam_ineq : ndarray, shape (m_ineq,)
        Inequality constraint multipliers (≥ 0 at solution).
    lam_lb : ndarray, shape (n,)
        Lower bound multipliers (≥ 0 at solution).
    lam_ub : ndarray, shape (n,)
        Upper bound multipliers (≥ 0 at solution).
    status : int
        0 = KKT conditions satisfied
        1 = maximum iterations reached
        2 = line search failed (step too small)
        3 = QP subproblem failed
    message : str
    success : bool
        True when status == 0.
    iterations : int
        Number of outer SQP iterations performed.
    history : list[IterationInfo]
        Per-iteration diagnostics (only populated when verbose=True).
    """

    x:         np.ndarray
    obj:       float
    lam_eq:    np.ndarray
    lam_ineq:  np.ndarray
    lam_lb:    np.ndarray
    lam_ub:    np.ndarray
    status:    int
    message:   str
    success:   bool
    iterations: int
    history:   list[IterationInfo] = field(default_factory=list)

    # Status code → message mapping
    _MESSAGES: dict = field(default=None, init=False, repr=False)

    def __post_init__(self):
        self._MESSAGES = {
            0: "Optimal solution found (KKT conditions satisfied)",
            1: "Maximum iterations reached",
            2: "Line search failed: step size below minimum",
            3: "QP subproblem failed",
        }
        if not self.message:
            self.message = self._MESSAGES.get(self.status, f"Unknown status {self.status}")
