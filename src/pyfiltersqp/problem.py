"""NLP problem interface."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

_INF = np.inf


@dataclass
class NLPProblem:
    """
    Standard NLP for pyfiltersqp::

        min   f(x)
        s.t.  c_eq(x)   = 0          (m_eq rows)
              c_ineq(x) ≤ 0          (m_ineq rows)
              xl ≤ x ≤ xu

    All callables must accept a 1-D ndarray of shape (n,).
    Jacobians are REQUIRED — no finite-difference fallback.

    Parameters
    ----------
    n : int
        Number of decision variables.
    m_eq : int
        Number of equality constraints.
    m_ineq : int
        Number of inequality constraints.
    xl, xu : ndarray, shape (n,)
        Variable bounds. Use ±np.inf for unbounded components.
    objective : callable  f(x) -> float
    gradient  : callable  ∇f(x) -> ndarray (n,)
    eq_constraints  : callable  c_eq(x)  -> ndarray (m_eq,)   or None if m_eq=0
    eq_jacobian     : callable  J_eq(x)  -> ndarray (m_eq, n) or None if m_eq=0
    ineq_constraints: callable  c_ineq(x)-> ndarray (m_ineq,) or None if m_ineq=0
    ineq_jacobian   : callable  J_ineq(x)-> ndarray (m_ineq,n)or None if m_ineq=0
    x0 : ndarray, shape (n,)
        Reference point used for shape validation at construction time.
    """

    n: int
    m_eq: int
    m_ineq: int
    xl: np.ndarray
    xu: np.ndarray
    objective: Callable
    gradient: Callable
    eq_constraints: Callable | None = None
    eq_jacobian: Callable | None = None   # may return dense ndarray or scipy sparse (CSR/CSC)
    ineq_constraints: Callable | None = None
    ineq_jacobian: Callable | None = None  # may return dense ndarray or scipy sparse (CSR/CSC)
    x0: np.ndarray | None = None

    def __post_init__(self):
        self.xl = np.asarray(self.xl, dtype=float)
        self.xu = np.asarray(self.xu, dtype=float)

        if self.xl.shape != (self.n,):
            raise ValueError(f"xl must have shape ({self.n},), got {self.xl.shape}")
        if self.xu.shape != (self.n,):
            raise ValueError(f"xu must have shape ({self.n},), got {self.xu.shape}")

        if self.m_eq > 0 and (self.eq_constraints is None or self.eq_jacobian is None):
            raise ValueError("m_eq > 0 but eq_constraints or eq_jacobian is None")
        if self.m_ineq > 0 and (self.ineq_constraints is None or self.ineq_jacobian is None):
            raise ValueError("m_ineq > 0 but ineq_constraints or ineq_jacobian is None")

        if self.x0 is not None:
            self._validate_shapes(np.asarray(self.x0, dtype=float))

    def _validate_shapes(self, x: np.ndarray) -> None:
        """Check all callable outputs have the expected shapes at point x."""
        f = self.objective(x)
        if not np.isscalar(f):
            raise ValueError(f"objective must return a scalar, got shape {np.shape(f)}")

        g = np.asarray(self.gradient(x))
        if g.shape != (self.n,):
            raise ValueError(f"gradient must return shape ({self.n},), got {g.shape}")

        if self.m_eq > 0:
            c = np.asarray(self.eq_constraints(x))
            if c.shape != (self.m_eq,):
                raise ValueError(f"eq_constraints must return shape ({self.m_eq},), got {c.shape}")
            J = self.eq_jacobian(x)   # may be dense or scipy sparse — both have .shape
            if J.shape != (self.m_eq, self.n):
                raise ValueError(f"eq_jacobian must return shape ({self.m_eq},{self.n}), got {J.shape}")

        if self.m_ineq > 0:
            c = np.asarray(self.ineq_constraints(x))
            if c.shape != (self.m_ineq,):
                raise ValueError(f"ineq_constraints must return shape ({self.m_ineq},), got {c.shape}")
            J = self.ineq_jacobian(x)   # may be dense or scipy sparse — both have .shape
            if J.shape != (self.m_ineq, self.n):
                raise ValueError(f"ineq_jacobian must return shape ({self.m_ineq},{self.n}), got {J.shape}")
