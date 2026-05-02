"""l₁ merit function and Armijo backtracking line search."""
from __future__ import annotations

import numpy as np


def l1_merit(
    f: float,
    c_eq: np.ndarray,
    c_ineq: np.ndarray,
    mu: float,
) -> float:
    """
    l₁ penalty merit function::

        φ_μ(x) = f(x) + μ * ( ||c_eq(x)||₁ + ||max(0, c_ineq(x))||₁ )

    Parameters
    ----------
    f : float
        Objective value at x.
    c_eq : ndarray
        Equality constraint values (should be zero at feasible points).
    c_ineq : ndarray
        Inequality constraint values (should be ≤ 0 at feasible points).
    mu : float
        Penalty parameter (must be > 0).
    """
    eq_viol   = np.sum(np.abs(c_eq))       if len(c_eq)   > 0 else 0.0
    ineq_viol = np.sum(np.maximum(0, c_ineq)) if len(c_ineq) > 0 else 0.0
    return f + mu * (eq_viol + ineq_viol)


def directional_deriv(
    g: np.ndarray,
    p: np.ndarray,
    c_eq: np.ndarray,
    c_ineq: np.ndarray,
    mu: float,
) -> float:
    """
    Directional derivative of φ_μ at x along p::

        D_p φ_μ(x) = gᵀ p - μ * ( ||c_eq||₁ + ||max(0, c_ineq)||₁ )

    This is a valid upper bound on the true directional derivative of the
    non-smooth l₁ merit (the true derivative may be smaller, making the
    Armijo condition easier to satisfy).
    """
    eq_viol   = np.sum(np.abs(c_eq))          if len(c_eq)   > 0 else 0.0
    ineq_viol = np.sum(np.maximum(0, c_ineq)) if len(c_ineq) > 0 else 0.0
    return float(g @ p) - mu * (eq_viol + ineq_viol)


def update_penalty(
    g: np.ndarray,
    p: np.ndarray,
    B,
    c_eq: np.ndarray,
    c_ineq: np.ndarray,
    mu: float,
    sigma: float = 0.01,
) -> float:
    """
    Increase the penalty parameter μ if needed so that p is a descent direction.

    The condition is::

        μ ≥ (gᵀp + (1/2) pᵀBp) / ((1 - σ) * ||violations||₁)

    μ only ever increases (never decreases).

    Parameters
    ----------
    B : ndarray or callable
        Dense Hessian matrix OR a callable ``B(v)`` that returns ``B @ v``.
        The callable form avoids materialising the dense matrix.

    Returns the (possibly updated) μ.
    """
    eq_viol   = np.sum(np.abs(c_eq))          if len(c_eq)   > 0 else 0.0
    ineq_viol = np.sum(np.maximum(0, c_ineq)) if len(c_ineq) > 0 else 0.0
    total_viol = eq_viol + ineq_viol

    if total_viol < 1e-14:
        return mu   # feasible point — penalty irrelevant

    Bp        = B(p) if callable(B) else B @ p
    numerator = float(g @ p) + 0.5 * float(p @ Bp)
    threshold = numerator / ((1.0 - sigma) * total_viol)

    if mu < threshold:
        mu = threshold * 1.1   # inflate by 10 %
    return mu


def armijo_backtrack(
    x: np.ndarray,
    p: np.ndarray,
    f0: float,
    c_eq0: np.ndarray,
    c_ineq0: np.ndarray,
    D_phi: float,
    mu: float,
    obj_fn,
    eq_fn,
    ineq_fn,
    rho: float = 0.5,
    eta: float = 1e-4,
    max_iter: int = 30,
    alpha_min: float = 1e-10,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Backtracking Armijo line search on the l₁ merit function.

    Finds α ∈ (0, 1] such that::

        φ_μ(x + α*p) ≤ φ_μ(x) + η * α * D_φ

    Parameters
    ----------
    x, p : ndarray
        Current point and search direction.
    f0 : float
        f(x).
    c_eq0, c_ineq0 : ndarray
        Constraint values at x.
    D_phi : float
        Directional derivative of φ_μ at x along p (from ``directional_deriv``).
    mu : float
        Current penalty parameter.
    obj_fn, eq_fn, ineq_fn : callable
        Problem callables.
    rho : float
        Backtracking factor (default 0.5).
    eta : float
        Armijo sufficient decrease constant (default 1e-4).
    max_iter : int
        Maximum backtracking steps.
    alpha_min : float
        Minimum step size before declaring failure.

    Returns
    -------
    alpha : float
        Accepted step size (may be alpha_min if line search failed).
    phi_new : float
        Merit value at the accepted point.
    c_eq_new, c_ineq_new : ndarray
        Constraint values at the accepted point.
    """
    phi0 = l1_merit(f0, c_eq0, c_ineq0, mu)
    alpha = 1.0

    for _ in range(max_iter):
        x_new = x + alpha * p
        f_new     = float(obj_fn(x_new))
        c_eq_new  = np.asarray(eq_fn(x_new))   if eq_fn   is not None else np.empty(0)
        c_ineq_new = np.asarray(ineq_fn(x_new)) if ineq_fn is not None else np.empty(0)
        phi_new = l1_merit(f_new, c_eq_new, c_ineq_new, mu)

        if phi_new <= phi0 + eta * alpha * D_phi:
            return alpha, phi_new, c_eq_new, c_ineq_new

        alpha *= rho
        if alpha < alpha_min:
            break

    # Line search failed — return smallest tried step
    return alpha_min, phi_new, c_eq_new, c_ineq_new


def filter_backtrack(
    x: np.ndarray,
    p: np.ndarray,
    f_curr: float,
    h_curr: float,
    c_eq_curr: np.ndarray,
    c_ineq_curr: np.ndarray,
    filt,
    obj_fn,
    eq_fn,
    ineq_fn,
    rho: float = 0.5,
    max_iter: int = 30,
    alpha_min: float = 1e-10,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """
    Filter line search: decrease α until the trial point (f, h) is
    filter-acceptable AND has sufficient decrease from (f_curr, h_curr).

    Parameters
    ----------
    x, p : ndarray
        Current point and search direction.
    f_curr : float
        Objective at x.
    h_curr : float
        Constraint violation at x (from ``constraint_violation``).
    c_eq_curr, c_ineq_curr : ndarray
        Constraint values at x (used only as fallback return values).
    filt : Filter
        The active filter (must already contain the current iterate).
    obj_fn, eq_fn, ineq_fn : callable or None
        Problem callables.
    rho : float
        Backtracking factor.
    max_iter : int
        Maximum backtracking steps.
    alpha_min : float
        Minimum step size before declaring failure.

    Returns
    -------
    alpha : float
        Accepted step size.  0.0 signals failure (no acceptable step found).
    f_new : float
        Objective at accepted point (at the last tried α on failure).
    c_eq_new, c_ineq_new : ndarray
        Constraint values at accepted point.
    """
    from ._filter import constraint_violation  # avoid circular import at module level

    alpha = 1.0
    f_new     = f_curr
    c_eq_new  = c_eq_curr
    c_ineq_new = c_ineq_curr

    for _ in range(max_iter):
        x_new      = x + alpha * p
        f_new      = float(obj_fn(x_new))
        c_eq_new   = np.asarray(eq_fn(x_new))   if eq_fn   is not None else np.empty(0)
        c_ineq_new = np.asarray(ineq_fn(x_new)) if ineq_fn is not None else np.empty(0)
        h_new      = constraint_violation(c_eq_new, c_ineq_new)

        if (filt.is_acceptable(f_new, h_new)
                and filt.has_sufficient_decrease(f_curr, h_curr, f_new, h_new)):
            return alpha, f_new, c_eq_new, c_ineq_new

        alpha *= rho
        if alpha < alpha_min:
            break

    # No acceptable step found
    return 0.0, f_new, c_eq_new, c_ineq_new
