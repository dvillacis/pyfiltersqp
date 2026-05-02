"""
Scalable NLP test problems for solver validation at n = 10 ... 10 000.

All problems are pure NumPy with exact gradients/Jacobians and documented
optima.  Most come from the More-Garbow-Hillstrom (1981) test set; the
constrained ones are simple parameterized constructions with closed-form
solutions for sanity-checking the SQP loop at scale.

Each builder returns ``(NLPProblem, x0, f_opt)``.

Builders
--------
* broyden_tridiagonal(n)        — unconstrained, MGH #30,  f* = 0
* discrete_bvp(n)               — unconstrained, MGH #28,  f* = 0
* extended_powell(n)            — unconstrained, MGH #13,  f* = 0  (singular Hessian)
* variably_dimensioned(n)       — unconstrained, MGH #25,  f* = 0
* eq_constrained_quadratic(n,m) — random PSD QP with m linear eq, f* known
* ineq_constrained_quadratic(n,m)— same with m linear ineq active at solution
* rosenbrock_chain(n)           — reuses scripts/profile_solver.make_rosenbrock_nlp
"""
from __future__ import annotations

import os
import importlib.util
from pathlib import Path

import numpy as np

from pyfiltersqp import NLPProblem


# ---------------------------------------------------------------------------
# Reuse the Rosenbrock-chain NLP defined in scripts/profile_solver.py
# ---------------------------------------------------------------------------

def _load_profile_solver():
    here = Path(__file__).resolve().parent.parent / "scripts" / "profile_solver.py"
    spec = importlib.util.spec_from_file_location("_profile_solver_for_tests", here)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rosenbrock_chain(n: int):
    """Eq + ineq + bounds, n must be even.  Defined in profile_solver.py."""
    mod = _load_profile_solver()
    prob, x0 = mod.make_rosenbrock_nlp(n)
    return prob, x0, 0.0   # objective is sum-of-squares-like; not exactly 0


# ---------------------------------------------------------------------------
# Unconstrained large-scale (More-Garbow-Hillstrom)
# ---------------------------------------------------------------------------

def broyden_tridiagonal(n: int):
    """
    MGH #30 — Broyden tridiagonal function (unconstrained).

        F_i(x) = (3 - 2 x_i) x_i - x_{i-1} - 2 x_{i+1} + 1
        with x_0 = x_{n+1} = 0,  i = 1..n

        f(x)   = sum_i F_i(x)^2

    f* = 0 (residuals can be driven to zero).  x0 = (-1, ..., -1).
    """
    x0 = -np.ones(n)

    def F(x):
        f = (3.0 - 2.0 * x) * x + 1.0
        f[1:]  -= x[:-1]
        f[:-1] -= 2.0 * x[1:]
        return f

    def obj(x):
        return float(np.dot(F(x), F(x)))

    def grad(x):
        Fx = F(x)
        diag = 3.0 - 4.0 * x
        g = 2.0 * diag * Fx
        g[:-1] += -2.0 * Fx[1:]
        g[1:]  += -4.0 * Fx[:-1]
        return g

    prob = NLPProblem(
        n=n, m_eq=0, m_ineq=0,
        xl=np.full(n, -np.inf), xu=np.full(n, np.inf),
        objective=obj, gradient=grad,
        x0=x0,
    )
    return prob, x0, 0.0


def discrete_bvp(n: int):
    """
    MGH #28 — Discrete boundary-value problem.

        F_i(x) = 2 x_i - x_{i-1} - x_{i+1} + (h²/2) (x_i + t_i + 1)³
        h = 1/(n+1),  t_i = i·h,  x_0 = x_{n+1} = 0

        f(x) = sum F_i^2

    f* = 0.  Recommended x0_i = t_i (t_i - 1).
    """
    h = 1.0 / (n + 1)
    t = np.arange(1, n + 1) * h
    x0 = t * (t - 1.0)

    def F(x):
        f = 2.0 * x + 0.5 * h * h * (x + t + 1.0) ** 3
        f[1:]  -= x[:-1]
        f[:-1] -= x[1:]
        return f

    def obj(x):
        Fx = F(x)
        return float(np.dot(Fx, Fx))

    def grad(x):
        Fx = F(x)
        diag = 2.0 + 1.5 * h * h * (x + t + 1.0) ** 2
        g = 2.0 * diag * Fx
        g[:-1] += 2.0 * (-1.0) * Fx[1:]
        g[1:]  += 2.0 * (-1.0) * Fx[:-1]
        return g

    prob = NLPProblem(
        n=n, m_eq=0, m_ineq=0,
        xl=np.full(n, -np.inf), xu=np.full(n, np.inf),
        objective=obj, gradient=grad,
        x0=x0,
    )
    return prob, x0, 0.0


def extended_powell(n: int):
    """
    MGH #13 — Extended Powell singular function.

    Requires n divisible by 4.  Per block of 4 variables (x_a, x_b, x_c, x_d):

        F_{4k+1} = x_a + 10 x_b
        F_{4k+2} = √5 (x_c - x_d)
        F_{4k+3} = (x_b - 2 x_c)²
        F_{4k+4} = √10 (x_a - x_d)²

    f* = 0.  Block initial point: (3, -1, 0, 1).  Hessian is singular at minimum.
    """
    if n % 4:
        raise ValueError("extended_powell requires n divisible by 4")
    k = n // 4
    x0 = np.tile([3.0, -1.0, 0.0, 1.0], k)
    s5  = np.sqrt(5.0)
    s10 = np.sqrt(10.0)

    def obj(x):
        a = x[0::4]; b = x[1::4]; c = x[2::4]; d = x[3::4]
        f1 = a + 10.0 * b
        f2 = s5  * (c - d)
        f3 = (b - 2.0 * c) ** 2
        f4 = s10 * (a - d) ** 2
        return float(np.sum(f1*f1) + np.sum(f2*f2) + np.sum(f3*f3) + np.sum(f4*f4))

    def grad(x):
        a = x[0::4]; b = x[1::4]; c = x[2::4]; d = x[3::4]
        f1 = a + 10.0 * b
        f2 = s5  * (c - d)
        bm2c = b - 2.0 * c
        amd  = a - d
        f3 = bm2c ** 2
        f4 = s10 * amd ** 2

        g = np.zeros(n)
        # ∂f/∂a: 2 f1 (1) + 2 f4 · (s10 · 2 · amd)
        g[0::4] = 2.0 * f1 + 4.0 * s10 * amd * f4
        # ∂f/∂b: 2 f1 · 10 + 2 f3 · 2(b - 2c)
        g[1::4] = 20.0 * f1 + 4.0 * bm2c * f3
        # ∂f/∂c: 2 f2 · s5 + 2 f3 · (-4)(b - 2c)
        g[2::4] = 2.0 * s5 * f2 - 8.0 * bm2c * f3
        # ∂f/∂d: 2 f2 · (-s5) + 2 f4 · (-2 s10 · amd)
        g[3::4] = -2.0 * s5 * f2 - 4.0 * s10 * amd * f4
        return g

    prob = NLPProblem(
        n=n, m_eq=0, m_ineq=0,
        xl=np.full(n, -np.inf), xu=np.full(n, np.inf),
        objective=obj, gradient=grad,
        x0=x0,
    )
    return prob, x0, 0.0


def variably_dimensioned(n: int):
    """
    MGH #25 — Variably dimensioned function.

        F_i      = x_i - 1                       , i = 1..n
        F_{n+1}  = sum_j j (x_j - 1)
        F_{n+2}  = (sum_j j (x_j - 1))²

        f(x) = sum F^2

    f* = 0 at x* = (1,...,1).  x0_i = 1 - i/n.
    """
    j = np.arange(1, n + 1, dtype=float)
    x0 = 1.0 - j / n

    def obj(x):
        d = x - 1.0
        s = float(np.dot(j, d))
        return float(np.dot(d, d) + s*s + s**4)

    def grad(x):
        d = x - 1.0
        s = float(np.dot(j, d))
        # d(d_i²)/dx_i = 2 d_i
        # d(s²)/dx_i = 2 s · j_i
        # d(s⁴)/dx_i = 4 s³ · j_i
        return 2.0 * d + (2.0 * s + 4.0 * s**3) * j

    prob = NLPProblem(
        n=n, m_eq=0, m_ineq=0,
        xl=np.full(n, -np.inf), xu=np.full(n, np.inf),
        objective=obj, gradient=grad,
        x0=x0,
    )
    return prob, x0, 0.0


# ---------------------------------------------------------------------------
# Constrained scalable problems with closed-form optima
# ---------------------------------------------------------------------------

def eq_constrained_quadratic(n: int, m_eq: int = None, seed: int = 42):
    """
    Random PSD quadratic with linear equality constraints.

        min   (1/2) xᵀ Q x + cᵀ x
        s.t.  A x = b

    Q = LᵀL + I (PSD, well-conditioned).  KKT system is linear → closed-form
    optimum used for f*.  Default m_eq = max(2, n//10).
    """
    if m_eq is None:
        m_eq = max(2, n // 10)
    rng = np.random.default_rng(seed)
    L  = rng.standard_normal((n, n)) / np.sqrt(n)
    Q  = L.T @ L + np.eye(n)
    c  = rng.standard_normal(n)
    A  = rng.standard_normal((m_eq, n))
    b  = rng.standard_normal(m_eq)

    # Solve [Q Aᵀ; A 0] [x; λ] = [-c; b] for x*
    KKT = np.block([[Q, A.T], [A, np.zeros((m_eq, m_eq))]])
    rhs = np.concatenate([-c, b])
    sol = np.linalg.solve(KKT, rhs)
    x_star = sol[:n]
    f_opt  = 0.5 * float(x_star @ Q @ x_star) + float(c @ x_star)

    x0 = np.zeros(n)

    def obj(x):
        return 0.5 * float(x @ Q @ x) + float(c @ x)

    def grad(x):
        return Q @ x + c

    def eq(x):
        return A @ x - b

    def jeq(x):
        return A

    prob = NLPProblem(
        n=n, m_eq=m_eq, m_ineq=0,
        xl=np.full(n, -np.inf), xu=np.full(n, np.inf),
        objective=obj, gradient=grad,
        eq_constraints=eq, eq_jacobian=jeq,
        x0=x0,
    )
    return prob, x0, f_opt


def ineq_constrained_quadratic(n: int, m_ineq: int = None, seed: int = 42):
    """
    Bound-only quadratic where m_ineq sign-flipped axes activate at the optimum.

        min   (1/2) ||x - c||²
        s.t.  -x_i ≤ 0,  i = 0..m_ineq-1     (= x_i ≥ 0)

    where c has c_i = -1 for i < m_ineq (forcing those bounds active) and
    c_i = +i/n otherwise.  f* = 0.5 · m_ineq.

    Default m_ineq = max(3, n//5).
    """
    if m_ineq is None:
        m_ineq = max(3, n // 5)
    c = np.where(np.arange(n) < m_ineq, -1.0, np.arange(n) / n)
    x_star = np.where(np.arange(n) < m_ineq, 0.0, c)
    f_opt  = 0.5 * float(np.sum((x_star - c) ** 2))
    x0 = np.ones(n)

    def obj(x):
        d = x - c
        return 0.5 * float(np.dot(d, d))

    def grad(x):
        return x - c

    def ineq(x):
        return -x[:m_ineq]

    Jineq = np.zeros((m_ineq, n))
    for i in range(m_ineq):
        Jineq[i, i] = -1.0

    def jineq(x):
        return Jineq

    prob = NLPProblem(
        n=n, m_eq=0, m_ineq=m_ineq,
        xl=np.full(n, -np.inf), xu=np.full(n, np.inf),
        objective=obj, gradient=grad,
        ineq_constraints=ineq, ineq_jacobian=jineq,
        x0=x0,
    )
    return prob, x0, f_opt


# ---------------------------------------------------------------------------
# Registry — name → builder.  Each builder takes n and returns (prob, x0, f*).
# ---------------------------------------------------------------------------

BUILDERS = {
    "broyden_tridiagonal":    broyden_tridiagonal,
    "discrete_bvp":           discrete_bvp,
    "extended_powell":        extended_powell,
    "variably_dimensioned":   variably_dimensioned,
    "eq_quadratic":           eq_constrained_quadratic,
    "ineq_quadratic":         ineq_constrained_quadratic,
    "rosenbrock_chain":       rosenbrock_chain,
}
