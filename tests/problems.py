"""
Hock-Schittkowski test problems with exact Jacobians.

All problems are taken from:
    Hock & Schittkowski (1981), "Test Examples for Nonlinear Programming Codes",
    Lecture Notes in Economics and Mathematical Systems, Vol. 187, Springer.

Problems included (in `PROBLEMS`):
    HS1   — 2 vars, bound only           f* = 0
    HS3   — 2 vars, bound only           f* = 0
    HS5   — 2 vars, bound only, sin obj  f* ≈ -1.91322
    HS6   — 2 vars, 1 nonlinear eq       f* = 0
    HS10  — 2 vars, 1 nonlinear ineq     f* = -1
    HS12  — 2 vars, 1 nonlinear ineq     f* = -30
    HS14  — 2 vars, 1 eq + 1 ineq        f* ≈ 1.39346499
    HS15  — 2 vars, 2 ineq, bounds       f* = 306.5
    HS22  — 2 vars, 2 ineq               f* = 1
    HS28  — 3 vars, 1 linear eq          f* = 0
    HS43  — 4 vars, 3 ineq (Rosen-Suzuki)  f* = -44
    HS50  — 5 vars, 3 linear eq          f* = 0
    HS71  — 4 vars, 1 eq + 1 ineq, bounds  f* ≈ 17.0140173
    HS100 — 7 vars, 4 ineq, bounds       f* ≈ 680.6300573

Each entry is a dict with keys:
    problem : NLPProblem
    x0      : ndarray   initial point
    x_opt   : ndarray   known optimal solution (for reference)
    f_opt   : float     known optimal objective
"""
from __future__ import annotations

import numpy as np
from pyfiltersqp import NLPProblem

_INF = np.inf


def _hs15() -> dict:
    """
    HS Problem 15.

        min   100*(x2 - x1²)² + (1 - x1)²
        s.t.  1 - x1*x2          ≤ 0   (c1: product ≥ 1)
              -x1 - x2²          ≤ 0   (c2: x1 + x2² ≥ 0)
              x1 ≤ 0.5,  x2 ≥ -0.5

    f* = 306.5  at  x* = (0.5, 2.0)
    """
    xl = np.array([-_INF, -0.5])
    xu = np.array([0.5,    _INF])
    x0 = np.array([-2.0, 1.0])
    x_opt = np.array([0.5, 2.0])
    f_opt = 306.5

    def obj(x):
        return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2

    def grad(x):
        return np.array([
            -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1.0 - x[0]),
             200.0 * (x[1] - x[0]**2),
        ])

    def ineq(x):
        return np.array([
            1.0 - x[0] * x[1],     # ≤ 0  →  x1*x2 ≥ 1
           -x[0] - x[1]**2,         # ≤ 0  →  x1 + x2² ≥ 0
        ])

    def jineq(x):
        return np.array([
            [-x[1],      -x[0]  ],
            [-1.0,   -2.0*x[1]  ],
        ])

    problem = NLPProblem(
        n=2, m_eq=0, m_ineq=2,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        ineq_constraints=ineq, ineq_jacobian=jineq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


def _hs71() -> dict:
    """
    HS Problem 71 — the classic 4-variable SQP demo problem.

        min   x1*x4*(x1+x2+x3) + x3
        s.t.  x1*x2*x3*x4 >= 25              (ineq, written as 25 - prod ≤ 0)
              x1² + x2² + x3² + x4² = 40    (eq)
              1 ≤ xi ≤ 5

    f* = 17.01401727...  at x* = (1, 4.742, 3.821, 1.379)
    """
    xl = np.ones(4)
    xu = 5.0 * np.ones(4)
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    x_opt = np.array([1.0, 4.74299963, 3.82114998, 1.37940829])
    f_opt = 17.01401727

    def obj(x):
        return x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]

    def grad(x):
        return np.array([
            x[3] * (2*x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ])

    def eq(x):
        return np.array([x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 40.0])

    def jeq(x):
        return np.array([[2*x[0], 2*x[1], 2*x[2], 2*x[3]]])

    def ineq(x):
        return np.array([25.0 - x[0]*x[1]*x[2]*x[3]])   # ≤ 0  →  product ≥ 25

    def jineq(x):
        return np.array([[
            -x[1]*x[2]*x[3],
            -x[0]*x[2]*x[3],
            -x[0]*x[1]*x[3],
            -x[0]*x[1]*x[2],
        ]])

    problem = NLPProblem(
        n=4, m_eq=1, m_ineq=1,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        eq_constraints=eq, eq_jacobian=jeq,
        ineq_constraints=ineq, ineq_jacobian=jineq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


def _hs100() -> dict:
    """
    HS Problem 100 — 7-variable, 4-inequality benchmark.

        min   (x1-10)² + 5(x2-12)² + x3⁴ + 3(x4-11)² + 10x5⁶
              + 7x6² + x7⁴ - 4x6x7 - 10x6 - 8x7
        s.t.  2x1² + 3x2⁴ + x3 + 4x4² + 5x5 - 127        ≤ 0   (g1)
              7x1 + 3x2 + 10x3² + x4 - x5 - 282           ≤ 0   (g2)
              23x1 + x2² + 6x6² - 8x7 - 196               ≤ 0   (g3)
              4x1² + x2² - 3x1x2 + 2x3² + 5x6 - 11x7      ≤ 0   (g4)
              -10 ≤ xi ≤ 10,  i = 1..7

    f* ≈ 680.6300573  at x* ≈ (2.330499, 1.951372, -0.477541,
                                4.365726, -0.624487, 1.038131, 1.594227)
    """
    xl = np.full(7, -10.0)
    xu = np.full(7,  10.0)
    x0 = np.array([1.0, 2.0, 0.0, 4.0, 0.0, 1.0, 1.0])
    x_opt = np.array([2.330499, 1.951372, -0.477541, 4.365726,
                      -0.624487, 1.038131, 1.594227])
    f_opt = 680.6300573

    def obj(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return ((x1 - 10)**2 + 5*(x2 - 12)**2 + x3**4 + 3*(x4 - 11)**2
                + 10*x5**6 + 7*x6**2 + x7**4 - 4*x6*x7 - 10*x6 - 8*x7)

    def grad(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return np.array([
            2*(x1 - 10),
            10*(x2 - 12),
            4*x3**3,
            6*(x4 - 11),
            60*x5**5,
            14*x6 - 4*x7 - 10,
            4*x7**3 - 4*x6 - 8,
        ])

    def ineq(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return np.array([
            2*x1**2 + 3*x2**4 + x3 + 4*x4**2 + 5*x5 - 127,       # g1 ≤ 0
            7*x1 + 3*x2 + 10*x3**2 + x4 - x5 - 282,               # g2 ≤ 0
            23*x1 + x2**2 + 6*x6**2 - 8*x7 - 196,                  # g3 ≤ 0
            4*x1**2 + x2**2 - 3*x1*x2 + 2*x3**2 + 5*x6 - 11*x7,   # g4 ≤ 0
        ])

    def jineq(x):
        x1, x2, x3, x4, x5, x6, x7 = x
        return np.array([
            # g1
            [4*x1,        12*x2**3,  1,       8*x4,  5,   0,       0  ],
            # g2
            [7,           3,         20*x3,   1,     -1,   0,       0  ],
            # g3
            [23,          2*x2,      0,       0,      0,   12*x6,  -8  ],
            # g4
            [8*x1 - 3*x2, 2*x2 - 3*x1, 4*x3, 0,     0,   5,      -11 ],
        ])

    problem = NLPProblem(
        n=7, m_eq=0, m_ineq=4,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        ineq_constraints=ineq, ineq_jacobian=jineq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


# ---------------------------------------------------------------------------
# Bound-only / nearly unconstrained
# ---------------------------------------------------------------------------

def _hs1() -> dict:
    """HS1 — Rosenbrock with single lower bound.  f* = 0 at x* = (1, 1)."""
    xl = np.array([-_INF, -1.5])
    xu = np.array([ _INF,  _INF])
    x0 = np.array([-2.0, 1.0])
    x_opt = np.array([1.0, 1.0])
    f_opt = 0.0

    def obj(x):
        return 100.0 * (x[1] - x[0]**2)**2 + (1.0 - x[0])**2

    def grad(x):
        return np.array([
            -400.0 * x[0] * (x[1] - x[0]**2) - 2.0 * (1.0 - x[0]),
             200.0 * (x[1] - x[0]**2),
        ])

    problem = NLPProblem(
        n=2, m_eq=0, m_ineq=0,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


def _hs3() -> dict:
    """HS3 — bound active at optimum.  f* = 0 at x* = (0, 0), x2 ≥ 0 active."""
    xl = np.array([-_INF, 0.0])
    xu = np.array([ _INF, _INF])
    x0 = np.array([10.0, 1.0])
    x_opt = np.array([0.0, 0.0])
    f_opt = 0.0
    eps = 1e-5

    def obj(x):
        return x[1] + eps * (x[1] - x[0])**2

    def grad(x):
        d = x[1] - x[0]
        return np.array([-2.0 * eps * d, 1.0 + 2.0 * eps * d])

    problem = NLPProblem(
        n=2, m_eq=0, m_ineq=0,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


def _hs5() -> dict:
    """
    HS5 — bounded sin-cos.

        min   sin(x1+x2) + (x1-x2)² - 1.5 x1 + 2.5 x2 + 1
        s.t.  -1.5 ≤ x1 ≤ 4,  -3 ≤ x2 ≤ 3

    f* = -√3/2 - π/3 ≈ -1.91322295
    x* = (-π/3 + 1/2, -π/3 - 1/2) ≈ (-0.547197, -1.547197)
    """
    xl = np.array([-1.5, -3.0])
    xu = np.array([ 4.0,  3.0])
    x0 = np.array([0.0, 0.0])
    x_opt = np.array([-np.pi/3 + 0.5, -np.pi/3 - 0.5])
    f_opt = -np.sqrt(3.0)/2.0 - np.pi/3.0

    def obj(x):
        s = x[0] + x[1]
        d = x[0] - x[1]
        return float(np.sin(s) + d*d - 1.5*x[0] + 2.5*x[1] + 1.0)

    def grad(x):
        c = np.cos(x[0] + x[1])
        d = x[0] - x[1]
        return np.array([c + 2.0*d - 1.5, c - 2.0*d + 2.5])

    problem = NLPProblem(
        n=2, m_eq=0, m_ineq=0,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


# ---------------------------------------------------------------------------
# Equality-only
# ---------------------------------------------------------------------------

def _hs6() -> dict:
    """HS6 — 1 nonlinear equality.  f* = 0 at x* = (1, 1)."""
    xl = np.array([-_INF, -_INF])
    xu = np.array([ _INF,  _INF])
    x0 = np.array([-1.2, 1.0])
    x_opt = np.array([1.0, 1.0])
    f_opt = 0.0

    def obj(x):
        return (1.0 - x[0])**2

    def grad(x):
        return np.array([-2.0 * (1.0 - x[0]), 0.0])

    def eq(x):
        return np.array([10.0 * (x[1] - x[0]**2)])

    def jeq(x):
        return np.array([[-20.0 * x[0], 10.0]])

    problem = NLPProblem(
        n=2, m_eq=1, m_ineq=0,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        eq_constraints=eq, eq_jacobian=jeq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


def _hs28() -> dict:
    """HS28 — 3 vars, 1 linear equality.  f* = 0 at x* = (0.5, -0.5, 0.5)."""
    xl = np.full(3, -_INF)
    xu = np.full(3,  _INF)
    x0 = np.array([-4.0, 1.0, 1.0])
    x_opt = np.array([0.5, -0.5, 0.5])
    f_opt = 0.0

    def obj(x):
        return (x[0] + x[1])**2 + (x[1] + x[2])**2

    def grad(x):
        a = 2.0 * (x[0] + x[1])
        b = 2.0 * (x[1] + x[2])
        return np.array([a, a + b, b])

    def eq(x):
        return np.array([x[0] + 2.0*x[1] + 3.0*x[2] - 1.0])

    def jeq(x):
        return np.array([[1.0, 2.0, 3.0]])

    problem = NLPProblem(
        n=3, m_eq=1, m_ineq=0,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        eq_constraints=eq, eq_jacobian=jeq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


# ---------------------------------------------------------------------------
# Inequality-only
# ---------------------------------------------------------------------------

def _hs10() -> dict:
    """
    HS10 — 1 nonlinear inequality.

        min   x1 - x2
        s.t.  -3x1² + 2 x1 x2 - x2² + 1 ≥ 0
              ⇒  3x1² - 2 x1 x2 + x2² - 1 ≤ 0

    f* = -1 at x* = (0, 1).
    """
    xl = np.full(2, -_INF)
    xu = np.full(2,  _INF)
    x0 = np.array([-10.0, 10.0])
    x_opt = np.array([0.0, 1.0])
    f_opt = -1.0

    def obj(x):
        return x[0] - x[1]

    def grad(x):
        return np.array([1.0, -1.0])

    def ineq(x):
        return np.array([3.0*x[0]**2 - 2.0*x[0]*x[1] + x[1]**2 - 1.0])

    def jineq(x):
        return np.array([[6.0*x[0] - 2.0*x[1], -2.0*x[0] + 2.0*x[1]]])

    problem = NLPProblem(
        n=2, m_eq=0, m_ineq=1,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        ineq_constraints=ineq, ineq_jacobian=jineq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


def _hs12() -> dict:
    """
    HS12 — 1 quadratic inequality.

        min   0.5 x1² + x2² - x1 x2 - 7 x1 - 7 x2
        s.t.  25 - 4 x1² - x2² ≥ 0   ⇒   4 x1² + x2² - 25 ≤ 0

    f* = -30 at x* = (2, 3).
    """
    xl = np.full(2, -_INF)
    xu = np.full(2,  _INF)
    x0 = np.array([0.0, 0.0])
    x_opt = np.array([2.0, 3.0])
    f_opt = -30.0

    def obj(x):
        return 0.5*x[0]**2 + x[1]**2 - x[0]*x[1] - 7.0*x[0] - 7.0*x[1]

    def grad(x):
        return np.array([x[0] - x[1] - 7.0, 2.0*x[1] - x[0] - 7.0])

    def ineq(x):
        return np.array([4.0*x[0]**2 + x[1]**2 - 25.0])

    def jineq(x):
        return np.array([[8.0*x[0], 2.0*x[1]]])

    problem = NLPProblem(
        n=2, m_eq=0, m_ineq=1,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        ineq_constraints=ineq, ineq_jacobian=jineq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


# ---------------------------------------------------------------------------
# Mixed equality + inequality
# ---------------------------------------------------------------------------

def _hs14() -> dict:
    """
    HS14 — 1 linear equality + 1 quadratic inequality.

        min   (x1 - 2)² + (x2 - 1)²
        s.t.  x1 - 2 x2 + 1 = 0
              -x1²/4 - x2² + 1 ≥ 0    ⇒  x1²/4 + x2² - 1 ≤ 0

    f* = 9 - 2.875·√7 ≈ 1.39346499 at x* = ((√7 - 1)/2, (√7 + 1)/4).
    """
    xl = np.full(2, -_INF)
    xu = np.full(2,  _INF)
    x0 = np.array([2.0, 2.0])
    sqrt7 = np.sqrt(7.0)
    x_opt = np.array([0.5*(sqrt7 - 1.0), 0.25*(sqrt7 + 1.0)])
    f_opt = 9.0 - 2.875 * sqrt7

    def obj(x):
        return (x[0] - 2.0)**2 + (x[1] - 1.0)**2

    def grad(x):
        return np.array([2.0*(x[0] - 2.0), 2.0*(x[1] - 1.0)])

    def eq(x):
        return np.array([x[0] - 2.0*x[1] + 1.0])

    def jeq(x):
        return np.array([[1.0, -2.0]])

    def ineq(x):
        return np.array([0.25*x[0]**2 + x[1]**2 - 1.0])

    def jineq(x):
        return np.array([[0.5*x[0], 2.0*x[1]]])

    problem = NLPProblem(
        n=2, m_eq=1, m_ineq=1,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        eq_constraints=eq, eq_jacobian=jeq,
        ineq_constraints=ineq, ineq_jacobian=jineq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


def _hs22() -> dict:
    """
    HS22 — 2 inequalities, both active at the optimum.

        min   (x1 - 2)² + (x2 - 1)²
        s.t.  -x1 - x2 + 2 ≥ 0    ⇒  x1 + x2 - 2 ≤ 0
              -x1² + x2    ≥ 0    ⇒  x1² - x2   ≤ 0

    f* = 1 at x* = (1, 1).
    """
    xl = np.full(2, -_INF)
    xu = np.full(2,  _INF)
    x0 = np.array([2.0, 2.0])
    x_opt = np.array([1.0, 1.0])
    f_opt = 1.0

    def obj(x):
        return (x[0] - 2.0)**2 + (x[1] - 1.0)**2

    def grad(x):
        return np.array([2.0*(x[0] - 2.0), 2.0*(x[1] - 1.0)])

    def ineq(x):
        return np.array([x[0] + x[1] - 2.0, x[0]**2 - x[1]])

    def jineq(x):
        return np.array([[1.0, 1.0], [2.0*x[0], -1.0]])

    problem = NLPProblem(
        n=2, m_eq=0, m_ineq=2,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        ineq_constraints=ineq, ineq_jacobian=jineq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


# ---------------------------------------------------------------------------
# Larger / more constraints
# ---------------------------------------------------------------------------

def _hs43() -> dict:
    """
    HS43 (Rosen-Suzuki) — 4 vars, 3 inequalities (3 active at optimum).

        min   x1² + x2² + 2 x3² + x4² - 5 x1 - 5 x2 - 21 x3 + 7 x4
        s.t.  x1² +  x2² +  x3² +  x4² + x1 - x2 + x3 - x4 - 8 ≤ 0
              x1² + 2 x2² +  x3² + 2 x4² - x1            - x4 - 10 ≤ 0
            2 x1² +  x2² +  x3²            + 2 x1 - x2 - x4 - 5  ≤ 0

    f* = -44 at x* = (0, 1, 2, -1).
    """
    xl = np.full(4, -_INF)
    xu = np.full(4,  _INF)
    x0 = np.zeros(4)
    x_opt = np.array([0.0, 1.0, 2.0, -1.0])
    f_opt = -44.0

    def obj(x):
        return (x[0]**2 + x[1]**2 + 2.0*x[2]**2 + x[3]**2
                - 5.0*x[0] - 5.0*x[1] - 21.0*x[2] + 7.0*x[3])

    def grad(x):
        return np.array([
            2.0*x[0] - 5.0,
            2.0*x[1] - 5.0,
            4.0*x[2] - 21.0,
            2.0*x[3] + 7.0,
        ])

    def ineq(x):
        return np.array([
            x[0]**2 +     x[1]**2 + x[2]**2 +     x[3]**2 + x[0] - x[1] + x[2] - x[3] -  8.0,
            x[0]**2 + 2.0*x[1]**2 + x[2]**2 + 2.0*x[3]**2 - x[0]                  - x[3] - 10.0,
            2.0*x[0]**2 + x[1]**2 + x[2]**2                + 2.0*x[0] - x[1]      - x[3] -  5.0,
        ])

    def jineq(x):
        return np.array([
            [2.0*x[0] + 1.0, 2.0*x[1] - 1.0, 2.0*x[2] + 1.0, 2.0*x[3] - 1.0],
            [2.0*x[0] - 1.0, 4.0*x[1],       2.0*x[2],       4.0*x[3] - 1.0],
            [4.0*x[0] + 2.0, 2.0*x[1] - 1.0, 2.0*x[2],      -1.0           ],
        ])

    problem = NLPProblem(
        n=4, m_eq=0, m_ineq=3,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        ineq_constraints=ineq, ineq_jacobian=jineq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


def _hs50() -> dict:
    """
    HS50 — 5 vars, 3 linear equalities, quartic objective.

        min   (x1-x2)² + (x2-x3)² + (x3-x4)⁴ + (x4-x5)²
        s.t.  x1 + 2 x2 + 3 x3 = 6
              x2 + 2 x3 + 3 x4 = 6
              x3 + 2 x4 + 3 x5 = 6

    f* = 0 at x* = (1, 1, 1, 1, 1).
    """
    xl = np.full(5, -_INF)
    xu = np.full(5,  _INF)
    x0 = np.array([35.0, -31.0, 11.0, 5.0, -5.0])
    x_opt = np.ones(5)
    f_opt = 0.0

    def obj(x):
        return ((x[0]-x[1])**2 + (x[1]-x[2])**2
                + (x[2]-x[3])**4 + (x[3]-x[4])**2)

    def grad(x):
        d12 = x[0] - x[1]
        d23 = x[1] - x[2]
        d34 = x[2] - x[3]
        d45 = x[3] - x[4]
        c34 = 4.0 * d34**3
        return np.array([
             2.0 * d12,
            -2.0 * d12 + 2.0 * d23,
            -2.0 * d23 + c34,
            -c34       + 2.0 * d45,
            -2.0 * d45,
        ])

    def eq(x):
        return np.array([
            x[0] + 2.0*x[1] + 3.0*x[2] - 6.0,
            x[1] + 2.0*x[2] + 3.0*x[3] - 6.0,
            x[2] + 2.0*x[3] + 3.0*x[4] - 6.0,
        ])

    def jeq(x):
        return np.array([
            [1.0, 2.0, 3.0, 0.0, 0.0],
            [0.0, 1.0, 2.0, 3.0, 0.0],
            [0.0, 0.0, 1.0, 2.0, 3.0],
        ])

    problem = NLPProblem(
        n=5, m_eq=3, m_ineq=0,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        eq_constraints=eq, eq_jacobian=jeq,
        x0=x0,
    )
    return dict(problem=problem, x0=x0, x_opt=x_opt, f_opt=f_opt)


# Registry — add more HS problems here as needed
PROBLEMS: dict[str, dict] = {
    "hs1":   _hs1(),
    "hs3":   _hs3(),
    "hs5":   _hs5(),
    "hs6":   _hs6(),
    "hs10":  _hs10(),
    "hs12":  _hs12(),
    "hs14":  _hs14(),
    "hs15":  _hs15(),
    "hs22":  _hs22(),
    "hs28":  _hs28(),
    "hs43":  _hs43(),
    "hs50":  _hs50(),
    "hs71":  _hs71(),
    "hs100": _hs100(),
}
