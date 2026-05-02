"""
Head-to-head benchmark: pyfiltersqp vs scipy SLSQP vs cyipopt IPOPT.

Runs the same problems through each solver and prints a wall-clock comparison.
The point is to know the gap (or lack thereof) before deciding to rewrite hot
paths in C/Numba.

Problems
--------
* HS71      — small classic SQP demo (n=4, m_eq=1, m_ineq=1)
* Rosenbrock-chain  — scalable n=50, 100, 200, 500 (1 eq + 1 ineq + bounds)

For each (problem, solver) we report best of 3 wall-clock runs, plus the
final objective and the L∞ KKT residual at the returned point.

Usage
-----
    uv run python scripts/bench_vs_solvers.py
    uv run python scripts/bench_vs_solvers.py --sizes 100,500
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pyfiltersqp import NLPProblem, solve as pyfsqp_solve

# Reuse Rosenbrock-chain definition from profile_solver.py
import importlib.util as _u
import os as _os
_spec = _u.spec_from_file_location(
    "_profile_solver",
    _os.path.join(_os.path.dirname(__file__), "profile_solver.py"),
)
_mod = _u.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
make_rosenbrock_nlp = _mod.make_rosenbrock_nlp


# ---------------------------------------------------------------------------
# HS71 (Hock-Schittkowski #71)
#   min   x0 x3 (x0 + x1 + x2) + x2
#   s.t.  x0 x1 x2 x3        >=  25     →  -(... - 25)         <= 0
#         x0² + x1² + x2² + x3² == 40
#         1 ≤ x ≤ 5
#   x* ≈ (1, 4.7429996, 3.8211503, 1.3794082);  f* ≈ 17.0140173
# ---------------------------------------------------------------------------

def make_hs71() -> tuple[NLPProblem, np.ndarray]:
    n = 4
    xl = np.full(n, 1.0)
    xu = np.full(n, 5.0)

    def obj(x):
        return float(x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2])

    def grad(x):
        return np.array([
            x[3] * (2.0 * x[0] + x[1] + x[2]),
            x[0] * x[3],
            x[0] * x[3] + 1.0,
            x[0] * (x[0] + x[1] + x[2]),
        ])

    def ceq(x):
        return np.array([x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 - 40.0])

    def Jeq(x):
        return np.array([[2.0 * x[0], 2.0 * x[1], 2.0 * x[2], 2.0 * x[3]]])

    def cineq(x):
        # x0 x1 x2 x3 >= 25  →  25 - x0 x1 x2 x3 ≤ 0
        return np.array([25.0 - x[0] * x[1] * x[2] * x[3]])

    def Jineq(x):
        return np.array([[-x[1] * x[2] * x[3],
                          -x[0] * x[2] * x[3],
                          -x[0] * x[1] * x[3],
                          -x[0] * x[1] * x[2]]])

    prob = NLPProblem(
        n=n, m_eq=1, m_ineq=1,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        eq_constraints=ceq, eq_jacobian=Jeq,
        ineq_constraints=cineq, ineq_jacobian=Jineq,
    )
    x0 = np.array([1.0, 5.0, 5.0, 1.0])
    return prob, x0


# ---------------------------------------------------------------------------
# KKT residual evaluator (problem-agnostic)
# ---------------------------------------------------------------------------

def kkt_inf(prob: NLPProblem, x: np.ndarray) -> float:
    """L∞ primal feasibility: max(||c_eq||∞, ||max(0, c_ineq)||∞, bound viol)."""
    n = prob.n
    feas = []
    if prob.m_eq > 0:
        feas.append(float(np.max(np.abs(prob.eq_constraints(x)))))
    if prob.m_ineq > 0:
        feas.append(float(np.max(np.maximum(0.0, prob.ineq_constraints(x)))))
    feas.append(float(np.max(np.maximum(0.0, prob.xl - x))))
    feas.append(float(np.max(np.maximum(0.0, x - prob.xu))))
    return max(feas)


# ---------------------------------------------------------------------------
# Solver runners — all return (wall_seconds, x_final, f_final, success)
# ---------------------------------------------------------------------------

def _time(fn: Callable, repeat: int = 3) -> tuple[float, object]:
    best = float("inf")
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best, out


def run_pyfsqp(prob: NLPProblem, x0: np.ndarray, backend: str) -> dict:
    kw = {}
    if backend == "implicit":
        kw["use_implicit_qp"] = True
    elif backend == "woodbury":
        kw["use_woodbury_admm"] = True
    elif backend != "osqp":
        raise ValueError(backend)

    def go():
        return pyfsqp_solve(prob, x0.copy(), **kw)

    dt, res = _time(go)
    return dict(
        time=dt,
        x=res.x,
        obj=res.obj,
        success=res.success,
        iters=res.iterations,
        kkt=kkt_inf(prob, res.x),
    )


def run_slsqp(prob: NLPProblem, x0: np.ndarray) -> dict:
    from scipy.optimize import minimize

    bounds = list(zip(prob.xl, prob.xu))
    constraints = []
    if prob.m_eq > 0:
        constraints.append(dict(type="eq",
                                fun=prob.eq_constraints,
                                jac=prob.eq_jacobian))
    if prob.m_ineq > 0:
        # SLSQP convention: c(x) ≥ 0  → use -c_ineq
        constraints.append(dict(type="ineq",
                                fun=lambda x: -prob.ineq_constraints(x),
                                jac=lambda x: -prob.ineq_jacobian(x)))

    def go():
        return minimize(prob.objective, x0.copy(), jac=prob.gradient,
                        method="SLSQP", bounds=bounds, constraints=constraints,
                        options=dict(maxiter=500, ftol=1e-8))

    dt, res = _time(go)
    return dict(
        time=dt, x=res.x, obj=float(res.fun), success=bool(res.success),
        iters=int(getattr(res, "nit", 0)),
        kkt=kkt_inf(prob, res.x),
    )


def run_ipopt(prob: NLPProblem, x0: np.ndarray) -> dict:
    from cyipopt import minimize_ipopt

    # minimize_ipopt: constraints in scipy.optimize style
    bounds = list(zip(prob.xl, prob.xu))
    constraints = []
    if prob.m_eq > 0:
        constraints.append(dict(type="eq",
                                fun=prob.eq_constraints,
                                jac=prob.eq_jacobian))
    if prob.m_ineq > 0:
        constraints.append(dict(type="ineq",
                                fun=lambda x: -prob.ineq_constraints(x),
                                jac=lambda x: -prob.ineq_jacobian(x)))

    def go():
        return minimize_ipopt(prob.objective, x0.copy(), jac=prob.gradient,
                              bounds=bounds, constraints=constraints,
                              options={"print_level": 0,
                                       "tol": 1e-8,
                                       "max_iter": 500})

    dt, res = _time(go)
    return dict(
        time=dt, x=np.asarray(res.x), obj=float(res.fun),
        success=bool(res.success),
        iters=int(getattr(res, "nit", 0)),
        kkt=kkt_inf(prob, np.asarray(res.x)),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

SOLVERS = [
    ("pyfsqp/osqp",     lambda p, x: run_pyfsqp(p, x, "osqp")),
    ("pyfsqp/implicit", lambda p, x: run_pyfsqp(p, x, "implicit")),
    ("pyfsqp/woodbury", lambda p, x: run_pyfsqp(p, x, "woodbury")),
    ("scipy SLSQP",     run_slsqp),
    ("cyipopt IPOPT",   run_ipopt),
]


def print_problem_results(name: str, prob: NLPProblem, x0: np.ndarray,
                          baseline: str = "cyipopt IPOPT") -> None:
    print(f"\n=== {name}   n={prob.n}  m_eq={prob.m_eq}  m_ineq={prob.m_ineq} ===")
    print(f"{'solver':<18} {'time(ms)':>10} {'iters':>6} {'obj':>14} "
          f"{'kkt':>10} {'rel-time':>10}  status")
    print("-" * 86)

    rows = []
    for label, fn in SOLVERS:
        try:
            r = fn(prob, x0)
            rows.append((label, r))
        except Exception as e:
            rows.append((label, dict(error=str(e))))

    base_time = next(
        (r["time"] for lbl, r in rows if lbl == baseline and "error" not in r),
        None,
    )

    for label, r in rows:
        if "error" in r:
            print(f"{label:<18} ERROR: {r['error'][:60]}")
            continue
        rel = (r["time"] / base_time) if base_time else float("nan")
        print(f"{label:<18} {r['time']*1000:>10.2f} {r['iters']:>6} "
              f"{r['obj']:>14.6e} {r['kkt']:>10.2e} "
              f"{rel:>9.2f}×  {'ok' if r['success'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", default="50,100,200,500",
                    help="Rosenbrock-chain sizes (default 50,100,200,500)")
    ap.add_argument("--baseline", default="cyipopt IPOPT",
                    help="Solver to use as 1.00× baseline in rel-time column")
    args = ap.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    # HS71 (small)
    prob, x0 = make_hs71()
    print_problem_results("HS71", prob, x0, baseline=args.baseline)

    # Rosenbrock chain at multiple sizes
    for n in sizes:
        prob, x0 = make_rosenbrock_nlp(n)
        print_problem_results(f"Rosenbrock-chain", prob, x0, baseline=args.baseline)

    print("\nNotes:")
    print("  * time(ms) is best of 3 runs.  rel-time is solver_time / baseline_time.")
    print("  * kkt is L∞ primal feasibility (||c_eq||, ||max(0,c_ineq)||, bound viol).")
    print("  * IPOPT solves an interior-point system per iter (different work per iter)")
    print("    so iter count is not directly comparable across solvers.")


if __name__ == "__main__":
    main()
