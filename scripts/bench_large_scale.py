"""
Large-scale solver validation.

Runs each problem in ``tests.large_scale_problems`` at a sweep of n values
through pyfiltersqp's three QP backends and cyipopt IPOPT.  Reports wall
time, iterations, final objective vs known optimum, and KKT residual.

The point is to find scaling regressions and bugs that don't show up on
HS-sized problems (n ≤ 7).

Usage
-----
    uv run python scripts/bench_large_scale.py
    uv run python scripts/bench_large_scale.py --sizes 50,500,2000
    uv run python scripts/bench_large_scale.py --problems broyden_tridiagonal,discrete_bvp
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

# Make the project root importable so `tests.large_scale_problems` resolves
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyfiltersqp import NLPProblem, solve as pyfsqp_solve
from tests.large_scale_problems import BUILDERS


# ---------------------------------------------------------------------------
# KKT residual (problem-agnostic)
# ---------------------------------------------------------------------------

def kkt_inf(prob: NLPProblem, x: np.ndarray) -> float:
    feas = [0.0]
    if prob.m_eq > 0:
        feas.append(float(np.max(np.abs(prob.eq_constraints(x)))))
    if prob.m_ineq > 0:
        feas.append(float(np.max(np.maximum(0.0, prob.ineq_constraints(x)))))
    feas.append(float(np.max(np.maximum(0.0, prob.xl - x))))
    feas.append(float(np.max(np.maximum(0.0, x - prob.xu))))
    return float(max(feas))


# ---------------------------------------------------------------------------
# Solver runners
# ---------------------------------------------------------------------------

def run_pyfsqp(prob: NLPProblem, x0: np.ndarray, backend: str,
               max_iter: int = 500) -> dict:
    kw = dict(max_iter=max_iter)
    if backend == "implicit":
        kw["use_implicit_qp"] = True
    elif backend == "woodbury":
        kw["use_woodbury_admm"] = True
    elif backend != "osqp":
        raise ValueError(backend)

    t0 = time.perf_counter()
    res = pyfsqp_solve(prob, x0.copy(), **kw)
    dt = time.perf_counter() - t0
    return dict(
        time=dt, iters=res.iterations, obj=res.obj,
        kkt=kkt_inf(prob, res.x), success=res.success,
    )


def run_ipopt(prob: NLPProblem, x0: np.ndarray, max_iter: int = 500) -> dict:
    from cyipopt import minimize_ipopt

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

    t0 = time.perf_counter()
    try:
        res = minimize_ipopt(prob.objective, x0.copy(), jac=prob.gradient,
                             bounds=bounds, constraints=constraints,
                             options={"print_level": 0,
                                      "tol": 1e-8,
                                      "max_iter": max_iter})
    except Exception as e:
        return dict(time=time.perf_counter() - t0, iters=0,
                    obj=float("nan"), kkt=float("nan"),
                    success=False, error=str(e))
    dt = time.perf_counter() - t0
    x = np.asarray(res.x)
    return dict(
        time=dt, iters=int(getattr(res, "nit", 0)), obj=float(res.fun),
        kkt=kkt_inf(prob, x), success=bool(res.success),
    )


SOLVERS = [
    ("pyfsqp/osqp",     lambda p, x: run_pyfsqp(p, x, "osqp")),
    ("pyfsqp/implicit", lambda p, x: run_pyfsqp(p, x, "implicit")),
    ("pyfsqp/woodbury", lambda p, x: run_pyfsqp(p, x, "woodbury")),
    ("cyipopt IPOPT",   lambda p, x: run_ipopt(p, x)),
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt_obj_gap(obj: float, f_opt: float) -> str:
    """Relative objective gap.  f_opt may be 0 → use absolute."""
    if not np.isfinite(obj):
        return "  nan  "
    if abs(f_opt) < 1e-8:
        return f"{abs(obj - f_opt):>8.1e}"
    return f"{abs(obj - f_opt) / max(1.0, abs(f_opt)):>8.1e}"


def print_problem_size(name: str, n: int, prob: NLPProblem,
                       x0: np.ndarray, f_opt: float) -> None:
    print(f"\n--- {name}  n={prob.n}  m_eq={prob.m_eq}  m_ineq={prob.m_ineq}  "
          f"f*={f_opt:.4g} ---")
    print(f"  {'solver':<18} {'time(ms)':>10} {'iters':>6} {'obj':>14} "
          f"{'gap':>10} {'kkt':>10}  status")

    for label, fn in SOLVERS:
        try:
            r = fn(prob, x0)
        except Exception as e:
            print(f"  {label:<18}  ERROR: {str(e)[:60]}")
            continue
        if r.get("error"):
            print(f"  {label:<18}  ERROR: {r['error'][:60]}")
            continue
        status = "ok" if r["success"] else "FAIL"
        print(f"  {label:<18} {r['time']*1000:>10.1f} {r['iters']:>6} "
              f"{r['obj']:>14.6e} {fmt_obj_gap(r['obj'], f_opt):>10} "
              f"{r['kkt']:>10.2e}  {status}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes",    default="50,200,1000",
                    help="comma-separated n values (default 50,200,1000)")
    ap.add_argument("--problems", default=",".join(BUILDERS.keys()),
                    help="comma-separated problem names")
    args = ap.parse_args()

    sizes = [int(s) for s in args.sizes.split(",")]
    names = [s.strip() for s in args.problems.split(",")]

    for name in names:
        if name not in BUILDERS:
            print(f"\n[skip] unknown problem {name!r}")
            continue
        builder = BUILDERS[name]
        print(f"\n=== {name} ===")
        for n in sizes:
            try:
                # extended_powell needs n divisible by 4
                if name == "extended_powell" and n % 4:
                    n = (n // 4) * 4 + 4
                # rosenbrock_chain needs n even
                if name == "rosenbrock_chain" and n % 2:
                    n += 1
                prob, x0, f_opt = builder(n)
            except Exception as e:
                print(f"  [skip] n={n}: builder error {e}")
                continue
            print_problem_size(name, n, prob, x0, f_opt)

    print()
    print("Notes:")
    print("  * gap   = |obj - f*| / max(1, |f*|)  (absolute when f* = 0)")
    print("  * kkt   = L∞ primal feasibility")
    print("  * IPOPT 'success' is its own convergence flag; gap+kkt are the")
    print("    independent ground truth across all solvers.")


if __name__ == "__main__":
    main()
