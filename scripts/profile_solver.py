"""
Profiler for pyfiltersqp — identifies the dominant cost at each problem size.

Runs a scalable Rosenbrock-chain NLP at increasing n and breaks down
where wall-clock time is spent per SQP iteration:

  • L-BFGS materialise()  — O(n²·memory), batched BLAS rank-1 updates
  • OSQP QP solve         — includes setup (first iter) + update (subsequent)
  • NLP function evals    — obj + grad + constraints + Jacobians at each point

Key finding (post v0.4 vectorised materialise):
  OSQP dense QP is now the bottleneck at EVERY problem size (70–97%).
  materialise is ~10% at n≤200, ~3% at n=500.
  Next lever: implicit L-BFGS QP to avoid building the dense B and A matrices.

Usage:
    uv run python scripts/profile_solver.py
    uv run python scripts/profile_solver.py --sizes 50,100,200,500,1000
    uv run python scripts/profile_solver.py --cprofile --n 200
    uv run python scripts/profile_solver.py --compare   # current vs pre-alloc materialise
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from functools import wraps
from typing import Callable

import numpy as np

from pyfiltersqp import NLPProblem, SQPSolver
from pyfiltersqp._lbfgs import LBFGSMemory
from pyfiltersqp._qp import QPSubproblem
from pyfiltersqp._implicit_qp import ImplicitLBFGSQP
from pyfiltersqp._qp_admm import WoodburyADMM


# ---------------------------------------------------------------------------
# Scalable Rosenbrock-chain NLP
# ---------------------------------------------------------------------------

def make_rosenbrock_nlp(n: int) -> tuple[NLPProblem, np.ndarray]:
    """
    Rosenbrock chain NLP of dimension n (rounded up to even).

    Objective:  sum_{i=0}^{n/2-1} [ 100*(x[2i+1]-x[2i]²)² + (1-x[2i])² ]
    Equality:   sum(x) = 0
    Inequality: -x[n-1] ≤ 1   (i.e., x[n-1] ≥ -1)
    Bounds:     -5 ≤ x ≤ 5

    x0 = tile([-1.2, 1.0], n//2)  — standard Rosenbrock start
    """
    if n % 2:
        n += 1

    xl = np.full(n, -5.0)
    xu = np.full(n,  5.0)

    def obj(x):
        p = x.reshape(-1, 2)
        return float(np.sum(100.0 * (p[:, 1] - p[:, 0] ** 2) ** 2
                            + (1.0 - p[:, 0]) ** 2))

    def grad(x):
        g = np.zeros(n)
        p = x.reshape(-1, 2)
        a, b = p[:, 0], p[:, 1]
        g[0::2] = -400.0 * (b - a ** 2) * a - 2.0 * (1.0 - a)
        g[1::2] =  200.0 * (b - a ** 2)
        return g

    def ceq(x):
        return np.array([np.sum(x)])

    def Jeq(x):
        return np.ones((1, n))

    def cineq(x):
        return np.array([-x[n - 1] - 1.0])   # x[n-1] >= -1

    def Jineq(x):
        J = np.zeros((1, n))
        J[0, n - 1] = -1.0
        return J

    prob = NLPProblem(
        n=n, m_eq=1, m_ineq=1,
        xl=xl, xu=xu,
        objective=obj, gradient=grad,
        eq_constraints=ceq, eq_jacobian=Jeq,
        ineq_constraints=cineq, ineq_jacobian=Jineq,
    )
    x0 = np.tile([-1.2, 1.0], n // 2)
    return prob, x0


# ---------------------------------------------------------------------------
# Component timing via monkey-patching
# ---------------------------------------------------------------------------

class _Timer:
    """Accumulates total time across all calls to a patched method."""

    def __init__(self):
        self.total: float = 0.0
        self.calls: int = 0

    def reset(self):
        self.total = 0.0
        self.calls = 0


def _patch(obj, method_name: str, timer: _Timer):
    """Replace obj.method_name with a timed wrapper (saved original restored later)."""
    original = getattr(obj, method_name)

    @wraps(original)
    def timed(*args, **kwargs):
        t0 = time.perf_counter()
        result = original(*args, **kwargs)
        timer.total += time.perf_counter() - t0
        timer.calls += 1
        return result

    setattr(obj, method_name, timed)
    return original


# ---------------------------------------------------------------------------
# Per-size benchmark
# ---------------------------------------------------------------------------

def benchmark(n: int, max_iter: int = 100, verbose: bool = False,
              backend: str = "osqp") -> dict:
    """
    Run the solver on the n-dimensional Rosenbrock NLP with component timing.

    Parameters
    ----------
    backend : {"osqp", "implicit", "woodbury"}
        QP backend to time.

    Returns a dict with timing breakdown.
    """
    prob, x0 = make_rosenbrock_nlp(n)
    n_actual = prob.n   # may differ from requested n if rounded up

    # Timers
    t_mat  = _Timer()   # LBFGSMemory.materialise (OSQP path only)
    t_qp   = _Timer()   # whichever backend's solve()
    t_eval = _Timer()   # NLP function evaluations (obj + grad + constraints + jacs)

    # Patch the materialise (only meaningful for OSQP path; the implicit/Woodbury
    # paths never call it but we keep the timer for a uniform reporting shape).
    orig_mat = _patch(LBFGSMemory, "materialise", t_mat)

    if backend == "osqp":
        orig_qp = _patch(QPSubproblem,    "solve", t_qp)
        qp_cls  = QPSubproblem
        kw      = dict(backend="osqp")
    elif backend == "implicit":
        orig_qp = _patch(ImplicitLBFGSQP, "solve", t_qp)
        qp_cls  = ImplicitLBFGSQP
        kw      = dict(backend="implicit")
    elif backend == "woodbury":
        orig_qp = _patch(WoodburyADMM,    "solve", t_qp)
        qp_cls  = WoodburyADMM
        kw      = dict(backend="woodbury")
    else:
        raise ValueError(f"unknown backend {backend!r}")

    # Patch the NLP callables to time eval cost
    eval_timer_active = [True]

    def timed_callable(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if eval_timer_active[0]:
                t0 = time.perf_counter()
                r = fn(*args, **kwargs)
                t_eval.total += time.perf_counter() - t0
                t_eval.calls += 1
                return r
            return fn(*args, **kwargs)
        return wrapper

    prob.objective          = timed_callable(prob.objective)
    prob.gradient           = timed_callable(prob.gradient)
    prob.eq_constraints     = timed_callable(prob.eq_constraints)
    prob.eq_jacobian        = timed_callable(prob.eq_jacobian)
    prob.ineq_constraints   = timed_callable(prob.ineq_constraints)
    prob.ineq_jacobian      = timed_callable(prob.ineq_jacobian)

    solver = SQPSolver(max_iter=max_iter, verbose=verbose, **kw)
    t_wall_0 = time.perf_counter()
    result = solver.solve(prob, x0)
    t_wall = time.perf_counter() - t_wall_0

    # Restore class-level patches
    LBFGSMemory.materialise = orig_mat
    qp_cls.solve            = orig_qp

    iters = result.iterations if result.iterations > 0 else 1

    return dict(
        n=n_actual,
        iters=iters,
        status=result.status,
        wall=t_wall,
        wall_per_iter=t_wall / iters,
        t_mat=t_mat.total,
        t_qp=t_qp.total,
        t_eval=t_eval.total,
        t_other=max(0.0, t_wall - t_mat.total - t_qp.total - t_eval.total),
        calls_mat=t_mat.calls,
        calls_qp=t_qp.calls,
    )


# ---------------------------------------------------------------------------
# Alternative materialise: pre-allocated temporaries (avoids np.outer allocs)
# ---------------------------------------------------------------------------

def materialise_prealloc(self: LBFGSMemory) -> np.ndarray:
    """
    Same batched two-loop as the production materialise(), but pre-allocates
    the rank-1 temporary matrix to avoid repeated heap allocs of n×n arrays.

    At n=500 each np.outer creates a 2 MB temporary.  Pre-allocating it once
    and using in-place subtraction removes GC pressure from the hot path.

    Whether this is faster depends on allocator overhead vs cache effects
    at the specific n — this variant is here to measure that trade-off.
    """
    m = len(self._s)
    B   = self._scale * np.eye(self.n)
    if m == 0:
        return B

    rho   = np.array([1.0 / (self._y[i] @ self._s[i]) for i in range(m)])
    alpha = np.empty((m, self.n))
    tmp   = np.empty((self.n, self.n))   # reused scratch for rank-1 outer product

    for i in range(m - 1, -1, -1):
        alpha[i] = rho[i] * (self._s[i] @ B)
        np.outer(self._y[i], alpha[i], out=tmp)
        B -= tmp

    for i in range(m):
        beta = rho[i] * (self._y[i] @ B)
        np.outer(self._s[i], alpha[i] - beta, out=tmp)
        B += tmp

    return B


def benchmark_compare(n: int, max_iter: int = 100) -> dict:
    """
    Run the solver twice: once with the current materialise, once with the
    pre-allocated version.  Returns timing for both.
    """
    prob, x0 = make_rosenbrock_nlp(n)
    n_actual = prob.n

    results = {}
    for label, mat_fn in [("current", None), ("prealloc", materialise_prealloc)]:
        t_mat = _Timer()

        if mat_fn is not None:
            # Install vectorised materialise on the class temporarily
            original_mat = LBFGSMemory.materialise
            LBFGSMemory.materialise = mat_fn

        orig_mat_patch = _patch(LBFGSMemory, "materialise", t_mat)

        solver = SQPSolver(max_iter=max_iter, verbose=False)
        t0 = time.perf_counter()
        result = solver.solve(prob, x0.copy())
        wall = time.perf_counter() - t0

        LBFGSMemory.materialise = orig_mat_patch

        if mat_fn is not None:
            LBFGSMemory.materialise = original_mat

        iters = max(result.iterations, 1)
        results[label] = dict(wall=wall, t_mat=t_mat.total, iters=iters,
                              wall_per_iter=wall / iters, status=result.status)

    results["n"] = n_actual
    speedup = results["current"]["t_mat"] / max(results["prealloc"]["t_mat"], 1e-9)
    results["materialise_speedup"] = speedup
    return results


# ---------------------------------------------------------------------------
# cProfile run
# ---------------------------------------------------------------------------

def run_cprofile(n: int, max_iter: int = 100, backend: str = "auto"):
    """Run cProfile and print the top-20 cumulative-time entries."""
    prob, x0 = make_rosenbrock_nlp(n)
    solver = SQPSolver(max_iter=max_iter, verbose=False, backend=backend)

    pr = cProfile.Profile()
    pr.enable()
    solver.solve(prob, x0)
    pr.disable()

    buf = io.StringIO()
    ps = pstats.Stats(pr, stream=buf).sort_stats("cumulative")
    ps.print_stats(20)
    print(buf.getvalue())


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

_HDR = (
    f"{'n':>6}  {'iters':>5}  {'total(s)':>9}  {'ms/iter':>8}  "
    f"{'materialise':>12}  {'osqp':>8}  {'eval':>8}  {'other':>8}  status"
)
_SEP = "-" * len(_HDR)


def _pct(part, total):
    if total < 1e-12:
        return " ─── "
    return f"{100.0 * part / total:5.1f}%"


def print_row(r: dict):
    wall = r["wall"]
    print(
        f"{r['n']:>6}  {r['iters']:>5}  {wall:>9.3f}  "
        f"{r['wall_per_iter']*1000:>8.2f}  "
        f"{_pct(r['t_mat'],  wall):>12}  "
        f"{_pct(r['t_qp'],   wall):>8}  "
        f"{_pct(r['t_eval'], wall):>8}  "
        f"{_pct(r['t_other'],wall):>8}  "
        f"{r['status']}"
    )


def print_compare_row(r: dict):
    cur = r["current"]
    alt = r["prealloc"]
    print(
        f"  n={r['n']:>5}  "
        f"current  {cur['wall_per_iter']*1000:>7.2f} ms/iter  "
        f"(materialise {cur['t_mat']*1000/max(cur['iters'],1):.2f} ms)  |  "
        f"prealloc  {alt['wall_per_iter']*1000:>7.2f} ms/iter  "
        f"(materialise {alt['t_mat']*1000/max(alt['iters'],1):.2f} ms)  "
        f"→ {r['materialise_speedup']:.1f}× speedup"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes",    default="50,100,200,500",
                    help="Comma-separated list of n values (default: 50,100,200,500)")
    ap.add_argument("--max-iter", type=int, default=100,
                    help="Max SQP iterations per run (default: 100)")
    ap.add_argument("--cprofile", action="store_true",
                    help="Run cProfile on --n and print top-20 cumtime entries")
    ap.add_argument("--n",        type=int, default=200,
                    help="n for --cprofile (default: 200)")
    ap.add_argument("--compare",  action="store_true",
                    help="Compare current vs vectorised materialise")
    ap.add_argument("--backend",  default="all",
                    choices=["all", "osqp", "implicit", "woodbury"],
                    help="QP backend to profile (default: all three)")
    args = ap.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]

    if args.cprofile:
        cp_backend = args.backend if args.backend != "all" else "auto"
        print(f"\n=== cProfile  n={args.n}  max_iter={args.max_iter}  "
              f"backend={cp_backend} ===\n")
        run_cprofile(args.n, args.max_iter, backend=cp_backend)
        return

    if args.compare:
        print(f"\n=== materialise: current (batched BLAS) vs pre-alloc (avoids np.outer temps) ===\n")
        for n in sizes:
            r = benchmark_compare(n, args.max_iter)
            print_compare_row(r)
        print()
        return

    backends = (["osqp", "implicit", "woodbury"]
                if args.backend == "all" else [args.backend])

    for backend in backends:
        print(f"\n=== pyfiltersqp component timing  backend={backend}  "
              f"(max_iter={args.max_iter}) ===\n")
        print(_HDR)
        print(_SEP)
        for n in sizes:
            r = benchmark(n, args.max_iter, backend=backend)
            print_row(r)
        print(_SEP)
    print(
        "\nColumns: % of total wall time spent in each component.\n"
        "  materialise — LBFGSMemory.materialise() : O(n²·memory) Python loop\n"
        "  osqp        — QPSubproblem.solve()       : OSQP dense QP\n"
        "  eval        — all NLP callable evaluations (f, g, c, J)\n"
        "  other       — solver loop overhead (KKT check, line search, L-BFGS update)\n"
        "\nRun with --compare to see the vectorised materialise speedup.\n"
        "Run with --cprofile to see a full per-function breakdown.\n"
    )


if __name__ == "__main__":
    main()
