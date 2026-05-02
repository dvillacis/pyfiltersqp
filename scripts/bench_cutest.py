"""
CUTEst benchmark — pyfiltersqp (auto backend) vs cyipopt IPOPT.

Setup
-----
Set these in your shell, or let the script auto-detect them from
``/Users/davidvillacis/src/`` (the layout this project was developed against)::

    export CUTEST=/path/to/CUTEst
    export SIFDECODE=/path/to/SIFDecode
    export ARCHDEFS=/path/to/archdefs
    export MASTSIF=$CUTEST/sif
    export MYARCH=mac.osx.gfo            # or whatever ``ls $CUTEST/objects`` shows
    export PYCUTEST_CACHE=/tmp/pycutest_cache  # any writable dir

The script pre-builds each requested problem in a separate subprocess
(pycutest can decode at most ONE new problem per Python session — a known
SIFDecode limitation).  Already-cached problems load instantly.

Usage
-----
    uv run python scripts/bench_cutest.py
    uv run python scripts/bench_cutest.py --problems ROSENBR,HS11
    uv run python scripts/bench_cutest.py --max-n 1000   # skip larger
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# Ensure tests/ is importable for the adapter
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ---------------------------------------------------------------------------
# Default problem list (the SIF deck that ships with CUTEst by default).
# Add more once the user installs the full MASTSIF deck.
# ---------------------------------------------------------------------------

DEFAULT_PROBLEMS = [
    # --- Hock-Schittkowski classics ---
    "HS1", "HS2", "HS3", "HS4", "HS5", "HS6", "HS7", "HS8",
    "HS9", "HS10", "HS11", "HS12", "HS14", "HS15", "HS16",
    "HS22", "HS23", "HS24", "HS25", "HS26", "HS27", "HS28",
    "HS35", "HS36", "HS37", "HS38",
    "HS43",                  # Rosen-Suzuki
    "HS44",
    "HS65",                  # mixed
    "HS71",                  # SQP demo
    "HS76",                  # QP
    "HS78",                  # eq-constrained
    "HS100",                 # 7-var ineq
    "HS104",                 # 8-var
    "HS113",                 # 10-var

    # --- Unconstrained classics ---
    "ROSENBR",               # n=2
    "ROSENMMX",              # n=5  mini-max
    "EXTROSNB",              # n=1000 (default size in SIF)
    "SISSER",                # n=2
    "POWELLBS",              # singular Hessian
    "GENROSE",               # generalised Rosenbrock
    "TQUARTIC",              # quartic

    # --- More-Garbow-Hillstrom (large-scale unconstrained) ---
    "NONDIA",                # n=5000 nondiagonal quartic
    "NONDQUAR",
    "ARWHEAD",               # n=5000 arrow head

    # --- Constrained mid/large ---
    "ALLINITC",              # n=3 mixed
    "DIPIGRI",               # HS100 variant
    "MARATOS",               # the Maratos problem itself
    "KISSING2",              # n=94 m_ineq=625
]


def auto_set_env() -> None:
    """Populate CUTEst env vars from ~/src/{CUTEst,SIFDecode,archdefs} when unset."""
    src = Path.home() / "src"
    defaults = {
        "CUTEST":    src / "CUTEst",
        "SIFDECODE": src / "SIFDecode",
        "ARCHDEFS":  src / "archdefs",
    }
    for k, v in defaults.items():
        if not os.environ.get(k) and v.exists():
            os.environ[k] = str(v)

    cutest_dir = os.environ.get("CUTEST")
    if cutest_dir:
        if not os.environ.get("MASTSIF"):
            # Prefer the full classical SIF deck (~/src/sif, ~1500 problems
            # cloned from bitbucket.org/optrove/sif) over the small demo set
            # that ships with CUTEst (~13 problems in $CUTEST/sif).
            full_sif = Path.home() / "src" / "sif"
            demo_sif = Path(cutest_dir) / "sif"
            for candidate in (full_sif, demo_sif):
                if candidate.exists() and any(candidate.glob("*.SIF")):
                    os.environ["MASTSIF"] = str(candidate)
                    break
        if not os.environ.get("MYARCH"):
            objects = Path(cutest_dir) / "objects"
            if objects.exists():
                arches = sorted(p.name for p in objects.iterdir() if p.is_dir())
                if arches:
                    os.environ["MYARCH"] = arches[0]
    if not os.environ.get("PYCUTEST_CACHE"):
        cache = Path("/tmp") / "pycutest_cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ["PYCUTEST_CACHE"] = str(cache)


# ---------------------------------------------------------------------------
# Pre-build phase: one new problem per subprocess (pycutest limitation)
# ---------------------------------------------------------------------------

def prebuild_problem(name: str) -> bool:
    """
    Decode and compile a CUTEst problem in a subprocess so each new
    SIFDecode invocation is isolated.  Returns True on success.
    """
    cmd = [
        sys.executable, "-c",
        f"import pycutest; pycutest.import_problem({name!r})",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, env=os.environ.copy())
    return res.returncode == 0


def ensure_built(names: list[str]) -> list[str]:
    """Pre-build any not-yet-cached problem.  Returns the list still missing."""
    cache = Path(os.environ["PYCUTEST_CACHE"]) / "pycutest_cache_holder"
    missing = []
    for nm in names:
        if (cache / nm).exists():
            continue
        print(f"  building {nm} ...", end="", flush=True)
        ok = prebuild_problem(nm)
        print(" ok" if ok else " FAILED")
        if not ok:
            missing.append(nm)
    return missing


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def kkt_inf(prob, x: np.ndarray) -> float:
    feas = [0.0]
    if prob.m_eq > 0:
        feas.append(float(np.max(np.abs(prob.eq_constraints(x)))))
    if prob.m_ineq > 0:
        feas.append(float(np.max(np.maximum(0.0, prob.ineq_constraints(x)))))
    feas.append(float(np.max(np.maximum(0.0, prob.xl - x))))
    feas.append(float(np.max(np.maximum(0.0, x - prob.xu))))
    return float(max(feas))


def run_pyfsqp(prob, x0):
    from pyfiltersqp import solve
    t0 = time.perf_counter()
    res = solve(prob, x0.copy(), max_iter=500)
    dt = time.perf_counter() - t0
    return dict(time=dt, iters=res.iterations, obj=res.obj,
                kkt=kkt_inf(prob, res.x), success=res.success)


def run_ipopt(prob, x0):
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
                             options={"print_level": 0, "tol": 1e-8, "max_iter": 500})
    except Exception as e:
        return dict(time=time.perf_counter() - t0, iters=0,
                    obj=float("nan"), kkt=float("nan"),
                    success=False, error=str(e))
    dt = time.perf_counter() - t0
    x = np.asarray(res.x)
    return dict(time=dt, iters=int(getattr(res, "nit", 0)),
                obj=float(res.fun), kkt=kkt_inf(prob, x),
                success=bool(res.success))


_WORKER_SCRIPT = r"""
import json, sys, time, numpy as np
sys.path.insert(0, sys.argv[2])   # project root for `from tests.cutest_adapter import wrap`

import pycutest
from tests.cutest_adapter import wrap

def kkt_inf(prob, x):
    f = [0.0]
    if prob.m_eq   > 0: f.append(float(np.max(np.abs(prob.eq_constraints(x)))))
    if prob.m_ineq > 0: f.append(float(np.max(np.maximum(0, prob.ineq_constraints(x)))))
    f.append(float(np.max(np.maximum(0.0, prob.xl - x))))
    f.append(float(np.max(np.maximum(0.0, x - prob.xu))))
    return float(max(f))

def run_pyfsqp(prob, x0):
    from pyfiltersqp import solve
    # Warm-up call: pays numba JIT compile + module init (~150 ms cold) so the
    # timed run measures actual solver work, not Python/numba startup.
    solve(prob, x0.copy(), max_iter=500)
    t0 = time.perf_counter()
    res = solve(prob, x0.copy(), max_iter=500)
    dt = time.perf_counter() - t0
    return dict(time=dt, iters=res.iterations, obj=float(res.obj),
                kkt=kkt_inf(prob, res.x), success=bool(res.success))

def run_ipopt(prob, x0):
    from cyipopt import minimize_ipopt
    bounds = list(zip(prob.xl, prob.xu))
    constraints = []
    if prob.m_eq > 0:
        constraints.append(dict(type="eq", fun=prob.eq_constraints, jac=prob.eq_jacobian))
    if prob.m_ineq > 0:
        constraints.append(dict(type="ineq",
                                fun=lambda x: -prob.ineq_constraints(x),
                                jac=lambda x: -prob.ineq_jacobian(x)))
    try:
        minimize_ipopt(prob.objective, x0.copy(), jac=prob.gradient,
                       bounds=bounds, constraints=constraints,
                       options={"print_level":0,"tol":1e-8,"max_iter":500})
    except Exception:
        pass
    t0 = time.perf_counter()
    try:
        res = minimize_ipopt(prob.objective, x0.copy(), jac=prob.gradient,
                             bounds=bounds, constraints=constraints,
                             options={"print_level":0,"tol":1e-8,"max_iter":500})
    except Exception as e:
        return dict(time=time.perf_counter()-t0, iters=0, obj=float("nan"),
                    kkt=float("nan"), success=False, error=str(e)[:100])
    dt = time.perf_counter() - t0
    x = np.asarray(res.x)
    return dict(time=dt, iters=int(getattr(res,"nit",0)),
                obj=float(res.fun), kkt=kkt_inf(prob, x),
                success=bool(res.success))

def run_slsqp(prob, x0):
    from scipy.optimize import minimize
    bounds = list(zip(prob.xl, prob.xu))
    constraints = []
    if prob.m_eq > 0:
        constraints.append(dict(type="eq",
                                fun=prob.eq_constraints,
                                jac=prob.eq_jacobian))
    if prob.m_ineq > 0:
        # SLSQP convention: c(x) >= 0  → use -c_ineq
        constraints.append(dict(type="ineq",
                                fun=lambda x: -prob.ineq_constraints(x),
                                jac=lambda x: -prob.ineq_jacobian(x)))
    # Warm-up
    try:
        minimize(prob.objective, x0.copy(), jac=prob.gradient,
                 method="SLSQP", bounds=bounds, constraints=constraints,
                 options=dict(maxiter=500, ftol=1e-8))
    except Exception:
        pass
    t0 = time.perf_counter()
    try:
        res = minimize(prob.objective, x0.copy(), jac=prob.gradient,
                       method="SLSQP", bounds=bounds, constraints=constraints,
                       options=dict(maxiter=500, ftol=1e-8))
    except Exception as e:
        return dict(time=time.perf_counter()-t0, iters=0, obj=float("nan"),
                    kkt=float("nan"), success=False, error=str(e)[:100])
    dt = time.perf_counter() - t0
    return dict(time=dt, iters=int(getattr(res,"nit",0)),
                obj=float(res.fun), kkt=kkt_inf(prob, np.asarray(res.x)),
                success=bool(res.success))

# Skip SLSQP for large-n problems where its O(n²) inner work dominates the
# entire bench wall budget — at n=5000 a single SLSQP call already exceeds
# 30 s on smooth quartic problems (TQUARTIC, NONDQUAR), and we run two
# (warm + timed).  The pyfsqp ↔ IPOPT comparison is the meaningful one
# above this threshold; SLSQP entries become "ok but irrelevantly slow".
SLSQP_MAX_N = 1000

name = sys.argv[1]
cu = pycutest.import_problem(name)
prob, x0 = wrap(cu)
out = dict(name=name, n=prob.n, m_eq=prob.m_eq, m_ineq=prob.m_ineq)
out["pyfsqp"] = run_pyfsqp(prob, x0)
out["ipopt"]  = run_ipopt(prob, x0)
if prob.n <= SLSQP_MAX_N:
    out["slsqp"]  = run_slsqp(prob, x0)
else:
    out["slsqp"]  = dict(time=None, iters=0, obj=float("nan"),
                         kkt=float("nan"), success=None,
                         skipped=f"n={prob.n} > SLSQP_MAX_N={SLSQP_MAX_N}")
print("__BENCH_JSON__" + json.dumps(out))
"""


def run_problem_in_subprocess(name: str, timeout_s: float = 120.0) -> dict | None:
    """One subprocess per problem so CUTEst's Fortran state is isolated.
    Wall-clock timeout protects against pathological non-terminations
    (we've seen single problems run > 1 hour on the demo deck)."""
    project_root = str(Path(__file__).resolve().parent.parent)
    cmd = [sys.executable, "-c", _WORKER_SCRIPT, name, project_root]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True,
                             env=os.environ.copy(), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return dict(name=name, error=f"timeout after {timeout_s:.0f}s")
    if res.returncode != 0:
        return dict(name=name, error=res.stderr.splitlines()[-1] if res.stderr else "no output")
    import json
    for line in res.stdout.splitlines():
        if line.startswith("__BENCH_JSON__"):
            return json.loads(line.removeprefix("__BENCH_JSON__"))
    return dict(name=name, error="no JSON in subprocess output")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problems", default=",".join(DEFAULT_PROBLEMS),
                    help="comma-separated CUTEst names")
    ap.add_argument("--max-n", type=int, default=None,
                    help="skip problems with n above this")
    ap.add_argument("--timeout", type=float, default=120.0,
                    help="per-problem subprocess timeout in seconds (default 120)")
    args = ap.parse_args()

    auto_set_env()
    for v in ("CUTEST", "SIFDECODE", "MASTSIF", "MYARCH", "PYCUTEST_CACHE"):
        if not os.environ.get(v):
            print(f"FATAL: env var {v} not set and could not be auto-detected.")
            sys.exit(2)

    names = [s.strip() for s in args.problems.split(",")]

    print("\n=== Pre-building problems (cached if already done) ===")
    missing = ensure_built(names)
    if missing:
        print(f"\n[skip] could not build: {missing}")
        names = [n for n in names if n not in missing]

    hdr = (f"{'name':<10} {'n':>5} {'eq':>3} {'iq':>4}  "
           f"{'pyfsqp':>9}  {'IPOPT':>9}  {'SLSQP':>9}   "
           f"{'vs IPOPT':>9}  {'gap':>9}   status")
    print("\n" + hdr)
    print("-" * len(hdr))

    def _fmt_t(r):
        if r is None or r.get("time") is None:
            return "    -    "
        return f"{r['time']*1000:>9.1f}"

    def _ok(r):
        if r is None: return "—"
        if r.get("skipped"): return "skip"
        return "ok" if r.get("success") else "F"

    for nm in names:
        out = run_problem_in_subprocess(nm, timeout_s=args.timeout)
        if out is None or "error" in out:
            print(f"{nm:<10}  ERROR: {(out or {}).get('error', 'unknown')[:80]}")
            continue
        if args.max_n is not None and out["n"] > args.max_n:
            print(f"{nm:<10} {out['n']:>5}  [skip — n > {args.max_n}]")
            continue

        rp, ri, rs = out["pyfsqp"], out["ipopt"], out.get("slsqp")
        speedup = (ri["time"] / rp["time"]) if (rp["time"] > 0 and ri["time"] > 0) else float("nan")
        if np.isfinite(rp["obj"]) and np.isfinite(ri["obj"]):
            gap = abs(rp["obj"] - ri["obj"]) / max(1.0, abs(ri["obj"]))
        else:
            gap = float("nan")
        status = f"pyfsqp={_ok(rp)} ipopt={_ok(ri)} slsqp={_ok(rs)}"
        print(f"{nm:<10} {out['n']:>5} {out['m_eq']:>3} {out['m_ineq']:>4}  "
              f"{_fmt_t(rp)}  {_fmt_t(ri)}  {_fmt_t(rs)}   "
              f"{speedup:>8.1f}× {gap:>9.2e}   {status}")


if __name__ == "__main__":
    main()
