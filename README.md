# pyfiltersqp

[![PyPI](https://img.shields.io/pypi/v/pyfiltersqp.svg)](https://pypi.org/project/pyfiltersqp/)
[![Python](https://img.shields.io/pypi/pyversions/pyfiltersqp.svg)](https://pypi.org/project/pyfiltersqp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/davidvillacis/pyfiltersqp/actions/workflows/ci.yml/badge.svg)](https://github.com/davidvillacis/pyfiltersqp/actions/workflows/ci.yml)

Open-source SQP (Sequential Quadratic Programming) solver in Python with
L-BFGS Hessian approximation, l₁-merit and filter line searches, and
first-class warm-starting. Faster than IPOPT on most CUTEst problems we've
benchmarked.

## Why

The open-source NLP ecosystem is dominated by interior-point methods
(IPOPT, ECOS, Clarabel). For SQP there's no modern, MIT-licensed,
Python-friendly option with sparse Jacobians and warm-starting baked in.
This fills that gap.

## Headline benchmarks

vs cyipopt IPOPT on the classical CUTEst deck (47 problems, n up to 5000):

| problem               | n     | pyfiltersqp | IPOPT     | speedup |
|-----------------------|-------|-------------|-----------|---------|
| ARWHEAD               | 5000  | 1.9 ms      | 132 ms    | **70×** |
| EXTROSNB              | 1000  | 9.4 ms      | 402 ms    | **43×** |
| HS25                  | 3     | 0.1 ms      | (fails)   | wins outright |
| TQUARTIC              | 5000  | 23 ms       | 277 ms    | 12×     |
| ROSENBR               | 2     | 3.4 ms      | 35 ms     | 10×     |
| HS71                  | 4     | 1.3 ms      | 4.9 ms    | 3.7×    |

41 of 45 successfully-solved problems beat IPOPT, by 1.3× – 70×.
See `scripts/bench_cutest.py` for the harness.

## Install

```bash
pip install pyfiltersqp                 # core
pip install "pyfiltersqp[numba]"        # +Numba JIT (2× faster L-BFGS recursion)
pip install "pyfiltersqp[dev]"          # +pytest for development
```

Requires Python ≥ 3.11. Core deps: `numpy`, `scipy`, `osqp`.

## Quick start

```python
import numpy as np
from pyfiltersqp import NLPProblem, solve

# HS71 — the classic 4-var SQP demo problem.
problem = NLPProblem(
    n=4, m_eq=1, m_ineq=1,
    xl=np.ones(4), xu=5 * np.ones(4),
    objective = lambda x: x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2],
    gradient  = lambda x: np.array([
        x[3]*(2*x[0]+x[1]+x[2]),
        x[0]*x[3],
        x[0]*x[3] + 1,
        x[0]*(x[0]+x[1]+x[2]),
    ]),
    eq_constraints = lambda x: np.array([x[0]**2+x[1]**2+x[2]**2+x[3]**2 - 40]),
    eq_jacobian    = lambda x: np.array([[2*x[0], 2*x[1], 2*x[2], 2*x[3]]]),
    ineq_constraints = lambda x: np.array([25 - x[0]*x[1]*x[2]*x[3]]),
    ineq_jacobian    = lambda x: np.array([[
        -x[1]*x[2]*x[3], -x[0]*x[2]*x[3],
        -x[0]*x[1]*x[3], -x[0]*x[1]*x[2],
    ]]),
)

result = solve(problem, x0=np.array([1., 5., 5., 1.]))
print(result.success, result.obj)        # True, 17.014017...
```

## Algorithm

- **Outer**: Sequential Quadratic Programming with l₁-merit Armijo line search;
  filter line search opt-in (Fletcher & Leyffer, 2002) for degenerate problems.
- **Hessian**: L-BFGS with Powell damping. Optional Numba JIT for the forward
  recursion. Compact (Byrd-Lu-Nocedal-Zhu) representation under the hood.
- **QP backend** (auto-selected by problem shape):
  - `ImplicitLBFGSQP` — primal-dual active-set + Schur complement, applies
    `B⁻¹` via the L-BFGS two-loop recursion in O(n·m) without ever forming
    the dense `n×n` Hessian. Best for small-to-medium n with few active
    constraints.
  - `WoodburyADMM` — Woodbury-identity ADMM on the compact L-BFGS factors,
    polished via a Schur solve on the inferred active set. Cost per ADMM
    step is O(n·m) and independent of the active set. Best for large n
    or many constraints.
  - `OSQP` — the original v0.1 path. Retained as a deprecated fallback;
    every benchmark we've run is faster on the implicit / Woodbury paths.
- **Robustness**: second-order correction on Maratos-class line search
  collapse, mid-solve fallback to filter on stagnation, anti-cycling
  (Bland's rule + one-step tabu) in active-set, watchdog reset of L-BFGS
  on tiny-α streaks.

See `DESIGN.md` for the full algorithm specification.

## Warm-starting

Designed for iterative MPCC strategies and homotopy methods where the
constraint set changes slightly between solves:

```python
result1 = solve(problem_v1, x0)
result2 = solve(problem_v2,
                x0=result1.x,
                lam_eq0=result1.lam_eq,
                lam_ineq0=result1.lam_ineq)
```

## Roadmap

See [`ROADMAP.md`](ROADMAP.md) and [`CHANGELOG.md`](CHANGELOG.md). The next
real-engineering items are PyPI publish (this commit), Woodbury custom KKT
solver inside OSQP for the n > 1000 range, and a projected-CG QP path for
n > 10 000.

## References

- Nocedal & Wright, *Numerical Optimization*, 2nd ed. (2006) — chapters 18 (SQP), 7 (L-BFGS)
- Fletcher & Leyffer (2002), "Nonlinear programming without a penalty function" — filter line search
- Byrd, Lu, Nocedal & Zhu (1994), "Representations of quasi-Newton matrices and their use in limited memory methods" — compact L-BFGS
- Powell (1978), "A fast algorithm for nonlinearly constrained optimization calculations" — damped BFGS
- Stellato et al. (2020), OSQP

## License

MIT — see [`LICENSE`](LICENSE).
