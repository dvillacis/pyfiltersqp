# Changelog

All notable changes to **pyfiltersqp** are documented here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — large-n equality-heavy backend

- **`ProjectedCGQP`** (`backend="pcg"`): projected-CG inner solve
  (Nocedal-Wright Alg 16.2) wrapped by the same primal-dual active-set
  loop as `ImplicitLBFGSQP` (Bland's rule, one-step tabu). Inexact-Newton
  style: loose `cg_tol=1e-4` and a hard `cg_max_iter` cap so a single SQP
  step is always bounded. Empirically wins by 2–6× on equality-heavy
  large-n problems (eq_quadratic at n=3000–5000) where direct Schur
  factorisation becomes the bottleneck. Exported as
  `pyfiltersqp.ProjectedCGQP`.
- **Auto-router now picks PCG** for equality-heavy problems at n > 50.
  Heuristic: `m_eq > max(10, n/20)` AND `n_bound < n/5` AND `m_ineq < m_eq`.
  No user opt-in required — `solve(...)` chooses automatically.
- **`LBFGSMemory.diag()`** — returns `diag(B_k)` in O(n·m).  Public
  utility intended for preconditioning structured KKT solvers; matches
  `np.diag(materialise())` to machine precision.

### Investigated and rejected

- **Jacobi preconditioner for PCG**: tried using `LBFGSMemory.diag()` as
  `M⁻¹` in projected CG; consistently slower than no preconditioner across
  every problem class.  L-BFGS B has σI as its dominant well-conditioned
  base; localised rank-2 contributions to `diag(B)` redistribute more
  than they condition, increasing CG iteration count.  PCG runs
  unpreconditioned by design.

## [0.3.0] — 2026-05-02

First public release on PyPI. The solver now beats cyipopt IPOPT on most of
the classical CUTEst deck (1.3× – 70× faster on 41/45 successful problems).

### Added — auto-routing & robustness

- **`SQPSolver(backend="auto")`** picks `implicit` or `woodbury` from problem
  shape (n, m_eq, m_ineq, finite bounds). Two-tier heuristic validated on
  large-scale benchmarks. The boolean flags `use_implicit_qp` /
  `use_woodbury_admm` remain as backward-compat aliases.
- **Mid-solve filter fallback**: when l₁-merit + SOC retry both fail, the
  solver sticky-promotes to `filter_backtrack` for the rest of the run.
  Recovers KISSING2 and stagnation-near-degenerate-KKT cases.
- **L-BFGS watchdog reset**: when α < 1e-2 for 3 consecutive accepted iters,
  reset memory and re-init scale from `1/‖g‖_∞`. Breaks bad-B → tiny-α →
  bad-y feedback loops; secondary effect: EXTROSNB n=1000 now finds the
  true global optimum (objective gap 64 → 1.4e-7).
- **Gradient-aware H₀ initialisation**: for badly-scaled inputs
  (`‖g(x0)‖_∞ > 1e4`) the first L-BFGS Hessian is scaled to give a
  unit-norm initial step.
- **Implicit active-set anti-cycling**: visited-set hashing → Bland's rule
  fallback, plus one-step tabu memory under Bland mode (refuses to undo
  the previous iter's add/remove). Eliminates the wild-step / negative-
  multiplier failure mode on degenerate problems like CUTEst HS23.

### Added — performance

- **`LBFGSMemory._matvec_batch(V)`**: single shared two-loop sweep applies
  `B⁻¹` to all columns of V at once. Used by `ImplicitLBFGSQP._schur_solve`
  and `WoodburyADMM._polish`.
- **WoodburyADMM dense fast-path**: when both Jacobians are dense and
  `m_constr ≤ 32`, stack `J_constr` as a plain ndarray. Eliminates the
  31 % scipy.sparse construction overhead on Rosenbrock-chain.
- **WoodburyADMM transpose caching**: `J_constr.T` computed once per
  `solve()` call, reused across all ADMM iterations and `solve_soc`.
- **Vectorised active-set scans** in `ImplicitLBFGSQP.solve` — NumPy
  argmax-on-mask replaces Python `for k in range(n)` violation loops.
  At n=1000 this is ~57 % → ~5 % of wall time.
- **Optional Numba JIT** via `pip install "pyfiltersqp[numba]"`. JIT-
  compiles `_lbfgs._precompute_Bs` (BLAS-routed `np.dot`, contiguous
  column copies, `cache=True`). Falls back gracefully when not installed.
  Net wins: 2× on broyden_tridiagonal n=1000, ~40 % on smaller HS-sized.

### Added — benchmarks & validation

- **CUTEst integration**: `tests/cutest_adapter.py` wraps a
  `pycutest.CUTEstProblem` as `NLPProblem` (splits range constraints,
  preserves bounds). `scripts/bench_cutest.py` is a head-to-head harness
  for pyfsqp / IPOPT / scipy SLSQP with subprocess isolation per problem
  and per-problem timeout. Optional install: `pip install "pyfiltersqp[cutest]"`.
- **Large-scale Python problems**: `tests/large_scale_problems.py` ships
  seven scalable NLPs with exact Jacobians (Broyden tridiagonal, discrete
  BVP, extended Powell singular, variably dimensioned, eq/ineq quadratics,
  Rosenbrock-chain).
- **`scripts/bench_large_scale.py`** runs them at n ∈ {50, 200, 1000} vs IPOPT.
- **Hock-Schittkowski test set** expanded from 3 to 14 problems — every
  one runs through OSQP, implicit, Woodbury, and filter modes. 141 unit
  tests total.

### Changed

- **OSQP path is now deprecated**. Selectable via `backend="osqp"` and
  retained for fallback, but slower than `implicit` / `woodbury` at every
  problem size in our benchmarks (≥ 10× slower at n ≥ 1000).
- **OSQP `polishing=False`** by default — OSQP 1.1.x's polish module
  emits status text directly to fd=1 regardless of `verbose`. The
  implicit and Woodbury paths each have their own Schur-complement
  polish, so multiplier accuracy is preserved on the recommended backends.
- **`LBFGSMemory._precompute_Bs`** now returns `(ndarray (n, m), beta)`
  instead of `(list[ndarray], beta)`. Internal API; callers updated.

### Fixed

- **WoodburyADMM polish step**: after ADMM convergence, the inferred
  active set drives one Schur solve via `lbfgs._matvec_batch` to lift
  multipliers to OSQP-`polish=True` quality. Required to converge HS15
  and similar problems where ADMM-tolerance multipliers contaminate the
  next L-BFGS y-update.
- **`SQPSolver` initial step on badly-scaled problems**: previously the
  default H₀ = I gave a Newton step of magnitude ‖g‖, collapsing the
  line search at iter 0 on MGH `variably_dimensioned` (where
  ‖g(x0)‖ ≈ 10¹¹). Now solves in 1 iter.

## [0.2.0]

### Added

- Filter line search: `Filter` class, `filter_backtrack`, second-order
  correction step, feasibility restoration on filter stagnation. Used
  on infeasible iterates only — feasible iterates fall through to the
  more-monotone l₁ merit.

## [0.1.0]

### Added

- `NLPProblem` dataclass (validated against the user-supplied callables).
- `SQPSolver` with L-BFGS, Powell damping, l₁-merit Armijo backtracking,
  KKT residual convergence test, OSQP-based QP subproblem.
- Hock-Schittkowski test problems (HS15, HS71, HS100).

[0.3.0]: https://github.com/dvillacis/pyfiltersqp/releases/tag/v0.3.0
[0.2.0]: https://github.com/dvillacis/pyfiltersqp/releases/tag/v0.2.0
[0.1.0]: https://github.com/dvillacis/pyfiltersqp/releases/tag/v0.1.0
