# Roadmap

## v0.1 — Working prototype (Python, dense, l₁ merit)

Goal: solve HS71 and all 8 MacMPEC benchmark problems via the pympcc adapter.

- [x] `NLPProblem` dataclass with validation
- [x] `SQPResult` and `IterationInfo` dataclasses
- [x] `LBFGSMemory` — store pairs, materialise dense `B_k`
- [x] `QPSubproblem` — OSQP wrapper, returns `(p, lam_eq, lam_ineq)`
- [x] `l1_merit()`, `armijo_backtrack()` — line search
- [x] `kkt_residuals()` — convergence check
- [x] `SQPSolver.solve()` — outer loop wiring everything together
- [x] `solve()` convenience function
- [x] HS test problems (HS15, HS71, HS100)
- [x] Basic test suite passes
- [x] `_FilterSQPAdapter` for pympcc
- [x] pympcc MacMPEC benchmarks pass with this backend (23/24; smoothing/gauvin xfail — filter line search v0.2 needed)

## v0.2 — Filter line search

Goal: match filterSQP quality on degenerate problems (MPCC solutions where LICQ fails).

- [x] `Filter` class — pairs `(f, h)`, dominance check, envelope update
- [x] `filter_backtrack()` in `_line_search.py` — filter-acceptance backtracking
- [x] `use_filter=True` in `SQPSolver` — opt-in filter path; l₁ merit used for feasible iterates
- [x] Second-order correction step (`QPSubproblem.solve_soc`) — reduces Maratos effect
- [x] Feasibility restoration — gradient descent on h when QP fails or filter stagnates
- [ ] gauvin/smoothing xfail: root cause is degenerate FB equality Jacobian (G=0, H≫0),
      not the line search — requires Jacobian regularisation or alternative smoothing (v0.3)

## v0.3 — Robustness and packaging

- [x] Damped BFGS (Powell damping) — Powell damping in `LBFGSMemory.update()` ensures yᵀs > 0
- [x] OSQP warm-starting between SQP iterations — OSQP instance reused via `update()` since v0.1
- [x] Sparse Jacobian support in `NLPProblem` (accept CSR/CSC, pass directly to OSQP)
- [x] `pyproject.toml` complete, `pip install pyfiltersqp` works
- [ ] Published to PyPI

## v0.4+ — Performance at large n

### Profiling findings (scripts/profile_solver.py, Rosenbrock-chain NLP)

Before vectorised materialise:
| n   | ms/iter | materialise | OSQP |
|-----|---------|-------------|------|
| 50  | 2.7     | 58%         | 33%  |
| 100 | 5.3     | 60%         | 35%  |
| 200 | 13.6    | 50%         | 48%  |
| 500 | 98      | 20%         | 80%  |

After vectorised materialise (batched two-loop via BLAS rank-1 updates):
| n   | ms/iter | materialise | OSQP |
|-----|---------|-------------|------|
| 50  | 0.9     | 11%         | 71%  |
| 100 | 2.1     | 12%         | 79%  |
| 200 | 8.2     | 11%         | 85%  |
| 500 | —       | 3%          | 97%  |

**Conclusion**: OSQP dense QP is now the bottleneck at every problem size.
The next lever is to avoid passing the full dense B and A matrices to OSQP.

### Summary of Python-level optimisations (all done)

| Optimisation | n=50 | n=100 | n=200 |
|---|---|---|---|
| Baseline (v0.1-v0.3) | 2.68 ms/iter | 5.31 ms/iter | 13.6 ms/iter |
| + vectorised `materialise()` | 0.93 ms/iter | 2.07 ms/iter | 8.2 ms/iter |
| + P/A in-place caching | **0.58 ms/iter** | **1.56 ms/iter** | **6.4 ms/iter** |
| **total speedup** | **4.6×** | **3.4×** | **2.1×** |

After all Python-level work, at n=100: OSQP ADMM is 48%, update_data_mat 15%,
materialise 15%.  The real lever now is avoiding the dense n×n B matrix altogether.

### Remaining items

- [x] Profile: L-BFGS materialise was the bottleneck at n≤200; OSQP dense QP at n≥200
- [x] Vectorised `materialise()` — batched two-loop via BLAS rank-1 (loop over memory, not n)
- [x] P/A in-place caching in `QPSubproblem` — eliminate scipy.sparse rebuild on each iter
      (removed 3.6× Python function calls, saved triu_indices + sp.vstack + COO/CSR/CSC chain)
- [x] Implicit L-BFGS QP: two custom QP solvers consume the (s, y) pairs directly,
      eliminating both `materialise` and `update_data_mat`:
      - `_implicit_qp.py` — primal-dual active-set + Schur complement using `lbfgs._matvec`
      - `_qp_admm.py`     — Woodbury-ADMM (no dense B_k); polishes via Schur on the inferred
                            active set so multipliers match OSQP `polish=True` quality
- [x] Sparse B support (subsumed by the implicit path above — B_k never materialised
      when `use_implicit_qp=True` or `use_woodbury_admm=True`).

---

## v0.5 — Compact L-BFGS (prerequisite for n > 1 000)

**The wall**: at n = 10 000, a dense B_k costs 800 MB and O(n²·m) work per iteration.
Neither materialise nor OSQP's dense P path can survive. This milestone removes the O(n²)
floor everywhere before touching the QP solver.

**Compact (outer-product) L-BFGS** (Byrd, Lu, Nocedal & Zhu, 1994):

```
B_k = σ_k I  +  Ψ_k M_k Ψ_kᵀ
Ψ_k = [Y_k,  σ_k S_k]   ∈ R^{n × 2m}   (S_k, Y_k are the stored pairs)
M_k ∈ R^{2m × 2m}                        (dense, but tiny: 2m ≤ 20)
```

- [x] `LBFGSMemory.matvec(v)` — compute B_k·v in O(n·m) via BFGS forward recursion
- [x] `LBFGSMemory.factors()` — return `(σ, Ψ, M)` for downstream QP assembly
- [x] `materialise()` updated — now returns B_k directly (no inversion in solver.py)
- [x] Unit test: `matvec` output matches `materialise @ v` for m ≤ 10, n ≤ 500
- [x] Powell damping uses `matvec(s)` = B_k s (was incorrectly using `_matvec(s)` = H_k s)

**Why first**: every subsequent milestone depends on having an O(n·m) Hessian representation.
Memory footprint drops from O(n²) → O(n·m); at n = 10 000, m = 10 that is ~1.6 MB vs 800 MB.

---

## v0.5.5 — Robustness, auto-routing, and CUTEst validation (2026-05-02)

Mid-tier work that turned a competitive-on-paper solver into one that beats IPOPT
on the standard test set.  No new architectural milestone — a packed session of
backend selection, line-search robustness, and benchmark infrastructure.

### Auto-routing

- [x] `SQPSolver(backend="auto")` (now default) — picks `implicit` or `woodbury`
      from problem shape.  Two-tier heuristic:
      * `n ≤ 50`: implicit unless `m_ineq > 2n` (saturated active set)
      * `n > 50`: implicit when `m_eq + m_ineq + #finite_bounds ≤ max(20, n/10)`
- [x] `osqp` retained as deprecated fallback (`backend="osqp"`); slower than the
      implicit / Woodbury paths at every size in our benchmarks
- [x] Backend boolean flags (`use_implicit_qp`, `use_woodbury_admm`) marked as
      backward-compat aliases for `backend=`

### Line-search robustness (Maratos + stagnation handling)

- [x] **SOC retry on l₁ merit collapse** (`solver.py`): when armijo backtracking
      drives α below 1e-10, re-solve QP with c(x+p) on the RHS to get d, accept
      x + (p + d) if its merit decreases.  Costs at most one extra `solve_soc`
      call when the line search would have failed anyway
- [x] **Mid-solve filter fallback**: when l₁ + SOC both fail, sticky-promote to
      `filter_backtrack` for the rest of the solve.  Recovered KISSING2 and the
      class of "stagnation near degenerate KKT" failures
- [x] **L-BFGS watchdog reset**: when α < 1e-2 for 3 consecutive accepted iters,
      reset memory and re-init scale from `1/‖g‖_∞`.  Breaks the
      bad-B → tiny-α → bad-y → worse-B feedback loop (HS23 forced-implicit
      converges; EXTROSNB n=1000 finds the *true* global optimum instead of a
      different local min)
- [x] **Gradient-aware H₀ init**: if `‖g(x0)‖_∞ > 10⁴`, set
      `lbfgs._scale = 1/‖g‖_∞` so the first Newton-like step has unit norm.
      Stops badly-scaled problems (e.g. MGH `variably_dimensioned`) from
      collapsing the line search at iter 0

### Implicit active-set anti-cycling

- [x] Visited-set hashing → switch to **Bland's rule** (lowest-index eligible)
      on revisit
- [x] One-step **tabu memory** under Bland mode: refuse to undo the previous
      iter's add/remove.  Period-2 cycles broken (HS23 wild-step pathology gone)

### Per-backend speedups

- [x] `LBFGSMemory._matvec_batch(V)` — single shared two-loop sweep for all
      columns of V; replaces `column_stack([_matvec(v) for v in cols])` in
      `ImplicitLBFGSQP._schur_solve` and `WoodburyADMM._polish`
- [x] WoodburyADMM dense-fast-path: when both Jacobians are dense and
      `m_constr ≤ 32`, stack `J_constr` as a plain ndarray (eliminates the
      31 % scipy.sparse overhead seen on Rosenbrock-chain)
- [x] WoodburyADMM transpose caching: compute `J_constr.T` once per `solve()`
      call, reuse across all ADMM iterations and `solve_soc`
- [x] Vectorised active-set scans in `ImplicitLBFGSQP.solve` — NumPy
      argmax-on-mask replaces Python `for k in range(n)` violation loops
      (~57 % → ~5 % of wall time on broyden_tridiagonal n=1000)

### Optional Numba JIT

- [x] `pip install pyfiltersqp[numba]` — opts into a JIT-compiled
      `_lbfgs._precompute_Bs` (BLAS-routed `np.dot`, contiguous column copies,
      `cache=True`).  Falls back gracefully to the pure-NumPy implementation
      when numba isn't installed.  Net wins: 2× on broyden_tridiagonal n=1000,
      40 % on smaller HS-sized problems.

### OSQP-path housekeeping

- [x] `polishing=False` in OSQP defaults — OSQP 1.1.x's polish module emits
      `"Polishing not needed..."` directly to fd=1 regardless of `verbose`
      and even survives runtime `os.dup2` redirection.  Implicit / Woodbury
      paths each have their own Schur-complement polish, so multiplier
      accuracy is preserved on the recommended backends.
- [x] `WoodburyADMM._polish`: after ADMM convergence, infer the active set
      from the dual and run one Schur solve via `lbfgs._matvec_batch` to
      lift multipliers to OSQP-`polish=True` quality.  Required to converge
      HS15 and similar problems where ADMM-tolerance multipliers contaminate
      the next L-BFGS y-update.

### Validation infrastructure

- [x] `tests/cutest_adapter.py` — wraps a `pycutest.CUTEstProblem` as an
      `NLPProblem` (splits cl/cu rows into eq/ineq, splits range rows)
- [x] `scripts/bench_cutest.py` — pyfsqp / IPOPT / scipy-SLSQP head-to-head
      on a curated CUTEst subset, with subprocess isolation per problem
      (CUTEst's Fortran state isn't safe across multiple `import_problem`
      calls in one process), per-problem timeout, and `SLSQP_MAX_N` skip
      (SLSQP's O(n²) inner work makes it irrelevant above n=1000)
- [x] `tests/large_scale_problems.py` — 7 scalable Python problems with
      exact Jacobians: Broyden tridiagonal, discrete BVP, extended Powell,
      variably dimensioned, eq-constrained QP, ineq-constrained QP, plus
      Rosenbrock-chain reused from `scripts/profile_solver.py`
- [x] `scripts/bench_large_scale.py` — pyfsqp vs cyipopt IPOPT on those at
      n ∈ {50, 200, 1000}
- [x] 14 Hock-Schittkowski problems wired into `tests/problems.py` (was 3),
      each parameterised across all four backends (osqp/implicit/woodbury/
      filter) — 141 tests total

### Final benchmark headlines (vs cyipopt IPOPT)

CUTEst classical deck (47 problems run, 45 ok):
* **41 / 45 wins** by 1.3× – 76× (geometric-mean speedup ≈ 4×)
* Largest gaps: ARWHEAD n=5000 (76×), EXTROSNB n=1000 (43×), HS25 (42× and
  IPOPT fails)
* 2 mutual fails: GENROSE multi-local-min, NONDQUAR flat-optimum tolerance

Large-scale Python benchmark (n=1000):
* ineq_quadratic: 12 ms vs IPOPT 370 ms (**31×**)
* rosenbrock_chain: 125 ms vs 2583 ms (**21×**)
* extended_powell: 18 ms vs 332 ms (**18×**)
* eq_quadratic: 107 ms vs 1237 ms (**12×**)

---

## v0.6 — Structured QP solve (n ~ 1 000 – 10 000)

OSQP's ADMM loop requires an explicit upper-triangular P (n*(n+1)/2 entries) and solves a
dense KKT factorisation each iteration — both scale as O(n²) or worse. The fix is to hand
OSQP a structured P, or replace OSQP with a Krylov-based solver that only needs B_k·v.

### 6a  OSQP custom KKT solver (lower-effort path)

OSQP ≥ 0.6 supports plugging in a custom linear system solver via `custom_kkt_solver`.
The KKT system OSQP needs to solve each ADMM step is:

```
[ P + ρI   Aᵀ ] [ x ]   [ rhs_x ]
[ A       -ρI ] [ y ] = [ rhs_y ]
```

With P = σI + Ψ M Ψᵀ, apply the **Woodbury identity** to factor the KKT system in O(n·m²)
(a diagonal solve + two n×2m matvecs + a 2m×2m dense solve), bypassing the n×n factorisation.

- [ ] Implement `_kkt_solver.py`: `WoodburyKKTSolver` satisfying OSQP's custom-solver API
- [ ] Wire into `QPSubproblem` via `osqp.Problem(custom_solver=WoodburyKKTSolver(...))`
- [ ] Benchmark: n = 1 000, 5 000, 10 000 — target < 100 ms/iter at n = 10 000
- [ ] Fall back to standard OSQP when n < threshold (e.g., n < 500)

### 6b  Projected-CG QP solver (higher-effort, no OSQP dependency for large n)

For purely equality-constrained or bound-only subproblems, a null-space projected CG
(Steihaug / Hestenes-Stiefel) converges in O(n·m·cg_iters) and needs only `matvec`.

- [x] `_qp_pcg.py` — `ProjectedCGQP`: active-set wrapper (mirrors ImplicitLBFGSQP
      with the same Bland + tabu anti-cycling) around a Nocedal-Wright Alg 16.2
      projected-CG inner solve.  Same `solve` / `solve_soc` interface as the
      other backends.  Selectable via `backend="pcg"`.
- [x] Inexact termination: relative residual `cg_tol` (default 1e-4 — loose
      by design, inexact-Newton style).  Inner iter cap `min(2(m_eq+50), 500)`
      bounds the cost when CG would otherwise wander.
- [x] Diagonal-of-B preconditioner experiment (rejected): tried Jacobi
      preconditioner ``M = diag(B_k)`` via ``LBFGSMemory.diag()``; consistently
      slower than no preconditioner because L-BFGS B already has its
      dominant component (σI) perfectly conditioned.  Localised rank-2
      contributions to ``diag(B)`` redistribute more than they condition.
      Method ``LBFGSMemory.diag()`` retained as a public utility for other
      contexts (v0.6a Woodbury-KKT preconditioning).
- [ ] Tighten `cg_tol` super-linearly as outer SQP converges
- [x] Auto-router heuristic: PCG when `m_eq > max(10, n/20)`,
      `n_bound < n/5`, and `m_ineq < m_eq`.  Empirical wins:
      eq_quadratic n=1000 → 53 ms (vs implicit 71, woodbury 161),
      n=3000 → 423 ms (vs 790, 1924), n=5000 → 1225 ms (vs 2346, 7073).

#### 6b.1  Block active-set + warm-start (2026-05-02)

Single-pivot active-set add/remove costs O(m_act) AS iters per QP, dominating
runtime when m_act is large.  Real bench numbers from
`ineq_constrained_quadratic(n=1000, m_ineq=200)` confirmed the bottleneck.

- [x] **Block add / block remove** in `ImplicitLBFGSQP` and `ProjectedCGQP`:
      add ALL violated inactive constraints / remove ALL sufficiently-negative-
      multiplier active constraints in one AS iter, with the existing visited-
      set cycle detector flipping to single-pivot Bland's rule on revisit.
      Cuts AS iters from O(m_act) to typically 2–3 per QP.  At n=1000,
      m_act=200: 201 AS iters / 192 ms → 2 AS iters / 5 ms (40× per-QP solve).
- [x] **Active-set warm-start across SQP iterations**: stash final active set
      at end of `solve()`, seed next `solve()` with it.  Truly fixed variables
      (lb ≈ ub) re-added on every call so the warm set stays consistent under
      bound changes.

#### 6b.2  Sparse Jacobian fast path (2026-05-02)

`_build` was densifying every active row via `J[i,:].todense()` even when J was
already sparse — measured ~5 % of `_qp_pcg.solve` cumulative time at n=10k,
m_eq=2k, and worse for PCG which calls `project()` (sparse @ vec) per CG iter.

- [x] `_build_sparse` in both backends — CSR slicing + `sp.vstack` instead
      of per-row densify; identity rows for bounds built from a single COO
      construction.
- [x] `_schur_solve` accepts sparse `A_act`; only `A_actᵀ` is densified for
      the L-BFGS batched matvec (intrinsically dense output, n × m_act).
- [x] PCG `_pcg` accepts sparse `A`: per-CG-iter `A @ v` and `Aᵀ @ v` use
      sparse matvec; the inner Cholesky on `AAᵀ` densifies once per AS iter.
- [x] Equivalence test (`test_sparse_jacobian_matches_dense`) parameterised
      across implicit and PCG backends.
- [x] Bench (n=2000, m_ineq=500, sparse-J): Implicit 0.50 s → 0.30 s (~1.7×);
      PCG 0.33 s → 0.017 s (~20×).

#### 6b.3  TV-MCP regression — rank-deficient projection (2026-05-03)

External report (`bilevel_mpcc_imaging` TV-MCP): pyfiltersqp called with PCG
backend on n=1792, m_eq=1280 sparse problem hung indefinitely with no output.
Live `PYFILTERSQP_DEBUG=1` instrumentation traced the failure:

* AS iter 5: `m_act=454`, single AS iter took **18 s** because Cholesky on
  `AAᵀ` failed (rank-deficient: m_act > n=400 after equalities + block-added
  ineqs) and the fallback `np.linalg.lstsq(AAT, rhs)` was being called *per
  CG inner iter*, costing O(m_act³) × hundreds of CG iters.

- [x] **Rank-deficient AAᵀ regulariser** (`_qp_pcg.py:_pcg`): on Cholesky
      failure, add `1e-10 · trace(AAᵀ) / m_act · I` and refactor.  Keeps every
      `project()` at O(m_act²) triangular-solve cost.  Cuts the 18 s AS iter
      to ~2 s on the same problem.

- [x] **Live debug instrumentation** behind `PYFILTERSQP_DEBUG=1` env var:
      `[PYFSQP] ENTER solve` / `[PCG] AS_iter=N m_act=… build=…ms pcg=…ms` /
      `[PCG] EXIT solve`.  Lets external callers diagnose stalls without
      forwarding `verbose=True` through multi-layer adapters.
- [x] **Per-iter live print under `verbose=True`** in `SQPSolver.solve` —
      previously only populated `result.history`.

- [x] **Cycle watchdog (item 6b.4)** — two-stage safety net in both
      `_qp_pcg.py` and `_implicit_qp.py`:
      1. **FORCE-BLAND**: when m_act > n for 3 consecutive AS iters, force
         single-pivot mode (block-add can otherwise toggle large groups of
         rows forever in the rank-deficient regime).  Persistence guard
         avoids tripping on transient block-add spikes.
      2. **Bland-window watchdog**: track m_act over the last K=30 iters
         spent in bland mode; exit AS loop when `max - min ≤ 5` (catches
         period-2 {599,600}, period-3 {544,545,546}, and any small-
         amplitude oscillation regardless of cycle length).  Returns the
         current iterate as a best-effort step; the outer SQP's line
         search rejects it if it isn't a descent direction.
      Verified: TV-MCP repro that previously hung indefinitely now exits
      cleanly within ~230 s on the first QP and continues to subsequent
      SQP iterations.  All 169 HS / unit tests still pass.
- [ ] **Constraint redundancy detection (item 6b.5)** — when m_eq + active
      ineqs exceed n, A_act has linearly dependent rows by counting alone.
      Pivoted-QR basis selection on A_act would identify the redundant rows
      and exclude them from the KKT system, eliminating the cycle root cause.
      Higher effort than the watchdog but the principled fix.
- [x] **Sparse augmented-KKT backend (item 6b.6, 2026-05-03)** —
      :class:`KKTSparseQP` in ``_qp_kkt.py``.  Solves the augmented KKT
      system::

          [ B + δI    A_actᵀ ] [ p ]   [ -g ]
          [ A_act      0     ] [ λ ] = [  b ]

      via ``scipy.sparse.linalg.splu`` on the *sparse base*

          M_base = [ (σ+δ_p) I    A_actᵀ ]
                   [ A_act        −δ_d I ]

      then applies Sherman-Morrison-Woodbury to recover the full
      L-BFGS Hessian ``B = σI + W D Wᵀ`` in ``2m + 1`` extra triangular
      solves.  Two-level regularisation: try unbiased ``δ_d = 0`` first
      (works for full-row-rank ``A_act``); fall back to IPOPT-style
      primal-dual shift ``δ_d = δ_p`` on splu failure (handles MPCC
      Scholtes-style zero-row Jacobians).  Same active-set wrapper as
      PCG: block add/remove, FORCE-BLAND when m_act > n, bland-window
      watchdog, warm-start.

      Headline numbers on the user's TV-MCP (image-bilevel MPCC,
      n=1792, m_eq=1280, m_ineq=192, sparse J, Scholtes ε-continuation):
      * Woodbury:  285 s per outer iter, never converged (dense
                   ``np.linalg.inv`` on m_constr × m_constr ≈ 1280²)
      * PCG:       hung indefinitely (rank-deficient ``A Aᵀ`` cycle)
      * **KKT:     9.1 s for the whole MCP solve loop (~7-23 ms per
                   AS iter)** — same order of magnitude as IPOPT/MA57
                   (2.9 s) using stock SciPy splu, no Fortran HSL deps.

      Opt-in via ``backend="kkt"``.  Auto-router heuristic intentionally
      not updated yet — needs cross-class benchmarks before promoting
      KKT to a default selection (existing PCG / Implicit / Woodbury
      auto-routes still cover their respective sweet spots).
- [ ] **Auto-router KKT branch** — bench KKT against PCG / Woodbury on
      eq_quadratic, ineq_quadratic, MPCC, PDE-discretised problems at
      n ∈ {500, 1000, 2000, 5000}.  Promote to auto-selection when
      ``m_eq + m_ineq > n/2`` AND constraint Jacobians sample sparse
      via a one-shot probe.

### Common to 6a and 6b

- [ ] Ensure Jacobians are always passed and stored as CSR/CSC — no silent dense paths
- [ ] Profile the full n = 10 000 stack; confirm < 5% time in non-QP code
- [ ] Add `large_n` integration test: Rosenbrock-chain NLP at n = 10 000, tol = 1e-4

---

## v0.7 — Parallel function evaluations and inexact gradients (n > 10 000)

At large n, the user's `objective`, `gradient`, `eq_constraints`, and `eq_jacobian` callables
can themselves be expensive (e.g., PDE residuals, FEM assembly). This milestone decouples
solver overhead from evaluation cost.

- [ ] **Async evaluation interface**: `NLPProblem.async_mode = True` — accept coroutines or
      concurrent.futures `Future`s for f, ∇f, c, J; solver awaits batched results
- [ ] **Parallel line search**: evaluate merit at multiple trial α values simultaneously
      (useful when each function evaluation is seconds of FEM/CFD work)
- [ ] **Finite-difference Jacobian option** (large-n only): sparse finite-difference via
      graph-coloring (scipy.optimize.approx_fprime structure detection) when exact Jacobians
      are unavailable — add `fd_jacobian=True` flag with explicit accuracy warning
- [ ] **Jacobian-free option** (very large n): replace explicit J in QP with a
      matrix-free constraint linearisation; use MINRES or LSQR for the KKT solve

---

## v0.8 — Scalability benchmarks and CUTEst validation

Before claiming n > 10 000 support, validate on standard large-scale NLP test sets.

- [x] **pycutest adapter + bench harness** (v0.5.5) — `tests/cutest_adapter.py`
      and `scripts/bench_cutest.py` cover ~47 problems from the classical deck
      with subprocess isolation and per-problem timeouts.  Comparison is against
      cyipopt IPOPT and scipy SLSQP.
- [ ] **Expand to the 50 largest CUTEst problems** (n up to 100 000) — current
      bench tops out at n = 5000.  Need to also exercise problems with sparse
      Jacobians end-to-end (CUTE deck has many).
- [ ] **Scaling table** (to add to DESIGN.md):

  | n      | ms/iter target | memory target | QP path     |
  |--------|---------------|---------------|-------------|
  | 1 000  | < 10 ms       | < 10 MB       | OSQP + Woodbury |
  | 5 000  | < 50 ms       | < 50 MB       | OSQP + Woodbury |
  | 10 000 | < 200 ms      | < 200 MB      | PCG or Woodbury |
  | 50 000 | < 2 s         | < 1 GB        | PCG + async eval |

- [ ] **Regression guard**: add `pytest -m large_n` marker; CI skips by default,
      runs on nightly / manual trigger
- [ ] **Memory profiling**: `scripts/profile_memory.py` using `tracemalloc`; assert
      peak RSS scales as O(n·m), not O(n²)
- [ ] **NONDQUAR-class tolerance issue**: when ADMM/active-set converge near a
      flat optimum, the dual residual stays at O(ρ·tol) which exceeds our fixed
      `tol_opt = 1e-6` even though the iterate is essentially optimal.  Adaptive
      tolerance scaling (relative to multiplier magnitude) would resolve.

---

## v0.9 — Distributed / HPC (n >> 100 000, optional)

- [ ] **MPI-parallel function evaluations** via `mpi4py`: domain-decomposed PDE problems
      where each MPI rank owns a slice of x and evaluates its local residual contribution
- [ ] **Distributed sparse Jacobian**: assemble J in parallel, reduce via MPI collective
- [ ] **GPU matvec for L-BFGS**: `cupy`-backed `LBFGSMemory.matvec` — beneficial when
      n > 100 000 and m = 20 (2 × n × m doubles for Ψ = 3.2 GB at n = 100 000 on GPU)
- [ ] **Block-structured QP**: exploit problem structure (e.g., ODE/DAE transcription)
      where J has banded/block-diagonal sparsity — use block KKT factorisation
