# pyfiltersqp — Design Document

> **Status**: Pre-implementation draft. All design decisions are open for revision before coding starts.
> Written as a companion to `pympcc`. Return here when ready to implement.

---

## 1. Motivation

The open-source NLP solver ecosystem is dominated by interior-point methods (IPOPT, ECOS, Clarabel).
For SQP there is no modern, open-source, Python-friendly option with:

- Warm-starting (essential for iterative MPCC strategies)
- Sparse Jacobian support
- Filter line search (better behaviour near degenerate MPCC solutions where LICQ fails)
- A clean Python interface that does not require a commercial license or a system-level binary

The closest prior art is `filterSQP` (Fletcher & Leyffer, 2002), which remains the reference
SQP implementation for MPCC research but has never had a maintained Python package.

**Primary consumer**: `pympcc` — the iterative strategies (Scholtes, Smoothing, LinFukushima)
need an NLP backend that warm-starts efficiently. IPOPT's warm-start is bolted on and fragile;
SQP warm-starting is architecturally natural.

---

## 2. Problem Form

The solver handles standard NLPs of the form:

```
min   f(x)
s.t.  c_eq(x)  = 0          (m_eq equality constraints)
      c_ineq(x) ≤ 0         (m_ineq inequality constraints)
      xl ≤ x ≤ xu            (variable bounds)
```

Bounds are handled natively as variable bounds in the QP subproblem,
NOT by converting them to inequality constraints.

The caller supplies Python callables for `f`, `∇f`, `c_eq`, `J_eq`, `c_ineq`, `J_ineq`.
Jacobians are required (no finite-difference fallback in v0.1 — callers must supply them).
Hessians are NOT required; L-BFGS approximates the Lagrangian Hessian.

---

## 3. Algorithm: SQP with L-BFGS and l₁ Merit Function

### 3.1 Outer loop

```
Initialise: x = x0, λ_eq = λ0_eq, λ_ineq = λ0_ineq (zeros if not warm-started)
            B = I  (L-BFGS Hessian approximation, identity on cold start)
            μ = μ0 (merit penalty parameter)

for k = 0, 1, 2, ..., max_iter:

    1. Evaluate f_k, g_k, c_eq_k, J_eq_k, c_ineq_k, J_ineq_k at x_k

    2. Check KKT conditions — if satisfied within tol, STOP (success)

    3. Solve QP subproblem (Section 3.2) for step p_k and new multipliers λ_k+1

    4. Update merit penalty μ (Section 3.3)

    5. Line search along p_k using l₁ merit function (Section 3.4)
       → accept step size α_k ∈ (0, 1]

    6. x_{k+1} = x_k + α_k * p_k

    7. Update L-BFGS (Section 3.5) with s_k = x_{k+1} - x_k and
       y_k = ∇_x L(x_{k+1}, λ_{k+1}) - ∇_x L(x_k, λ_{k+1})

    8. λ_eq = λ_{k+1}_eq,  λ_ineq = λ_{k+1}_ineq

STOP: max_iter reached (failure)
```

### 3.2 QP Subproblem

At each iteration solve:

```
min   (1/2) pᵀ B_k p + g_kᵀ p
s.t.  J_eq_k  p = -c_eq_k           (linearised equalities)
      J_ineq_k p ≤ -c_ineq_k        (linearised inequalities)
      xl - x_k ≤ p ≤ xu - x_k      (bound constraints on step)
```

This is a convex QP when `B_k` is positive definite (L-BFGS guarantees this).
Solved via **OSQP** (MIT license).

**OSQP reformulation** (OSQP solves `min (1/2)xᵀPx + qᵀx  s.t.  l ≤ Ax ≤ u`):

```
P  = B_k
q  = g_k
A  = [ J_eq_k   ]    l = [ -c_eq_k        ]    u = [ -c_eq_k   ]
     [ J_ineq_k ]        [ -inf * ones     ]        [ -c_ineq_k ]
     [ I        ]        [ xl - x_k        ]        [ xu - x_k  ]
```

The QP multipliers from OSQP map directly to SQP multipliers:
- `λ_eq  = osqp_y[:m_eq]`
- `λ_ineq = osqp_y[m_eq:m_eq+m_ineq]`  (should be ≥ 0 at solution)

**Infeasible QP subproblems**: When OSQP returns infeasible, fall back to a feasibility
restoration step (minimise `||c_eq||² + ||max(0,c_ineq)||²`) before retrying.
Implement this only after the basic loop works.

**Warm-starting the QP**: OSQP supports warm-starting its primal/dual iterates.
Pass the previous QP solution as `x_warm`, `y_warm` to OSQP at each SQP iteration.
This cuts QP solve time significantly on subsequent iterations.

### 3.3 Merit Penalty Update

Use the non-smooth l₁ merit function:

```
φ_μ(x) = f(x) + μ * ( ||c_eq(x)||₁ + ||max(0, c_ineq(x))||₁ )
```

Update `μ` whenever the current value is too small to make `p_k` a descent direction:

```
if μ < (g_kᵀ p_k + (1/2) p_kᵀ B_k p_k) / ((1-σ) * (||c_eq_k||₁ + ||max(0,c_ineq_k)||₁)):
    μ = <right-hand side> * 1.1   # inflate by 10%
```

where `σ ∈ (0,1)` is a small constant (default 0.01). `μ` only increases, never decreases.

### 3.4 Line Search

Backtracking Armijo line search on the l₁ merit function:

```
α = 1
while φ_μ(x_k + α*p_k) > φ_μ(x_k) + η*α*D_φ_k:
    α *= ρ     # default ρ = 0.5, η = 1e-4

# D_φ_k = directional derivative of φ_μ at x_k along p_k
D_φ_k = g_kᵀ p_k - μ * (||c_eq_k||₁ + ||max(0,c_ineq_k)||₁)
```

Parameters: `max_ls_iter = 30`, `rho = 0.5`, `eta = 1e-4`.

**v0.2**: Replace with a filter — pairs `(f(x), h(x))` where `h` is the constraint violation.
A step is accepted if it is not dominated by any entry in the filter.
The filter avoids the Maratos effect and is crucial for good behaviour on MPCC solutions.
See Fletcher & Leyffer (2002) for the filter mechanism.

### 3.5 L-BFGS Hessian Update

Maintain the last `m` (default 10) vector pairs `{(s_i, y_i)}`.

```
s_k = x_{k+1} - x_k
y_k = (∇f(x_{k+1}) + J_eq(x_{k+1})ᵀ λ_eq + J_ineq(x_{k+1})ᵀ λ_ineq)
    - (∇f(x_k)     + J_eq(x_k)ᵀ     λ_eq + J_ineq(x_k)ᵀ     λ_ineq)
```

**Damped BFGS** (Powell damping) to guarantee `y_kᵀ s_k > 0`:

```
if y_kᵀ s_k >= 0.2 * s_kᵀ B_k s_k:
    θ = 1
else:
    θ = 0.8 * s_kᵀ B_k s_k / (s_kᵀ B_k s_k - y_kᵀ s_k)
r_k = θ * y_k + (1-θ) * B_k s_k   # use r_k instead of y_k in L-BFGS
```

L-BFGS two-loop recursion to compute `B_k⁻¹ v` (used to compute the QP objective
implicitly — but since we use OSQP we need `B_k` explicitly, not just its action).

**Important**: OSQP needs the explicit matrix `B_k`, not just the L-BFGS action.
Materialise `B_k` as a dense `(n, n)` matrix from the L-BFGS factors using the
two-loop recursion applied to each column of `I_n`. This is O(n²·m) — fine for
small-to-medium n. For large n (n > 1000), switch to the implicit L-BFGS action
with a Krylov-based QP solver (future work).

Alternatively, pass `B_k` to OSQP in factored form via a custom linear solver —
investigate OSQP's `custom_kkt_solver` API for this.

---

## 4. KKT Conditions

Declare convergence when all three conditions hold simultaneously:

```
primal_eq   = ||c_eq(x)||_inf             ≤ tol_feas   (default 1e-6)
primal_ineq = ||max(0, c_ineq(x))||_inf  ≤ tol_feas
dual        = ||g - J_eqᵀ λ_eq - J_ineqᵀ λ_ineq||_inf ≤ tol_opt  (default 1e-6)
compl       = ||λ_ineq ⊙ c_ineq(x)||_inf              ≤ tol_comp (default 1e-6)
```

Also check `λ_ineq ≥ -tol_comp` (dual feasibility for inequalities).

---

## 5. Warm-Starting

The key feature differentiating this from SciPy SLSQP.

**What to carry between solves**:
```python
WarmStart = namedtuple('WarmStart', ['x', 'lam_eq', 'lam_ineq', 'lbfgs_pairs'])
```

- `x`: primal solution from previous solve
- `lam_eq`, `lam_ineq`: dual variables
- `lbfgs_pairs`: the stored `{(s_i, y_i)}` pairs (optional — often better to reset)

**In `pympcc` context**: The Scholtes/Smoothing/LinFukushima outer loops change
only the RHS of the complementarity constraints between iterations. The warm start
from iteration k is passed directly as `x0` and `lam0` to iteration k+1.
The L-BFGS pairs should be **reset** between outer iterations because the constraint
structure changes (different ε), making the old curvature information misleading.

---

## 6. Interface Design

### 6.1 NLP Problem Interface

```python
class NLPProblem:
    """
    Defines a standard NLP for pyfiltersqp.

    All callables must accept a 1-D ndarray of shape (n,) and return
    the appropriate output. Jacobians are REQUIRED (no finite differences).
    """
    n: int           # number of variables
    m_eq: int        # number of equality constraints
    m_ineq: int      # number of inequality constraints
    xl: np.ndarray   # lower bounds on x, shape (n,)
    xu: np.ndarray   # upper bounds on x, shape (n,)

    objective:     Callable[[np.ndarray], float]
    gradient:      Callable[[np.ndarray], np.ndarray]        # shape (n,)
    eq_constraints: Callable[[np.ndarray], np.ndarray]       # shape (m_eq,)
    eq_jacobian:    Callable[[np.ndarray], np.ndarray]       # shape (m_eq, n) or sparse
    ineq_constraints: Callable[[np.ndarray], np.ndarray]     # shape (m_ineq,)
    ineq_jacobian:    Callable[[np.ndarray], np.ndarray]     # shape (m_ineq, n) or sparse
```

Implement as a dataclass with `__post_init__` validation (mirror `pympcc.MPCCProblem`).

### 6.2 Solver Interface

```python
class SQPSolver:
    def __init__(
        self,
        tol_feas: float = 1e-6,
        tol_opt: float  = 1e-6,
        tol_comp: float = 1e-6,
        max_iter: int   = 200,
        max_ls_iter: int = 30,
        lbfgs_memory: int = 10,
        mu0: float  = 1.0,      # initial merit penalty
        rho: float  = 0.5,      # line search backtracking factor
        eta: float  = 1e-4,     # Armijo sufficient decrease constant
        verbose: bool = False,
    ): ...

    def solve(
        self,
        problem: NLPProblem,
        x0: np.ndarray,
        lam_eq0:   np.ndarray | None = None,
        lam_ineq0: np.ndarray | None = None,
    ) -> SQPResult: ...
```

### 6.3 Result

```python
@dataclass
class SQPResult:
    x:          np.ndarray   # primal solution
    obj:        float        # f(x)
    lam_eq:     np.ndarray   # equality multipliers
    lam_ineq:   np.ndarray   # inequality multipliers (≥ 0 at solution)
    lam_lb:     np.ndarray   # lower bound multipliers (≥ 0)
    lam_ub:     np.ndarray   # upper bound multipliers (≥ 0)
    status:     int          # 0=success, 1=max_iter, 2=line_search_fail, 3=qp_fail
    message:    str
    success:    bool
    iterations: int
    history:    list[IterationInfo]   # per-iteration data (optional, verbose mode)
```

`IterationInfo` stores: `x`, `obj`, `primal_feas`, `dual_feas`, `compl`, `step_size`, `qp_iters`.

### 6.4 pympcc Adapter

The adapter lives in `pympcc/_nlp.py` as `_FilterSQPAdapter` and exposes exactly
the same `solve(x0, lagrange=None, zl=None, zu=None) -> (x, info)` as cyipopt,
where `info` contains `"obj_val"`, `"status"`, `"status_msg"`, `"mult_g"`,
`"mult_x_L"`, `"mult_x_U"`.

```python
class _FilterSQPAdapter:
    def __init__(self, n, m, xl, xu, cl, cu, obj_fn, grad_fn, con_fn, jac_fn,
                 solver_options=None): ...

    def solve(self, x0, lagrange=None, zl=None, zu=None) -> tuple[np.ndarray, dict]:
        # 1. Split cl/cu into eq (cl==cu) and ineq (cl<cu) rows
        # 2. Build NLPProblem
        # 3. Call SQPSolver.solve() with warm start from lagrange
        # 4. Pack result into cyipopt-compatible info dict
        ...
```

The constraint split (step 1 above) is the main adaptation needed because `pympcc`
strategies pass a single `c(x)` with bounds `[cl, cu]`, while `NLPProblem` separates
equalities from inequalities.

---

## 7. Module Layout

```
pyfiltersqp/
├── src/
│   └── pyfiltersqp/
│       ├── __init__.py          # exports: NLPProblem, SQPSolver, SQPResult, solve()
│       ├── problem.py           # NLPProblem dataclass + validation
│       ├── solver.py            # SQPSolver: outer SQP loop
│       ├── result.py            # SQPResult, IterationInfo dataclasses
│       ├── _lbfgs.py            # LBFGSMemory: stores pairs, materialises B_k
│       ├── _qp.py               # QPSubproblem: wraps OSQP, returns (p, lam_eq, lam_ineq)
│       ├── _line_search.py      # armijo_backtrack(), l1_merit(), directional_deriv()
│       ├── _kkt.py              # kkt_residuals() -> (primal_eq, primal_ineq, dual, compl)
│       └── _filter.py           # Filter class (v0.2 — stub only in v0.1)
├── tests/
│   ├── problems.py              # Hock-Schittkowski test problems (HS15, HS71, HS100)
│   ├── test_basic.py            # unconstrained + equality-only + inequality
│   ├── test_warmstart.py        # verify warm-start cuts iteration count
│   └── test_pympcc_adapter.py   # run pympcc MacMPEC benchmarks via this backend
├── DESIGN.md                    # this file
├── ROADMAP.md                   # phased implementation plan
├── CLAUDE.md                    # AI assistant guidance
└── pyproject.toml
```

---

## 8. Dependencies

| Package | Role | License |
|---------|------|---------|
| `numpy` | arrays, linear algebra | BSD |
| `scipy` | sparse matrix utilities (CSC format for OSQP) | BSD |
| `osqp`  | QP subproblem solver | Apache 2.0 |

Optional / future:
| `pympcc` | integration test consumer | — |
| `numba`  | JIT-compile the L-BFGS materialisation loop | BSD |

No C++ in v0.1. The outer SQP loop is pure Python + NumPy; OSQP provides the
performance-critical inner solve in compiled C.

---

## 9. Key Implementation Notes

### L-BFGS materialisation
Materialising `B_k` as a dense matrix is O(n²·m). For the MacMPEC benchmark problems
(n ≤ ~50) this is negligible. For larger problems the bottleneck will be here.
Profile before optimising — OSQP will likely dominate for n > 200.

### Constraint indexing
Keep `m_eq` equality rows first, `m_ineq` inequality rows second, consistently throughout.
The OSQP `A` matrix and multiplier arrays must respect this ordering.

### OSQP settings
Default OSQP settings are tuned for general QPs. For the SQP context:
- `warm_starting = True` (carry between SQP iterations)
- `eps_abs = 1e-8`, `eps_rel = 1e-8` (tighter than OSQP defaults)
- `max_iter = 4000`
- `polish = True` (exact solution refinement — important for multiplier quality)

### Numerical safeguards
- Clip step size: `alpha_min = 1e-10` before declaring line search failure
- Skip L-BFGS update if `||s_k|| < 1e-14` (degenerate step)
- Reset L-BFGS to identity if condition number of `B_k` exceeds 1e12

---

## 10. Testing Strategy

### Unit tests
- `_lbfgs.py`: verify `B_k v` matches finite-difference result on a random problem
- `_qp.py`: verify QP solution against direct OSQP call
- `_line_search.py`: verify Armijo condition is satisfied

### Integration tests (Hock-Schittkowski)
Use HS15, HS71, HS100 (freely available, simple, known optimal values):
- HS15: 2 vars, 2 ineq — tests basic inequality handling
- HS71: 4 vars, 1 eq + 1 ineq — the classic SQP demo problem
- HS100: 7 vars, 4 ineq — tests robustness

### pympcc integration tests
Port `tests/test_macmpec.py` benchmarks to use `pyfiltersqp` as backend.
Target: all 8 MacMPEC problems × Scholtes + Smoothing + LinFukushima strategies,
same accuracy thresholds as the IPOPT baseline.

---

## 11. References

- Nocedal & Wright, *Numerical Optimization*, 2nd ed. (2006) — Chapters 18 (SQP), 7 (L-BFGS)
- Fletcher & Leyffer (2002), "Nonlinear programming without a penalty function", *Mathematical Programming* — filter line search
- Scholtes (2001), "Convergence properties of a regularization scheme for MPCCs" — the Scholtes relaxation pympcc implements
- OSQP paper: Stellato et al. (2020), *Mathematical Programming Computation*
- Powell (1978), "A fast algorithm for nonlinearly constrained optimization calculations" — damped BFGS
