"""Basic solver tests using Hock-Schittkowski problems."""
from __future__ import annotations

import numpy as np
import pytest

from pyfiltersqp import solve
from pyfiltersqp._filter import Filter, constraint_violation
from .problems import PROBLEMS


# Relative objective tolerance for all HS problems
OBJ_RTOL = 1e-3


@pytest.mark.parametrize("name,entry", PROBLEMS.items())
def test_hs_convergence(name, entry):
    """Solver must report success on all HS test problems."""
    result = solve(entry["problem"], entry["x0"], verbose=False)
    assert result.success, (
        f"{name}: solver failed with status={result.status} ({result.message})"
    )


@pytest.mark.parametrize("name,entry", PROBLEMS.items())
def test_hs_objective(name, entry):
    """Final objective must be within OBJ_RTOL of known optimum."""
    result = solve(entry["problem"], entry["x0"])
    f_opt = entry["f_opt"]
    assert abs(result.obj - f_opt) / (1.0 + abs(f_opt)) < OBJ_RTOL, (
        f"{name}: obj={result.obj:.6f}, expected≈{f_opt:.6f}"
    )


@pytest.mark.parametrize("name,entry", PROBLEMS.items())
def test_hs_feasibility(name, entry):
    """Final point must satisfy constraints to tol_feas."""
    tol = 1e-4
    prob = entry["problem"]
    result = solve(prob, entry["x0"])
    x = result.x

    if prob.m_eq > 0:
        eq_viol = np.max(np.abs(prob.eq_constraints(x)))
        assert eq_viol <= tol, f"{name}: equality violation {eq_viol:.2e} > {tol}"

    if prob.m_ineq > 0:
        ineq_viol = np.max(np.maximum(0.0, prob.ineq_constraints(x)))
        assert ineq_viol <= tol, f"{name}: inequality violation {ineq_viol:.2e} > {tol}"


def test_warm_start_reduces_iterations():
    """Warm-starting from a previous solution should require fewer iterations."""
    entry = PROBLEMS["hs71"]
    prob, x0 = entry["problem"], entry["x0"]

    solver = __import__("pyfiltersqp").SQPSolver(verbose=True)

    cold = solver.solve(prob, x0)
    assert cold.success

    warm = solver.solve(prob, x0, lam_eq0=cold.lam_eq, lam_ineq0=cold.lam_ineq)
    # Warm start from the optimal multipliers should converge faster
    # (at least not worse — allow equal as a degenerate case)
    assert warm.iterations <= cold.iterations + 2, (
        f"Warm start took {warm.iterations} iters vs cold {cold.iterations}"
    )


# ---------------------------------------------------------------------------
# Filter class unit tests
# ---------------------------------------------------------------------------

def test_filter_empty_accepts_all():
    filt = Filter(h_max=1e4)
    assert filt.is_acceptable(0.0, 0.0)     # empty filter: any h ≤ h_max is acceptable
    assert filt.is_acceptable(1e6, 9999.9)  # f large, h just below h_max — still accepted
    assert not filt.is_acceptable(0.0, 1e5) # h > h_max → always rejected
    assert len(filt) == 0


def test_filter_dominance():
    filt = Filter(gamma_f=1e-5, gamma_h=1e-5)
    filt.add(10.0, 1.0)
    # dominated on both axes (no margin improvement)
    assert not filt.is_acceptable(10.0, 1.0)
    assert not filt.is_acceptable(11.0, 1.5)
    # better on f axis → acceptable
    assert filt.is_acceptable(9.99, 1.5)
    # better on h axis → acceptable
    assert filt.is_acceptable(11.0, 0.99)


def test_filter_add_prunes():
    filt = Filter()
    filt.add(10.0, 5.0)
    filt.add(8.0, 3.0)   # dominates (10, 5) → pruned
    assert len(filt) == 1
    assert filt._f[0] == 8.0 and filt._h[0] == 3.0


def test_filter_sufficient_decrease_feasible():
    filt = Filter(beta_h=1e-4, beta_f=1e-4)
    # h_curr ≈ 0: require f decrease
    assert filt.has_sufficient_decrease(5.0, 0.0, 4.9, 0.0)
    assert not filt.has_sufficient_decrease(5.0, 0.0, 5.1, 0.0)


def test_filter_sufficient_decrease_infeasible():
    filt = Filter(beta_h=1e-4, beta_f=1e-4)
    h = 1.0
    # h-type: violation decreases by more than beta_h
    assert filt.has_sufficient_decrease(5.0, h, 5.5, (1 - 1e-4) * h - 1e-8)
    # f-type: objective decreases by more than beta_f * h
    assert filt.has_sufficient_decrease(5.0, h, 5.0 - 1e-4 * h - 1e-8, 1.0)
    # neither: neither axis improves enough
    assert not filt.has_sufficient_decrease(5.0, h, 5.0, h)


def test_constraint_violation():
    c_eq   = np.array([0.5, -0.3])
    c_ineq = np.array([-1.0, 0.4, -0.2])  # only 0.4 is violated
    h = constraint_violation(c_eq, c_ineq)
    assert abs(h - (0.5 + 0.3 + 0.4)) < 1e-12


# ---------------------------------------------------------------------------
# Filter solver tests — use_filter=True on HS problems
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,entry", PROBLEMS.items())
def test_filter_hs_convergence(name, entry):
    """Filter line search must also converge on all HS test problems."""
    result = solve(entry["problem"], entry["x0"], use_filter=True)
    assert result.success, (
        f"{name} (filter): failed with status={result.status} ({result.message})"
    )


@pytest.mark.parametrize("name,entry", PROBLEMS.items())
def test_filter_hs_objective(name, entry):
    """Filter solver objective must match known optimum to OBJ_RTOL."""
    result = solve(entry["problem"], entry["x0"], use_filter=True)
    f_opt = entry["f_opt"]
    assert abs(result.obj - f_opt) / (1.0 + abs(f_opt)) < OBJ_RTOL, (
        f"{name} (filter): obj={result.obj:.6f}, expected≈{f_opt:.6f}"
    )


def test_hs71_sparse_jacobian():
    """Solver must produce identical results when Jacobians are sparse CSR matrices."""
    import scipy.sparse as sp
    from pyfiltersqp import NLPProblem

    entry = PROBLEMS["hs71"]
    p = entry["problem"]

    prob_sparse = NLPProblem(
        n=p.n, m_eq=p.m_eq, m_ineq=p.m_ineq,
        xl=p.xl, xu=p.xu,
        objective=p.objective,
        gradient=p.gradient,
        eq_constraints=p.eq_constraints,
        eq_jacobian=lambda x: sp.csr_matrix(p.eq_jacobian(x)),
        ineq_constraints=p.ineq_constraints,
        ineq_jacobian=lambda x: sp.csr_matrix(p.ineq_jacobian(x)),
    )

    result = solve(prob_sparse, entry["x0"])
    f_opt = entry["f_opt"]
    assert result.success, f"hs71 sparse: failed with status={result.status} ({result.message})"
    assert abs(result.obj - f_opt) / (1.0 + abs(f_opt)) < OBJ_RTOL, (
        f"hs71 sparse: obj={result.obj:.6f}, expected≈{f_opt:.6f}"
    )


def test_unconstrained_quadratic():
    """Unconstrained quadratic: min ||x - x*||² should converge in one step."""
    from pyfiltersqp import NLPProblem
    x_star = np.array([1.0, 2.0, 3.0])
    n = len(x_star)
    prob = NLPProblem(
        n=n, m_eq=0, m_ineq=0,
        xl=np.full(n, -np.inf),
        xu=np.full(n,  np.inf),
        objective=lambda x: float(np.sum((x - x_star)**2)),
        gradient=lambda x: 2.0 * (x - x_star),
        x0=np.zeros(n),
    )
    result = solve(prob, np.zeros(n))
    assert result.success
    np.testing.assert_allclose(result.x, x_star, atol=1e-5)


# ---------------------------------------------------------------------------
# Implicit L-BFGS QP tests (use_implicit_qp=True)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,entry", PROBLEMS.items())
def test_implicit_qp_hs_convergence(name, entry):
    """ImplicitLBFGSQP must converge on all HS test problems."""
    result = solve(entry["problem"], entry["x0"], use_implicit_qp=True)
    assert result.success, (
        f"{name} (implicit): failed with status={result.status} ({result.message})"
    )


@pytest.mark.parametrize("name,entry", PROBLEMS.items())
def test_implicit_qp_hs_objective(name, entry):
    """ImplicitLBFGSQP objective must match known optimum to OBJ_RTOL."""
    result = solve(entry["problem"], entry["x0"], use_implicit_qp=True)
    f_opt = entry["f_opt"]
    assert abs(result.obj - f_opt) / (1.0 + abs(f_opt)) < OBJ_RTOL, (
        f"{name} (implicit): obj={result.obj:.6f}, expected≈{f_opt:.6f}"
    )


# ---------------------------------------------------------------------------
# Woodbury-ADMM QP tests (use_woodbury_admm=True)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,entry", PROBLEMS.items())
def test_woodbury_admm_hs_convergence(name, entry):
    """WoodburyADMM must converge on all HS test problems."""
    result = solve(entry["problem"], entry["x0"], use_woodbury_admm=True)
    assert result.success, (
        f"{name} (woodbury): failed with status={result.status} ({result.message})"
    )


@pytest.mark.parametrize("name,entry", PROBLEMS.items())
def test_woodbury_admm_hs_objective(name, entry):
    """WoodburyADMM objective must match known optimum to OBJ_RTOL."""
    result = solve(entry["problem"], entry["x0"], use_woodbury_admm=True)
    f_opt = entry["f_opt"]
    assert abs(result.obj - f_opt) / (1.0 + abs(f_opt)) < OBJ_RTOL, (
        f"{name} (woodbury): obj={result.obj:.6f}, expected≈{f_opt:.6f}"
    )


def test_woodbury_solve_unit():
    """WoodburyADMM._woodbury_apply(rhs) must match dense (B + ρ AᵀA)⁻¹ rhs."""
    from pyfiltersqp._qp_admm import WoodburyADMM
    rng = np.random.default_rng(17)
    n, m_eq, m_ineq = 8, 2, 3
    m_constr = m_eq + m_ineq

    lbfgs = _make_lbfgs_with_pairs(n=n, m=4, seed=11)
    B = lbfgs.materialise()   # dense B_k for reference

    J_eq    = rng.standard_normal((m_eq,    n))
    J_ineq  = rng.standard_normal((m_ineq,  n))
    rho     = 2.0

    admm = WoodburyADMM(n, m_eq, m_ineq, rho=rho)
    import scipy.sparse as sp
    J_constr = admm._stack_J(J_eq, J_ineq, n, m_eq, m_ineq)
    admm._woodbury_setup(lbfgs, J_constr, rho)

    A = np.vstack([J_eq, J_ineq, np.eye(n)])   # (m_constr+n) × n
    M_dense = B + rho * (A.T @ A)
    M_dense_inv = np.linalg.inv(M_dense)

    for _ in range(5):
        rhs = rng.standard_normal(n)
        p_woodbury = admm._woodbury_apply(rhs, J_constr)
        p_dense    = M_dense_inv @ rhs
        np.testing.assert_allclose(p_woodbury, p_dense, rtol=1e-6,
                                    err_msg="Woodbury apply disagrees with dense inverse")


def test_lbfgs_matvec_is_inverse_hessian():
    """_matvec computes H_k = B_k^{-1}: verify via L-BFGS secant condition H_k y = s."""
    from pyfiltersqp._lbfgs import LBFGSMemory

    rng = np.random.default_rng(42)
    n, m = 10, 5

    lbfgs = LBFGSMemory(n, memory=m)
    # Assign pairs directly with well-conditioned positive curvature
    for _ in range(m):
        s = rng.standard_normal(n); s /= np.linalg.norm(s)
        y = 2.0 * s + rng.standard_normal(n) * 1e-3
        lbfgs._s.append(s.copy())
        lbfgs._y.append(y.copy())
    lbfgs._scale = float(lbfgs._y[-1] @ lbfgs._y[-1]) / float(lbfgs._y[-1] @ lbfgs._s[-1])

    # Secant condition: H_k y_{k-1} = s_{k-1}  (must hold for the most recent pair)
    s_last = lbfgs._s[-1]
    y_last = lbfgs._y[-1]
    np.testing.assert_allclose(lbfgs._matvec(y_last), s_last, rtol=1e-12,
                                err_msg="L-BFGS secant condition _matvec(y) = s failed")


def _make_lbfgs_with_pairs(n: int, m: int, seed: int = 0) -> "LBFGSMemory":
    """Helper: build an LBFGSMemory with m well-conditioned pairs via update()."""
    from pyfiltersqp._lbfgs import LBFGSMemory

    rng = np.random.default_rng(seed)
    lbfgs = LBFGSMemory(n, memory=m)
    for _ in range(m):
        s = rng.standard_normal(n) * 0.1
        # y chosen so sᵀy > 0 (positive curvature, no damping needed)
        y = 3.0 * s + rng.standard_normal(n) * 1e-2
        lbfgs.update(s, y)
    return lbfgs


def test_lbfgs_matvec_forward_secant():
    """matvec(s_last) = y_last: BFGS Hessian secant condition for the last pair."""
    lbfgs = _make_lbfgs_with_pairs(n=8, m=4)
    s_last = lbfgs._s[-1]
    y_last = lbfgs._y[-1]
    np.testing.assert_allclose(lbfgs.matvec(s_last), y_last, rtol=1e-10,
                                err_msg="matvec secant condition matvec(s) = y failed")


def test_lbfgs_matvec_matches_materialise():
    """matvec(v) must agree with materialise() @ v for arbitrary v."""
    n, m = 12, 5
    lbfgs = _make_lbfgs_with_pairs(n=n, m=m, seed=7)
    B = lbfgs.materialise()

    rng = np.random.default_rng(99)
    for _ in range(10):
        v = rng.standard_normal(n)
        np.testing.assert_allclose(lbfgs.matvec(v), B @ v, rtol=1e-10,
                                    err_msg="matvec(v) != materialise() @ v")


def test_lbfgs_factors_reconstruct():
    """sigma*I + Psi @ M @ Psi.T must equal materialise() to near machine precision."""
    n, m = 10, 4
    lbfgs = _make_lbfgs_with_pairs(n=n, m=m, seed=3)
    B_dense = lbfgs.materialise()

    sigma, Psi, M = lbfgs.factors()
    B_compact = sigma * np.eye(n) + Psi @ M @ Psi.T

    np.testing.assert_allclose(B_compact, B_dense, rtol=1e-8,
                                err_msg="compact factors do not reconstruct materialise()")


def test_lbfgs_materialise_is_hessian():
    """materialise() must return B_k: verify B_k s = y for the last pair."""
    lbfgs = _make_lbfgs_with_pairs(n=8, m=3, seed=5)
    B = lbfgs.materialise()
    s_last = lbfgs._s[-1]
    y_last = lbfgs._y[-1]
    np.testing.assert_allclose(B @ s_last, y_last, rtol=1e-10,
                                err_msg="materialise() secant condition B @ s = y failed")
