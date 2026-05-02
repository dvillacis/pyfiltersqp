"""
Adapter: pycutest problem  →  pyfiltersqp.NLPProblem.

CUTEst constraint convention:    cl ≤ c(x) ≤ cu
pyfiltersqp convention:          c_eq(x) = 0,   c_ineq(x) ≤ 0,   xl ≤ x ≤ xu

Per-row mapping of c_i in CUTEst:
    cl_i == cu_i        → equality:    c_pcu_i(x) - cl_i = 0
    cl_i = -inf, cu_i finite → ineq:   c_pcu_i(x) - cu_i ≤ 0
    cl_i finite, cu_i = inf  → ineq:   cl_i - c_pcu_i(x) ≤ 0
    both finite, cl_i < cu_i → two ineq rows (range constraint split):
        c_pcu_i(x) - cu_i ≤ 0
        cl_i - c_pcu_i(x) ≤ 0

Bounds (pcu.bl / pcu.bu) map directly to pyfiltersqp's xl, xu (NaN/inf preserved).

Usage
-----
    import pycutest
    from tests.cutest_adapter import wrap

    cu = pycutest.import_problem("HS71")
    prob, x0 = wrap(cu)        # prob is an NLPProblem
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from pyfiltersqp import NLPProblem


def wrap(pcu) -> Tuple[NLPProblem, np.ndarray]:
    """
    Wrap a pycutest CUTEstProblem as an NLPProblem.

    Parameters
    ----------
    pcu : pycutest.CUTEstProblem
        Already imported via ``pycutest.import_problem(name, ...)``.

    Returns
    -------
    prob : NLPProblem
    x0   : ndarray, shape (n,)
        The CUTEst-recommended starting point.
    """
    n = int(pcu.n)
    m = int(pcu.m)

    xl = np.where(pcu.bl <= -1e19, -np.inf, pcu.bl).astype(float)
    xu = np.where(pcu.bu >=  1e19,  np.inf, pcu.bu).astype(float)
    x0 = np.asarray(pcu.x0, dtype=float).copy()

    if m == 0:
        prob = NLPProblem(
            n=n, m_eq=0, m_ineq=0,
            xl=xl, xu=xu,
            objective=lambda x, _p=pcu: float(_p.obj(x)),
            gradient =lambda x, _p=pcu: np.asarray(_p.grad(x), dtype=float),
            x0=x0,
        )
        return prob, x0

    # Classify each CUTEst row.
    cl = np.asarray(pcu.cl, dtype=float)
    cu = np.asarray(pcu.cu, dtype=float)
    is_eq = (cl == cu)

    eq_rows: list[int]   = []
    eq_off:  list[float] = []                         # c_pcu - eq_off = 0

    # ineq_spec[i] = (row, sign): sign=+1 means c_pcu - cu, sign=-1 means cl - c_pcu
    ineq_spec: list[tuple[int, int, float]] = []

    for i in range(m):
        if is_eq[i]:
            eq_rows.append(i)
            eq_off.append(cl[i])
        else:
            cu_finite = cu[i] <  1e19
            cl_finite = cl[i] > -1e19
            if cu_finite:
                ineq_spec.append((i, +1, cu[i]))      # c - cu ≤ 0
            if cl_finite:
                ineq_spec.append((i, -1, cl[i]))      # cl - c ≤ 0
            # If neither finite (free row), skip — CUTEst sometimes has these.

    m_eq   = len(eq_rows)
    m_ineq = len(ineq_spec)

    eq_idx     = np.asarray(eq_rows, dtype=np.intp)
    eq_off_arr = np.asarray(eq_off,  dtype=float)

    ineq_rows  = np.asarray([s[0] for s in ineq_spec], dtype=np.intp)
    ineq_signs = np.asarray([s[1] for s in ineq_spec], dtype=float)
    ineq_off   = np.asarray([s[2] for s in ineq_spec], dtype=float)

    def _eq(x, _p=pcu, _idx=eq_idx, _off=eq_off_arr):
        c = np.asarray(_p.cons(x), dtype=float)
        return c[_idx] - _off

    def _jeq(x, _p=pcu, _idx=eq_idx):
        # pycutest cons(x, gradient=True) returns (c, J)
        _, J = _p.cons(x, gradient=True)
        return np.asarray(J, dtype=float)[_idx]

    def _ineq(x, _p=pcu, _rows=ineq_rows, _signs=ineq_signs, _off=ineq_off):
        c = np.asarray(_p.cons(x), dtype=float)
        return _signs * c[_rows] - _signs * _off

    def _jineq(x, _p=pcu, _rows=ineq_rows, _signs=ineq_signs):
        _, J = _p.cons(x, gradient=True)
        return _signs[:, None] * np.asarray(J, dtype=float)[_rows]

    prob = NLPProblem(
        n=n, m_eq=m_eq, m_ineq=m_ineq,
        xl=xl, xu=xu,
        objective=lambda x, _p=pcu: float(_p.obj(x)),
        gradient =lambda x, _p=pcu: np.asarray(_p.grad(x), dtype=float),
        eq_constraints=_eq if m_eq   > 0 else None,
        eq_jacobian   =_jeq if m_eq  > 0 else None,
        ineq_constraints=_ineq if m_ineq > 0 else None,
        ineq_jacobian   =_jineq if m_ineq > 0 else None,
        x0=x0,
    )
    return prob, x0
