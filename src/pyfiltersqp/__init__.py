"""
pyfiltersqp — open-source SQP solver with L-BFGS and l₁ merit / filter line search.

Quickstart
----------
>>> from pyfiltersqp import NLPProblem, solve
>>> import numpy as np
>>>
>>> problem = NLPProblem(
...     n=2, m_eq=0, m_ineq=1,
...     xl=np.array([-np.inf, -np.inf]),
...     xu=np.array([ np.inf,  np.inf]),
...     objective=lambda x: (x[0]-2)**2 + (x[1]-1)**2,
...     gradient=lambda x: np.array([2*(x[0]-2), 2*(x[1]-1)]),
...     ineq_constraints=lambda x: np.array([x[0]**2 + x[1]**2 - 2]),
...     ineq_jacobian=lambda x: np.array([[2*x[0], 2*x[1]]]),
... )
>>> result = solve(problem, x0=np.array([0.5, 0.5]))
>>> result.success, result.obj
"""
from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .problem import NLPProblem
from .result import SQPResult, IterationInfo
from .solver import SQPSolver, solve
from ._qp_admm import WoodburyADMM

try:
    __version__ = _pkg_version("pyfiltersqp")
except PackageNotFoundError:                       # pragma: no cover
    # Source/editable install without metadata installed.
    __version__ = "0.0.0+unknown"

__all__ = [
    "NLPProblem", "SQPResult", "IterationInfo", "SQPSolver", "solve",
    "WoodburyADMM", "__version__",
]
