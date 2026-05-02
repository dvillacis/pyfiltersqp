"""
Fletcher-Leyffer filter for SQP globalisation.

A filter stores (f_i, h_i) pairs where h_i = constraint violation.  A new iterate
is *filter-acceptable* when no stored pair dominates it, i.e. there is no entry
for which BOTH the objective and the violation are at least as bad (with a small
margin γ).

References
----------
Fletcher, R., & Leyffer, S. (2002). Nonlinear programming without a penalty
  function. *Mathematical Programming*, 91(2), 239–269.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Standalone helper — also used in solver.py and _line_search.py
# ---------------------------------------------------------------------------

def constraint_violation(c_eq: np.ndarray, c_ineq: np.ndarray) -> float:
    """
    l₁ constraint violation measure::

        h(x) = ‖c_eq(x)‖₁ + ‖max(0, c_ineq(x))‖₁
    """
    h = 0.0
    if c_eq.size > 0:
        h += float(np.sum(np.abs(c_eq)))
    if c_ineq.size > 0:
        h += float(np.sum(np.maximum(0.0, c_ineq)))
    return h


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class Filter:
    """
    Fletcher-Leyffer filter for SQP globalisation.

    A trial point (f, h) is **dominated** by stored entry (f_i, h_i) when::

        f  ≥  f_i − γ_f · h_i   AND   h  ≥  h_i − γ_h · h_i

    i.e. it is not appreciably better than (f_i, h_i) on either axis.
    A point is *filter-acceptable* when it is not dominated by any entry
    AND h ≤ h_max.

    A step from (f_curr, h_curr) has **sufficient decrease** when at least one of::

        h_trial  <  (1 − β_h) · h_curr        (h-type: violation decreases)
        f_trial  <  f_curr − β_f · h_curr     (f-type: objective decreases relative to h)

    Typically both conditions are required together with filter-acceptability
    to accept a step.

    Parameters
    ----------
    h_max : float
        Hard ceiling on constraint violation; points with h > h_max are always
        rejected.  Updated dynamically after initialisation if needed.
    gamma_f, gamma_h : float
        Margin parameters for the dominance check (default 1e-5).
    beta_h : float
        Relative decrease threshold for the h-type condition (default 1e-4).
    beta_f : float
        Relative f threshold for the f-type condition (default 1e-4).
    """

    def __init__(
        self,
        h_max:   float = 1e4,
        gamma_f: float = 1e-5,
        gamma_h: float = 1e-5,
        beta_h:  float = 1e-4,
        beta_f:  float = 1e-4,
    ):
        self.h_max   = h_max
        self.gamma_f = gamma_f
        self.gamma_h = gamma_h
        self.beta_h  = beta_h
        self.beta_f  = beta_f
        self._f: list[float] = []
        self._h: list[float] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def is_acceptable(self, f: float, h: float) -> bool:
        """True iff (f, h) is filter-acceptable (not dominated, h ≤ h_max)."""
        if h > self.h_max:
            return False
        gf, gh = self.gamma_f, self.gamma_h
        for fi, hi in zip(self._f, self._h):
            if f >= fi - gf * hi and h >= hi - gh * hi:
                return False
        return True

    def has_sufficient_decrease(
        self, f_curr: float, h_curr: float, f_trial: float, h_trial: float
    ) -> bool:
        """
        True iff the trial point has sufficient decrease relative to the
        current point (h-type or f-type condition).
        """
        if h_curr < 1e-14:
            return f_trial < f_curr          # feasible current: require f decrease
        return (
            h_trial < (1.0 - self.beta_h) * h_curr
            or f_trial < f_curr - self.beta_f * h_curr
        )

    def add(self, f: float, h: float) -> None:
        """
        Add (f, h) to the filter, pruning dominated entries.

        An existing (f_i, h_i) is pruned when the new entry dominates it,
        i.e. f ≤ f_i AND h ≤ h_i (the new entry is no worse on both axes).
        """
        keep_f, keep_h = [], []
        for fi, hi in zip(self._f, self._h):
            if not (f <= fi and h <= hi):
                keep_f.append(fi)
                keep_h.append(hi)
        keep_f.append(f)
        keep_h.append(h)
        self._f, self._h = keep_f, keep_h

    def reset(self) -> None:
        """Clear all filter entries."""
        self._f.clear()
        self._h.clear()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._f)

    def __repr__(self) -> str:  # pragma: no cover
        entries = ", ".join(
            f"(f={fi:.3e}, h={hi:.3e})" for fi, hi in zip(self._f, self._h)
        )
        return f"Filter([{entries}])"
