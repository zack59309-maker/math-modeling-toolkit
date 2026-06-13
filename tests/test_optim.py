"""Tests for mm_optim.py - optimization functions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scripts.mm_optim import solve_lp


def test_solve_lp_basic():
    """Simple LP: min -3x - 2y with constraints."""
    result = solve_lp(
        c=[-3, -2],
        A_ub=[[2, 1], [1, 1]],
        b_ub=[18, 8],
        bounds=[(0, None), (0, None)],
    )
    assert result["status"] == "success"
    assert "x" in result
    assert "fval" in result
    x = result["x"]
    # Check constraints
    assert 2 * x[0] + x[1] <= 18 + 1e-8
    assert x[0] + x[1] <= 8 + 1e-8
    assert x[0] >= -1e-8
    assert x[1] >= -1e-8


def test_solve_lp_no_constraints():
    """Unbounded-ish LP should still produce a result."""
    result = solve_lp(
        c=[1, 1],
        bounds=[(0, 10), (0, 10)],
    )
    assert result["status"] == "success"
    assert len(result["x"]) == 2
