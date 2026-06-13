"""Tests for mm_eval.py - evaluation/decision functions."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scripts.mm_eval import ahp_consistency, ahp_weight, topsis, entropy_weight


def test_ahp_consistency_consistent():
    """Perfectly consistent 3x3 matrix should have CR=0."""
    matrix = np.array([
        [1,   2,   4],
        [1/2, 1,   2],
        [1/4, 1/2, 1],
    ])
    result = ahp_consistency(matrix)
    assert result["is_consistent"] is True
    assert result["CR"] < 0.1
    assert len(result["weight"]) == 3
    assert abs(np.sum(result["weight"]) - 1.0) < 1e-6


def test_ahp_weight():
    """ahp_weight returns normalized weights."""
    matrix = np.array([
        [1,   3,   5],
        [1/3, 1,   3],
        [1/5, 1/3, 1],
    ])
    weights = ahp_weight(matrix)
    assert len(weights) == 3
    assert abs(np.sum(weights) - 1.0) < 1e-6
    assert np.all(weights > 0)


def test_topsis_basic():
    """TOPSIS with 4 samples, 3 criteria."""
    data = np.array([
        [8,   7,   8],
        [7,   8,   7],
        [9,   6,   9],
        [6,   9,   6],
    ])
    result = topsis(data)
    assert "score" in result
    assert "rank" in result
    assert len(result["score"]) == 4
    assert len(result["rank"]) == 4
    assert 0 <= result["score"].min() <= result["score"].max() <= 1


def test_topsis_with_weights():
    """TOPSIS with custom weights."""
    data = np.array([
        [8, 7, 8],
        [7, 8, 7],
    ])
    weights = np.array([0.5, 0.3, 0.2])
    result = topsis(data, weights=weights)
    assert len(result["score"]) == 2


def test_entropy_weight():
    """Entropy weight calculation."""
    data = np.array([
        [100, 200, 150],
        [120, 180, 160],
        [110, 210, 140],
    ])
    weights = entropy_weight(data)
    assert len(weights) == 3
    assert abs(np.sum(weights) - 1.0) < 1e-6
    assert np.all(weights >= 0)
