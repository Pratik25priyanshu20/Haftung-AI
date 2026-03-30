"""Tests for weight ablation grid generation."""
from __future__ import annotations

import pytest

from evaluation.runners.run_weight_ablation import generate_weight_grid


def test_grid_sums_to_one():
    """All weight combos must sum to 1.0."""
    grid = generate_weight_grid()
    for w_llm, w_cov, w_base in grid:
        total = round(w_llm + w_cov + w_base, 4)
        assert total == 1.0, f"Weights ({w_llm}, {w_cov}, {w_base}) sum to {total}"


def test_grid_contains_default():
    """The default weights (0.4, 0.3, 0.3) must be in the grid."""
    grid = generate_weight_grid()
    assert (0.4, 0.3, 0.3) in grid


def test_grid_respects_bounds():
    """All weights must be within [min_w, max_w]."""
    min_w = 0.1
    max_w = 0.8
    grid = generate_weight_grid(min_w=min_w, max_w=max_w)
    for w_llm, w_cov, w_base in grid:
        assert min_w <= w_llm <= max_w, f"w_llm={w_llm} out of bounds"
        assert min_w <= w_cov <= max_w, f"w_cov={w_cov} out of bounds"
        assert min_w <= w_base <= max_w, f"w_base={w_base} out of bounds"


def test_grid_nonempty():
    """Grid must produce at least one combination."""
    grid = generate_weight_grid()
    assert len(grid) > 0


def test_grid_no_duplicates():
    """No duplicate weight combos."""
    grid = generate_weight_grid()
    assert len(grid) == len(set(grid))


def test_grid_custom_step():
    """Custom step size should work."""
    grid = generate_weight_grid(step=0.2, min_w=0.2, max_w=0.6)
    for w_llm, w_cov, w_base in grid:
        total = round(w_llm + w_cov + w_base, 4)
        assert total == 1.0
        assert 0.2 <= w_llm <= 0.6
        assert 0.2 <= w_cov <= 0.6
        assert 0.2 <= w_base <= 0.6
