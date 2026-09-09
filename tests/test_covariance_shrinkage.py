"""Tests for the hierarchical shrinkage estimator in covariance_prior/build_covariance_registry.py.

Imported by file path since the script lives outside the installed `state` package.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parent.parent / "covariance_prior" / "build_covariance_registry.py"
spec = importlib.util.spec_from_file_location("build_covariance_registry", MODULE_PATH)
bcr = importlib.util.module_from_spec(spec)
sys.modules["build_covariance_registry"] = bcr
spec.loader.exec_module(bcr)


def dense_covariance(counts: np.ndarray) -> np.ndarray:
    centered = counts - counts.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / (counts.shape[0] - 1)


class TestComputeShrinkageWeights:
    def test_weights_sum_to_one_minus_delta(self):
        w = bcr.compute_shrinkage_weights(n_own=100, n_parent=200, n_global=300, kappa=1000.0, delta=0.05)
        assert w.own + w.parent + w.global_ == pytest.approx(0.95, abs=1e-9)

    def test_more_own_controls_gives_more_own_weight(self):
        w_small = bcr.compute_shrinkage_weights(n_own=50, n_parent=1000, n_global=0, kappa=1000.0, delta=0.05)
        w_large = bcr.compute_shrinkage_weights(n_own=5000, n_parent=1000, n_global=0, kappa=1000.0, delta=0.05)
        assert w_large.own > w_small.own

    def test_no_pools_falls_back_entirely_to_own(self):
        w = bcr.compute_shrinkage_weights(n_own=100, n_parent=0, n_global=0, kappa=1000.0, delta=0.05)
        assert w.parent == 0.0
        assert w.global_ == 0.0
        assert w.own == pytest.approx(0.95, abs=1e-9)

    def test_pool_split_proportional_to_pool_sizes(self):
        w = bcr.compute_shrinkage_weights(n_own=100, n_parent=300, n_global=100, kappa=1000.0, delta=0.05)
        # parent pool is 3x the global pool -> should get ~3x the weight
        assert w.parent == pytest.approx(3 * w.global_, rel=1e-6)


class TestShrunkLowRankCovariance:
    def test_matches_weighted_dense_covariance_at_full_rank(self):
        """With rank == min(N_total, G), the low-rank reconstruction should exactly match the
        weighted sum of dense covariances (up to the additive delta*I ridge)."""
        rng = np.random.default_rng(0)
        g = 6
        own = rng.standard_normal((20, g)).astype(np.float32) * 2.0
        parent = rng.standard_normal((15, g)).astype(np.float32) * 3.0
        global_ = rng.standard_normal((10, g)).astype(np.float32) * 0.5

        weights = bcr.ShrinkageWeights(own=0.5, parent=0.3, global_=0.15, delta=0.05)

        V, eigenvalues, gene_variance = bcr.shrunk_low_rank_covariance(own, parent, global_, weights, rank=g)

        expected = (
            weights.own * dense_covariance(own)
            + weights.parent * dense_covariance(parent)
            + weights.global_ * dense_covariance(global_)
            + weights.delta * np.eye(g)
        )

        reconstructed = (V * eigenvalues) @ V.T
        np.testing.assert_allclose(reconstructed, expected, atol=1e-3)
        np.testing.assert_allclose(np.diag(expected), gene_variance, atol=1e-3)

    def test_zero_weight_pool_has_no_effect(self):
        rng = np.random.default_rng(1)
        g = 5
        own = rng.standard_normal((20, g)).astype(np.float32)
        parent = rng.standard_normal((20, g)).astype(np.float32) * 100.0  # would dominate if used

        weights_with_parent = bcr.ShrinkageWeights(own=0.95, parent=0.0, global_=0.0, delta=0.05)
        V1, eig1, var1 = bcr.shrunk_low_rank_covariance(own, parent, None, weights_with_parent, rank=g)
        V2, eig2, var2 = bcr.shrunk_low_rank_covariance(own, None, None, weights_with_parent, rank=g)

        np.testing.assert_allclose((V1 * eig1) @ V1.T, (V2 * eig2) @ V2.T, atol=1e-4)
        np.testing.assert_allclose(var1, var2, atol=1e-4)

    def test_no_shrinkage_matches_plain_estimator(self):
        """weights=(1,0,0,0) should reduce to the original unshrunk low-rank estimator."""
        rng = np.random.default_rng(2)
        g = 8
        own = rng.standard_normal((30, g)).astype(np.float32)
        weights = bcr.ShrinkageWeights(own=1.0, parent=0.0, global_=0.0, delta=0.0)

        V, eigenvalues, gene_variance = bcr.shrunk_low_rank_covariance(own, None, None, weights, rank=g)
        expected = dense_covariance(own)
        reconstructed = (V * eigenvalues) @ V.T
        np.testing.assert_allclose(reconstructed, expected, atol=1e-3)

    def test_rank_padding_when_fewer_components_available(self):
        rng = np.random.default_rng(3)
        g = 50
        own = rng.standard_normal((6, g)).astype(np.float32)  # only 5 nontrivial components
        weights = bcr.ShrinkageWeights(own=1.0, parent=0.0, global_=0.0, delta=0.01)

        V, eigenvalues, _ = bcr.shrunk_low_rank_covariance(own, None, None, weights, rank=20)
        assert V.shape == (g, 20)
        assert eigenvalues.shape == (20,)
        # padded components should sit at the ridge floor
        assert np.allclose(eigenvalues[5:], weights.delta, atol=1e-6)
        assert np.allclose(V[:, 5:], 0.0)
