"""Tests for CovariancePriorCrossAttention.register_context — on-the-fly covariance estimation
for a brand-new context at inference time (spec's "Case 2/3: new context" workflow), without
rerunning the offline precompute script.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from state.tx.models.cross_attention import GLOBAL_CONTEXT_NAME, CovariancePriorCrossAttention


N_GENES = 12
RANK = 4
D_MODEL = 16


def dense_covariance(counts: np.ndarray) -> np.ndarray:
    centered = counts - counts.mean(axis=0, keepdims=True)
    return (centered.T @ centered) / (counts.shape[0] - 1)


def reconstruct(module: CovariancePriorCrossAttention, idx: int) -> np.ndarray:
    V = module.V[idx].detach().cpu().numpy()
    eig = module.log_eigenvalues[idx].exp().detach().cpu().numpy()
    return (V * eig) @ V.T


@pytest.fixture
def registry_with_global(tmp_path: Path) -> str:
    """Registry with two known contexts plus a __global__ pooled pseudo-context, mirroring
    what build_covariance_registry.py now produces."""
    rng = np.random.default_rng(0)
    gene_symbols = [f"GENE{i}" for i in range(N_GENES)]

    countsA = rng.standard_normal((60, N_GENES)).astype(np.float32) * 2.0
    countsB = rng.standard_normal((40, N_GENES)).astype(np.float32) * 0.5

    def low_rank(counts):
        centered = counts - counts.mean(axis=0, keepdims=True)
        _u, s, vt = np.linalg.svd(centered, full_matrices=False)
        k = min(RANK, s.shape[0])
        eig = (s[:k] ** 2) / (counts.shape[0] - 1)
        V = np.zeros((N_GENES, RANK), dtype=np.float32)
        eigenvalues = np.full((RANK,), 1e-6, dtype=np.float32)
        V[:, :k] = vt[:k].T
        eigenvalues[:k] = eig
        var = centered.var(axis=0, ddof=1).astype(np.float32)
        return V, eigenvalues, var

    V_a, eig_a, var_a = low_rank(countsA)
    V_b, eig_b, var_b = low_rank(countsB)

    # crude "global" pool: just decompose the concatenation directly (test fixture only)
    V_g, eig_g, var_g = low_rank(np.concatenate([countsA, countsB], axis=0))

    out = tmp_path / "registry.npz"
    np.savez_compressed(
        out,
        V=np.stack([V_a, V_b, V_g]).astype(np.float16),
        eigenvalues=np.stack([eig_a, eig_b, eig_g]),
        gene_variance=np.stack([var_a, var_b, var_g]),
        context_names=np.array(["dsA.cellX", "dsA.cellY", GLOBAL_CONTEXT_NAME]),
        gene_symbols=np.array(gene_symbols),
        n_control_cells=np.array([60, 40, 100], dtype=np.int64),
    )
    return str(out)


class TestRegisterContext:
    def test_context_count_and_lookup_shape(self, registry_with_global):
        module = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        n_before = module.n_contexts
        rng = np.random.default_rng(1)
        new_counts = rng.standard_normal((30, N_GENES)).astype(np.float32)

        idx = module.register_context("dsB.cellZ", new_counts)
        assert idx == n_before
        assert module.n_contexts == n_before + 1
        assert "dsB.cellZ" in module.context_name_to_idx

        module.eval()
        kv, mask = module.lookup(torch.tensor([idx]), torch.tensor([0]))
        assert kv.shape == (1, 1, D_MODEL)
        assert mask is None

    def test_registered_context_differs_from_no_cov(self, registry_with_global):
        module = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        rng = np.random.default_rng(2)
        new_counts = rng.standard_normal((50, N_GENES)).astype(np.float32) * 3.0
        idx = module.register_context("dsB.cellZ", new_counts)

        module.eval()
        with torch.no_grad():
            kv, _ = module.lookup(torch.tensor([idx]), torch.tensor([3]))
        assert not torch.allclose(kv[0, 0], module.no_cov_token[0])

    def test_no_parent_no_global_matches_own_only_estimate(self, registry_with_global):
        """With use_global=False and no parent, registering should reduce to a pure own-context
        (plus ridge) estimate — verified by reconstructing the dense covariance. Uses data with
        genuine rank-4 structure (a low-rank truncation of pure noise wouldn't correlate well
        with the full-rank covariance, since there'd be no real structure to capture)."""
        module = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        rng = np.random.default_rng(3)
        latent = rng.standard_normal((80, RANK)).astype(np.float32)
        loadings = rng.standard_normal((RANK, N_GENES)).astype(np.float32)
        counts = latent @ loadings + 0.01 * rng.standard_normal((80, N_GENES)).astype(np.float32)

        idx = module.register_context("dsB.cellZ", counts, kappa=1000.0, delta=0.05, use_global=False)

        # w_own = 80/(80+1000) ~= 0.074; scaled by (1-delta)=0.95 -> ~0.070; no pool -> alpha forced to 1.0
        # (see compute_shrinkage_weights: pool_total==0 => alpha=1.0), so this should be the
        # *unshrunk* own covariance (scaled by (1-delta)) + delta*I.
        expected = 0.95 * dense_covariance(counts) + 0.05 * np.eye(N_GENES)
        reconstructed = reconstruct(module, idx)
        # low-rank truncation at RANK=4 vs full rank (80 cells, 12 genes) — compare only the
        # dominant structure via Frobenius correlation rather than exact equality.
        assert np.corrcoef(reconstructed.flatten(), expected.flatten())[0, 1] > 0.9

    def test_uses_global_pool_automatically_when_present(self, registry_with_global):
        module = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        rng = np.random.default_rng(4)
        # Very few own cells -> heavy reliance on the global pool
        counts = rng.standard_normal((5, N_GENES)).astype(np.float32) * 10.0

        idx_with_global = module.register_context("dsB.tiny", counts, kappa=1000.0, use_global=True)
        reconstructed_with_global = reconstruct(module, idx_with_global)

        module2 = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        idx_without_global = module2.register_context("dsB.tiny", counts, kappa=1000.0, use_global=False)
        reconstructed_without_global = reconstruct(module2, idx_without_global)

        assert not np.allclose(reconstructed_with_global, reconstructed_without_global, atol=1e-4)

    def test_parent_context_pulls_toward_parent(self, registry_with_global):
        module = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        rng = np.random.default_rng(5)
        counts = rng.standard_normal((5, N_GENES)).astype(np.float32)

        idx = module.register_context(
            "dsA.cellZ", counts, kappa=1000.0, parent_context="dsA.cellX", use_global=False
        )
        reconstructed = reconstruct(module, idx)
        parent_cov = reconstruct(module, module.context_name_to_idx["dsA.cellX"])

        # With only 5 own cells, the parent pool should dominate — high correlation expected.
        assert np.corrcoef(reconstructed.flatten(), parent_cov.flatten())[0, 1] > 0.8

    def test_gene_column_realignment(self, registry_with_global):
        module = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        rng = np.random.default_rng(6)

        # Provide columns in reverse order, plus one extra unknown gene column.
        reversed_genes = list(reversed(module.gene_symbols)) + ["UNKNOWN_GENE"]
        counts_reversed = rng.standard_normal((40, N_GENES + 1)).astype(np.float32)

        idx = module.register_context("dsB.reversed", counts_reversed, gene_names=reversed_genes, use_global=False)
        assert module.V.shape[1] == N_GENES  # still aligned to registry gene count

    def test_gene_column_realignment_no_match_raises(self, registry_with_global):
        module = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        rng = np.random.default_rng(7)
        counts = rng.standard_normal((10, 3)).astype(np.float32)
        with pytest.raises(ValueError):
            module.register_context("bad", counts, gene_names=["NOT_A", "NOT_B", "NOT_C"])

    def test_reset_online_contexts(self, registry_with_global):
        module = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        n_before = module.n_contexts
        rng = np.random.default_rng(8)
        module.register_context("dsB.cellZ", rng.standard_normal((30, N_GENES)).astype(np.float32))
        assert module.n_contexts == n_before + 1

        module.reset_online_contexts()
        assert module.n_contexts == n_before
        assert "dsB.cellZ" not in module.context_name_to_idx
        assert module.V.shape[0] == n_before

    def test_registering_does_not_require_grad_tracking(self, registry_with_global):
        """register_context is inference-time only — should not build a grad graph."""
        module = CovariancePriorCrossAttention(registry_with_global, D_MODEL)
        rng = np.random.default_rng(9)
        counts = rng.standard_normal((20, N_GENES)).astype(np.float32)
        module.register_context("dsB.cellZ", counts)
        assert not module.V.requires_grad


class TestModelIntegration:
    def _make_model(self, registry_path):
        from state.tx.models.state_transition import StateTransitionPerturbationModel

        hidden_dim = D_MODEL
        return StateTransitionPerturbationModel(
            input_dim=8,
            hidden_dim=hidden_dim,
            output_dim=8,
            pert_dim=5,
            dropout=0.0,
            embed_key="X_hvg",
            use_cov_cross_attn=True,
            cov_registry_path=registry_path,
            cross_attn_freq=2,
            transformer_backbone_key="llama",
            transformer_backbone_kwargs={
                "bidirectional_attention": True,
                "hidden_size": hidden_dim,
                "intermediate_size": hidden_dim * 2,
                "num_hidden_layers": 4,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": hidden_dim // 4,
                "max_position_embeddings": 6,
                "use_cache": False,
            },
            n_encoder_layers=1,
            n_decoder_layers=1,
            cell_set_len=4,
            predict_residual=True,
            loss="energy",
            distributional_loss="energy",
        )

    def test_registered_context_usable_in_forward(self, registry_with_global):
        model = self._make_model(registry_with_global)
        rng = np.random.default_rng(10)
        new_counts = rng.standard_normal((25, N_GENES)).astype(np.float32) * 2.0

        model.register_covariance_context("newds.newtype", new_counts)
        assert "newds.newtype" in model._cov_context_name_to_idx  # shared dict reference updates live

        model.eval()
        S = 4
        batch = {
            "ctrl_cell_emb": torch.randn(S, 8),
            "pert_emb": torch.randn(S, 5),
            "pert_name": ["GENE0"] * S,
            "cell_type": ["newtype"] * S,
            "dataset_name": ["newds"] * S,
        }
        with torch.no_grad():
            out = model(batch, padded=True)
        assert out.shape == (S, 8)

    def test_reset_covariance_contexts_wrapper(self, registry_with_global):
        model = self._make_model(registry_with_global)
        rng = np.random.default_rng(11)
        model.register_covariance_context("newds.newtype", rng.standard_normal((20, N_GENES)).astype(np.float32))
        n_after_register = model.cov_module.n_contexts

        model.reset_covariance_contexts()
        assert model.cov_module.n_contexts == n_after_register - 1
        assert "newds.newtype" not in model.cov_module.context_name_to_idx

    def test_register_without_cov_module_raises(self):
        from state.tx.models.state_transition import StateTransitionPerturbationModel

        model = StateTransitionPerturbationModel(
            input_dim=8,
            hidden_dim=D_MODEL,
            output_dim=8,
            pert_dim=5,
            dropout=0.0,
            embed_key="X_hvg",
            use_cov_cross_attn=False,
            transformer_backbone_key="llama",
            transformer_backbone_kwargs={
                "bidirectional_attention": True,
                "hidden_size": D_MODEL,
                "intermediate_size": D_MODEL * 2,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 4,
                "head_dim": D_MODEL // 4,
                "max_position_embeddings": 6,
                "use_cache": False,
            },
            n_encoder_layers=1,
            n_decoder_layers=1,
            cell_set_len=4,
            predict_residual=True,
            loss="energy",
            distributional_loss="energy",
        )
        with pytest.raises(ValueError):
            model.register_covariance_context("x.y", np.zeros((5, 8), dtype=np.float32))
