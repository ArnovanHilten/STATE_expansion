"""Tests for CovariancePriorCrossAttention and its integration into StateTransitionPerturbationModel."""

from pathlib import Path

import numpy as np
import pytest
import torch

from state.tx.models.cross_attention import CovariancePriorCrossAttention


N_CONTEXTS = 2
N_GENES = 10
RANK = 4
D_MODEL = 16


@pytest.fixture
def mock_registry(tmp_path: Path) -> str:
    """Create a minimal covariance_registry.npz for testing."""
    rng = np.random.default_rng(0)

    # Orthonormal-ish per-context loadings (doesn't need to be exact for shape/behavior tests)
    V = rng.standard_normal((N_CONTEXTS, N_GENES, RANK)).astype(np.float16)
    eigenvalues = np.abs(rng.standard_normal((N_CONTEXTS, RANK))).astype(np.float32) + 1.0
    gene_variance = np.abs(rng.standard_normal((N_CONTEXTS, N_GENES))).astype(np.float32)
    context_names = np.array(["dsA.cellX", "dsA.cellY"])
    gene_symbols = np.array([f"GENE{i}" for i in range(N_GENES)])

    out = tmp_path / "covariance_registry.npz"
    np.savez_compressed(
        out,
        V=V,
        eigenvalues=eigenvalues,
        gene_variance=gene_variance,
        context_names=context_names,
        gene_symbols=gene_symbols,
        n_control_cells=np.array([115, 480], dtype=np.int64),
    )
    return str(out)


# ---------------------------------------------------------------------------
# CovariancePriorCrossAttention unit tests
# ---------------------------------------------------------------------------


class TestCovariancePriorCrossAttention:
    def test_lookup_shape(self, mock_registry):
        module = CovariancePriorCrossAttention(mock_registry, D_MODEL)
        context_idx = torch.tensor([0, 1, 0])
        gene_idx = torch.tensor([1, 2, 3])
        kv, mask = module.lookup(context_idx, gene_idx)
        assert kv.shape == (3, 1, D_MODEL)
        assert mask is None

    def test_same_gene_different_context_differs(self, mock_registry):
        """Core design principle: z_cov(c1, j) != z_cov(c2, j) for the same gene j."""
        module = CovariancePriorCrossAttention(mock_registry, D_MODEL)
        module.eval()
        gene_idx = torch.tensor([3, 3])
        context_idx = torch.tensor([0, 1])
        with torch.no_grad():
            kv, _ = module.lookup(context_idx, gene_idx)
        assert not torch.allclose(kv[0], kv[1]), "Same gene in different contexts should differ"

    def test_unknown_context_or_gene_uses_no_cov_token(self, mock_registry):
        module = CovariancePriorCrossAttention(mock_registry, D_MODEL)
        module.eval()
        with torch.no_grad():
            kv_unknown_ctx, _ = module.lookup(torch.tensor([-1]), torch.tensor([2]))
            kv_unknown_gene, _ = module.lookup(torch.tensor([0]), torch.tensor([-1]))
            expected = module.no_cov_token
        assert torch.allclose(kv_unknown_ctx[0, 0], expected[0])
        assert torch.allclose(kv_unknown_gene[0, 0], expected[0])

    def test_gradient_flows(self, mock_registry):
        module = CovariancePriorCrossAttention(mock_registry, D_MODEL)
        # Include one unknown pair so both the encoder and the [NO_COV] fallback
        # token receive gradient.
        context_idx = torch.tensor([0, 1, -1])
        gene_idx = torch.tensor([1, 2, -1])
        kv, _ = module.lookup(context_idx, gene_idx)
        kv.sum().backward()
        for name, p in module.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_embedding_buffers_not_trainable(self, mock_registry):
        module = CovariancePriorCrossAttention(mock_registry, D_MODEL)
        param_names = {n for n, _ in module.named_parameters()}
        assert "V" not in param_names
        assert "log_eigenvalues" not in param_names
        assert "gene_variance" not in param_names

    def test_cov_dropout_none_forces_fallback(self, mock_registry):
        """cov_dropout={'none': 1.0} should always force the [NO_COV] token during training."""
        module = CovariancePriorCrossAttention(mock_registry, D_MODEL, cov_dropout={"none": 1.0})
        module.train()
        context_idx = torch.tensor([0, 1, 0, 1])
        gene_idx = torch.tensor([1, 2, 3, 4])
        with torch.no_grad():
            kv, _ = module.lookup(context_idx, gene_idx)
        for i in range(4):
            assert torch.allclose(kv[i, 0], module.no_cov_token[0])

    def test_cov_dropout_disabled_in_eval(self, mock_registry):
        """cov_dropout must not corrupt lookups at eval time even if configured."""
        module = CovariancePriorCrossAttention(mock_registry, D_MODEL, cov_dropout={"none": 1.0})
        module.eval()
        context_idx = torch.tensor([0])
        gene_idx = torch.tensor([1])
        with torch.no_grad():
            kv, _ = module.lookup(context_idx, gene_idx)
        assert not torch.allclose(kv[0, 0], module.no_cov_token[0])


# ---------------------------------------------------------------------------
# Integration: end-to-end forward pass through StateTransitionPerturbationModel
# ---------------------------------------------------------------------------


def _make_st_model(mock_registry: str, use_qc: bool = False, qc_emb_path: str | None = None):
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    hidden_dim = D_MODEL
    n_layers = 6
    nhead = 4
    cell_set_len = 4

    kwargs = dict(
        input_dim=8,
        hidden_dim=hidden_dim,
        output_dim=8,
        pert_dim=5,
        dropout=0.0,
        embed_key="X_hvg",
        use_cov_cross_attn=True,
        cov_registry_path=mock_registry,
        cross_attn_freq=2,
        transformer_backbone_key="llama",
        transformer_backbone_kwargs={
            "bidirectional_attention": True,
            "hidden_size": hidden_dim,
            "intermediate_size": hidden_dim * 2,
            "num_hidden_layers": n_layers,
            "num_attention_heads": nhead,
            "num_key_value_heads": nhead,
            "head_dim": hidden_dim // nhead,
            "max_position_embeddings": cell_set_len + 2,
            "use_cache": False,
        },
        n_encoder_layers=1,
        n_decoder_layers=1,
        cell_set_len=cell_set_len,
        predict_residual=True,
        loss="energy",
        distributional_loss="energy",
    )
    if use_qc:
        kwargs["use_qc_cross_attn"] = True
        kwargs["qc_emb_path"] = qc_emb_path
        kwargs["qc_mode"] = "combined"

    return StateTransitionPerturbationModel(**kwargs)


def test_st_forward_with_cov_cross_attn(mock_registry):
    model = _make_st_model(mock_registry)
    model.eval()

    B, S = 2, 4
    batch = {
        "ctrl_cell_emb": torch.randn(B * S, 8),
        "pert_emb": torch.randn(B * S, 5),
        "pert_name": ["GENE1"] * S + ["GENE2"] * S,
        "cell_type": ["cellX"] * S + ["cellY"] * S,
        "dataset_name": ["dsA"] * (B * S),
    }
    with torch.no_grad():
        out = model(batch, padded=True)
    assert out.shape == (B * S, 8)


def test_st_cov_context_changes_output(mock_registry):
    """Same perturbed gene, different cell_type context, should give different predictions."""
    model = _make_st_model(mock_registry)
    model.eval()

    S = 4
    base_batch = {
        "ctrl_cell_emb": torch.randn(S, 8),
        "pert_emb": torch.randn(S, 5),
        "pert_name": ["GENE3"] * S,
        "dataset_name": ["dsA"] * S,
    }
    batch_x = {**base_batch, "cell_type": ["cellX"] * S}
    batch_y = {**base_batch, "cell_type": ["cellY"] * S}

    with torch.no_grad():
        out_x = model(batch_x, padded=True)
        out_y = model(batch_y, padded=True)

    assert not torch.allclose(out_x, out_y), "Different context should yield different output"


def test_st_cov_unknown_context_falls_back(mock_registry):
    """Missing cell_type/dataset_name info should not crash — falls back to [NO_COV]."""
    model = _make_st_model(mock_registry)
    model.eval()

    B, S = 1, 4
    batch = {
        "ctrl_cell_emb": torch.randn(B * S, 8),
        "pert_emb": torch.randn(B * S, 5),
        "pert_name": ["GENE1"] * S,
        # no cell_type, no dataset_name
    }
    with torch.no_grad():
        out = model(batch, padded=True)
    assert out.shape == (B * S, 8)


def test_st_gradient_flows_through_cov_cross_attn(mock_registry):
    model = _make_st_model(mock_registry)
    model.train()

    B, S = 3, 4
    batch = {
        "ctrl_cell_emb": torch.randn(B * S, 8),
        "pert_emb": torch.randn(B * S, 5),
        "pert_name": ["GENE1"] * S + ["GENE2"] * S + ["UNKNOWN_GENE"] * S,
        "cell_type": ["cellX"] * S + ["cellY"] * S + ["cellX"] * S,
        "dataset_name": ["dsA"] * (B * S),
    }
    out = model(batch, padded=True)
    out.sum().backward()

    for name, p in model.cov_module.named_parameters():
        assert p.grad is not None, f"No gradient for cov_module.{name}"
    for name, p in model.cross_attn_layers.named_parameters():
        assert p.grad is not None, f"No gradient for cross_attn_layers.{name}"


def test_st_qc_and_cov_combined(mock_registry, tmp_path):
    """Both static (QuantumCell) and covariance-prior cross-attention enabled together."""
    rng = np.random.default_rng(1)
    n_genes, total_dim = N_GENES, 5
    embedding = rng.standard_normal((n_genes, total_dim)).astype(np.float32)
    mask_any = np.zeros((n_genes,), dtype=bool)
    gene_symbols = np.array([f"GENE{i}" for i in range(n_genes)])
    qc_path = tmp_path / "qc.npz"
    np.savez_compressed(
        qc_path,
        embedding=embedding,
        mask_per_source=np.zeros((n_genes, 1), dtype=bool),
        mask_any=mask_any,
        gene_ids=gene_symbols,
        gene_symbols=gene_symbols,
        source_names=np.array(["src0"]),
        source_dims=np.array([total_dim], dtype=np.int32),
    )

    model = _make_st_model(mock_registry, use_qc=True, qc_emb_path=str(qc_path))
    model.eval()

    B, S = 2, 4
    batch = {
        "ctrl_cell_emb": torch.randn(B * S, 8),
        "pert_emb": torch.randn(B * S, 5),
        "pert_name": ["GENE1"] * S + ["GENE2"] * S,
        "cell_type": ["cellX"] * S + ["cellY"] * S,
        "dataset_name": ["dsA"] * (B * S),
    }
    with torch.no_grad():
        out = model(batch, padded=True)
    assert out.shape == (B * S, 8)

    # 1 QC source ("combined" mode) + 1 covariance token = 2 KV tokens total
    model.enable_attn_weight_collection(True)
    with torch.no_grad():
        model(batch, padded=True)
    assert model._last_qc_attn_weights is not None
    assert model._last_qc_attn_weights.shape[-1] == 2
