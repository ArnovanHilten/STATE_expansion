"""Tests for GeneEmbeddingCrossAttention and QuantumCell cross-attention integration."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from state.tx.models.cross_attention import GeneEmbeddingCrossAttention, QuantumCellCrossAttentionLayer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_GENES = 10
SOURCE_DIMS = [4, 6, 3]  # 3 small sources; total_dim = 13
N_SOURCES = len(SOURCE_DIMS)
TOTAL_DIM = sum(SOURCE_DIMS)
D_MODEL = 16


@pytest.fixture
def mock_npz(tmp_path: Path) -> str:
    """Create a minimal gene_embeddings_combined.npz for testing."""
    rng = np.random.default_rng(0)
    embedding = rng.standard_normal((N_GENES, TOTAL_DIM)).astype(np.float32)
    # L2-normalise rows
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embedding /= norms

    # Make gene 5 missing from source 1 and gene 7 missing from all sources
    mask_per_source = np.zeros((N_GENES, N_SOURCES), dtype=bool)
    mask_per_source[5, 1] = True
    mask_per_source[7, :] = True

    mask_any = mask_per_source.all(axis=1)
    gene_ids = np.array([f"ENSG{i:011d}" for i in range(N_GENES)])
    source_names = np.array([f"src{i}" for i in range(N_SOURCES)])
    source_dims = np.array(SOURCE_DIMS, dtype=np.int32)

    out = tmp_path / "gene_embeddings_combined.npz"
    np.savez_compressed(
        out,
        embedding=embedding,
        mask_per_source=mask_per_source,
        mask_any=mask_any,
        gene_ids=gene_ids,
        source_names=source_names,
        source_dims=source_dims,
    )
    return str(out)


# ---------------------------------------------------------------------------
# QuantumCellCrossAttentionLayer tests
# ---------------------------------------------------------------------------

def test_cross_attn_layer_output_shape():
    layer = QuantumCellCrossAttentionLayer(D_MODEL, nhead=4)
    B, S, N = 2, 8, N_SOURCES
    x = torch.randn(B, S, D_MODEL)
    kv = torch.randn(B, N, D_MODEL)
    out = layer(x, kv)
    assert out.shape == (B, S, D_MODEL)


def test_cross_attn_layer_with_padding_mask():
    layer = QuantumCellCrossAttentionLayer(D_MODEL, nhead=4)
    B, S, N = 2, 8, N_SOURCES
    x = torch.randn(B, S, D_MODEL)
    kv = torch.randn(B, N, D_MODEL)
    mask = torch.zeros(B, N, dtype=torch.bool)
    mask[0, -1] = True  # mask last source for first batch item
    out = layer(x, kv, key_padding_mask=mask)
    assert out.shape == (B, S, D_MODEL)


def test_cross_attn_layer_gradient_flows():
    layer = QuantumCellCrossAttentionLayer(D_MODEL, nhead=4)
    x = torch.randn(2, 8, D_MODEL, requires_grad=True)
    kv = torch.randn(2, N_SOURCES, D_MODEL)
    out = layer(x, kv)
    out.sum().backward()
    assert x.grad is not None
    for p in layer.parameters():
        assert p.grad is not None


# ---------------------------------------------------------------------------
# GeneEmbeddingCrossAttention tests
# ---------------------------------------------------------------------------

class TestGeneEmbeddingCrossAttentionPerSource:
    def test_lookup_shape(self, mock_npz):
        module = GeneEmbeddingCrossAttention(mock_npz, D_MODEL, mode="per_source")
        gene_idx = torch.tensor([0, 1, 2])
        kv, mask = module.lookup(gene_idx)
        assert kv.shape == (3, N_SOURCES, D_MODEL)

    def test_lookup_unknown_gene(self, mock_npz):
        module = GeneEmbeddingCrossAttention(mock_npz, D_MODEL, mode="per_source")
        gene_idx = torch.tensor([-1, 0])
        kv, mask = module.lookup(gene_idx)
        assert kv.shape == (2, N_SOURCES, D_MODEL)
        # Unknown gene (-1): KV is zeroed out to avoid NaN in MHA when all sources would be masked
        assert (kv[0] == 0).all(), "KV for unknown gene should be zeroed"

    def test_lookup_missing_source_masked(self, mock_npz):
        module = GeneEmbeddingCrossAttention(mock_npz, D_MODEL, mode="per_source")
        gene_idx = torch.tensor([5])  # gene 5 has source 1 missing
        kv, mask = module.lookup(gene_idx)
        assert mask is not None
        assert mask[0, 1], "Source 1 of gene 5 should be masked"

    def test_lookup_fully_absent_gene_no_nan(self, mock_npz):
        module = GeneEmbeddingCrossAttention(mock_npz, D_MODEL, mode="per_source")
        gene_idx = torch.tensor([7])  # absent from all sources
        kv, mask = module.lookup(gene_idx)
        # Should not produce NaN — all-masked row is zeroed and mask is cleared
        assert not torch.isnan(kv).any()
        assert mask is None or not mask[0].all()

    def test_projections_are_trainable(self, mock_npz):
        module = GeneEmbeddingCrossAttention(mock_npz, D_MODEL, mode="per_source")
        trainable = [p for p in module.parameters() if p.requires_grad]
        assert len(trainable) > 0, "Projection layers should be trainable"

    def test_embedding_buffer_not_trainable(self, mock_npz):
        module = GeneEmbeddingCrossAttention(mock_npz, D_MODEL, mode="per_source")
        # Buffers are not parameters — check the embedding is not in parameters
        param_names = {n for n, _ in module.named_parameters()}
        assert "embedding" not in param_names


class TestGeneEmbeddingCrossAttentionCombined:
    def test_lookup_shape(self, mock_npz):
        module = GeneEmbeddingCrossAttention(mock_npz, D_MODEL, mode="combined")
        gene_idx = torch.tensor([0, 1, 2])
        kv, mask = module.lookup(gene_idx)
        assert kv.shape == (3, 1, D_MODEL)

    def test_lookup_unknown_gene_masked(self, mock_npz):
        module = GeneEmbeddingCrossAttention(mock_npz, D_MODEL, mode="combined")
        gene_idx = torch.tensor([-1])
        kv, mask = module.lookup(gene_idx)
        # Unknown gene: KV is zeroed to prevent NaN in MHA
        assert (kv[0] == 0).all(), "KV for unknown gene should be zeroed in combined mode"


# ---------------------------------------------------------------------------
# Integration: end-to-end forward pass through StateTransitionPerturbationModel
# ---------------------------------------------------------------------------

def _make_st_model(mock_npz: str, qc_mode: str = "per_source"):
    """Build a tiny StateTransitionPerturbationModel with QC cross-attention enabled."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    hidden_dim = D_MODEL
    n_layers = 6
    nhead = 4
    cell_set_len = 4

    return StateTransitionPerturbationModel(
        input_dim=8,
        hidden_dim=hidden_dim,
        output_dim=8,
        pert_dim=5,
        dropout=0.0,
        embed_key="X_hvg",
        # QC cross-attention
        use_qc_cross_attn=True,
        qc_emb_path=mock_npz,
        qc_mode=qc_mode,
        cross_attn_freq=2,
        # Backbone
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


@pytest.mark.parametrize("qc_mode", ["per_source", "combined"])
def test_st_forward_with_qc_cross_attn(mock_npz, qc_mode):
    model = _make_st_model(mock_npz, qc_mode=qc_mode)
    model.eval()

    B, S = 2, 4
    batch = {
        "ctrl_cell_emb": torch.randn(B * S, 8),
        "pert_emb": torch.randn(B * S, 5),
        "pert_gene_idx": torch.tensor([1, 1, 1, 1, 3, 3, 3, 3], dtype=torch.long),
    }
    with torch.no_grad():
        out = model(batch, padded=True)
    assert out.shape == (B * S, 8), f"Expected ({B*S}, 8), got {out.shape}"


@pytest.mark.parametrize("qc_mode", ["per_source", "combined"])
def test_st_gradient_flows_through_cross_attn(mock_npz, qc_mode):
    model = _make_st_model(mock_npz, qc_mode=qc_mode)
    model.train()

    B, S = 2, 4
    batch = {
        "ctrl_cell_emb": torch.randn(B * S, 8),
        "pert_emb": torch.randn(B * S, 5),
        "pert_gene_idx": torch.tensor([0] * (B * S), dtype=torch.long),
    }
    out = model(batch, padded=True)
    loss = out.sum()
    loss.backward()

    # Check gradients flow into cross-attention layers
    for name, p in model.cross_attn_layers.named_parameters():
        assert p.grad is not None, f"No gradient for cross_attn_layers.{name}"

    # Check gradients flow into QC projection layers
    for name, p in model.qc_module.named_parameters():
        assert p.grad is not None, f"No gradient for qc_module.{name}"


def test_st_without_pert_gene_idx_fallback(mock_npz):
    """When pert_gene_idx is absent from batch, the model should still run (using -1 idx)."""
    model = _make_st_model(mock_npz)
    model.eval()

    B, S = 2, 4
    batch = {
        "ctrl_cell_emb": torch.randn(B * S, 8),
        "pert_emb": torch.randn(B * S, 5),
        # no pert_gene_idx
    }
    with torch.no_grad():
        out = model(batch, padded=True)
    assert out.shape == (B * S, 8)


def test_st_backward_compat_without_qc(mock_npz):
    """Model instantiated without use_qc_cross_attn should behave exactly as before."""
    from state.tx.models.state_transition import StateTransitionPerturbationModel

    model = StateTransitionPerturbationModel(
        input_dim=8,
        hidden_dim=D_MODEL,
        output_dim=8,
        pert_dim=5,
        dropout=0.0,
        embed_key="X_hvg",
        use_qc_cross_attn=False,
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
    model.eval()
    assert model.qc_module is None
    assert model.cross_attn_layers is None

    B, S = 2, 4
    batch = {
        "ctrl_cell_emb": torch.randn(B * S, 8),
        "pert_emb": torch.randn(B * S, 5),
    }
    with torch.no_grad():
        out = model(batch, padded=True)
    assert out.shape == (B * S, 8)
