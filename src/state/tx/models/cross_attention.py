from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class QuantumCellCrossAttentionLayer(nn.Module):
    """Pre-norm multi-head cross-attention + feed-forward residual block.

    Q: cell hidden states  (B, S, d_model)
    KV: projected QuantumCell gene embeddings  (B, N, d_model)
    """

    def __init__(self, d_model: int, nhead: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x_n = self.norm_q(x)
        kv_n = self.norm_kv(kv)
        x_attn, _ = self.cross_attn(x_n, kv_n, kv_n, key_padding_mask=key_padding_mask)
        x = x_attn + residual
        x = self.ff(self.ff_norm(x)) + x
        return x


class GeneEmbeddingCrossAttention(nn.Module):
    """Loads QuantumCell gene_embeddings_combined.npz and provides lookup for cross-attention KV.

    Two modes
    ---------
    "per_source"  (default, more expressive):
        22 separate Linear(source_dim_i → d_model) projections.
        Returns (B, 22, d_model) KV and (B, 22) key_padding_mask.
        The model can differentially attend to each biological source.

    "combined":
        Single Linear(total_dim → d_model) projection.
        Returns (B, 1, d_model) KV and (B, 1) key_padding_mask.
        Matches the approach described in EMBEDDINGS.md.
    """

    def __init__(
        self,
        emb_path: str,
        d_model: int,
        mode: Literal["per_source", "combined"] = "per_source",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.mode = mode
        self.d_model = d_model

        data = np.load(emb_path, allow_pickle=True)
        embedding: np.ndarray = data["embedding"].astype(np.float32)    # (G, D)
        mask_per_source: np.ndarray = data["mask_per_source"].astype(bool)  # (G, S)
        mask_any: np.ndarray = data["mask_any"].astype(bool)            # (G,)
        source_dims: np.ndarray = data["source_dims"].astype(np.int64)  # (S,)

        # Frozen buffers — not trained
        self.register_buffer("embedding", torch.from_numpy(embedding))
        self.register_buffer("mask_per_source", torch.from_numpy(mask_per_source))
        self.register_buffer("mask_any", torch.from_numpy(mask_any))
        self.register_buffer("source_dims", torch.from_numpy(source_dims))

        n_genes, total_dim = embedding.shape
        n_sources = len(source_dims)
        self.n_genes = n_genes
        self.n_sources = n_sources
        self.total_dim = total_dim

        if mode == "per_source":
            col_starts = np.concatenate([[0], np.cumsum(source_dims[:-1])]).tolist()
            self.col_starts: list[int] = [int(s) for s in col_starts]
            self.source_dims_list: list[int] = [int(d) for d in source_dims.tolist()]
            self.source_projs = nn.ModuleList(
                [nn.Linear(d, d_model) for d in self.source_dims_list]
            )
        else:
            self.combined_proj = nn.Linear(total_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def lookup(
        self, gene_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Look up and project QuantumCell embeddings for a batch of gene indices.

        Args:
            gene_indices: (B,) LongTensor. Use -1 for unknown genes.

        Returns:
            kv:              (B, N, d_model) — projected embedding tokens
            key_padding_mask:(B, N) bool — True = ignore this KV token (absent source / unknown gene)
                             None when no masking is needed.
        """
        B = gene_indices.shape[0]
        device = gene_indices.device

        # Clamp -1 (unknown) to 0 for indexing; we'll mask these out separately
        unknown_mask = gene_indices < 0                    # (B,) bool
        safe_indices = gene_indices.clamp(min=0)           # (B,)

        if self.mode == "per_source":
            kv_tokens = []
            for i, (start, dim, proj) in enumerate(
                zip(self.col_starts, self.source_dims_list, self.source_projs)
            ):
                src_emb = self.embedding[safe_indices, start : start + dim]  # (B, src_dim)
                kv_tokens.append(proj(src_emb))                              # (B, d_model)
            kv = torch.stack(kv_tokens, dim=1)  # (B, N, d_model)

            # padding mask: True where source is absent OR gene is unknown
            pad_mask = self.mask_per_source[safe_indices]   # (B, N) bool
            if unknown_mask.any():
                pad_mask = pad_mask.clone()
                pad_mask[unknown_mask] = True               # mask all sources for unknown genes

            # If no token is masked, return None to avoid unnecessary masking overhead
            key_padding_mask: Optional[torch.Tensor] = pad_mask if pad_mask.any() else None
        else:
            full_emb = self.embedding[safe_indices]          # (B, D)
            kv = self.combined_proj(full_emb).unsqueeze(1)  # (B, 1, d_model)

            # padding mask: True when gene absent from all sources or unknown
            pad_mask_1d = self.mask_any[safe_indices]        # (B,) bool
            if unknown_mask.any():
                pad_mask_1d = pad_mask_1d.clone()
                pad_mask_1d[unknown_mask] = True
            key_padding_mask = pad_mask_1d.unsqueeze(1) if pad_mask_1d.any() else None

        # Zero out hidden state for fully-masked genes to avoid NaN in attention
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=-1)        # (B,) — every token is masked
            if all_masked.any():
                kv = kv.clone()
                kv[all_masked] = 0.0
                # Use None mask for those genes so MHA doesn't produce NaN
                # We keep the mask but zero the values; MHA with all-True mask returns NaN,
                # so we set those rows' mask to False (attend to zeros) instead.
                key_padding_mask = key_padding_mask.clone()
                key_padding_mask[all_masked] = False

        kv = self.dropout(kv)
        return kv, key_padding_mask
