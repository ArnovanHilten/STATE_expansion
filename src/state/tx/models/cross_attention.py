from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .covariance_math import (
    compute_shrinkage_weights,
    stack_and_decompose,
    weighted_block_from_counts,
    weighted_block_from_low_rank,
)

GLOBAL_CONTEXT_NAME = "__global__"  # matches covariance_prior/build_covariance_registry.py


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
        # Set to True externally to enable weight collection (eval only).
        self._collect_attn_weights: bool = False
        self._last_attn_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        x: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x_n = self.norm_q(x)
        kv_n = self.norm_kv(kv)

        # When all KV tokens in a row are masked, softmax(-inf, ..., -inf) = NaN.
        # nan_to_num fixes the forward but NaN gradients still flow in the backward.
        # Instead: temporarily unmask slot 0 for those rows so softmax stays finite,
        # then zero out their attention output so no signal leaks (KV is already
        # zeroed by GeneEmbeddingCrossAttention.lookup for these rows).
        null_rows: Optional[torch.Tensor] = None
        safe_mask = key_padding_mask
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=-1)  # (B,)
            if all_masked.any():
                safe_mask = key_padding_mask.clone()
                safe_mask[all_masked, 0] = False       # unmask slot 0 → finite softmax
                null_rows = all_masked

        if not self.training and self._collect_attn_weights:
            # need_weights=True disables flash-attn; only triggered when explicitly requested.
            x_attn, attn_w = self.cross_attn(
                x_n, kv_n, kv_n,
                key_padding_mask=safe_mask,
                need_weights=True,
                average_attn_weights=True,  # average over heads → (B, S_q, N_kv)
            )
            self._last_attn_weights = attn_w.detach()  # (B, S_q, N_sources)
        else:
            x_attn, _ = self.cross_attn(x_n, kv_n, kv_n, key_padding_mask=safe_mask)
            self._last_attn_weights = None

        if null_rows is not None:
            x_attn = x_attn.masked_fill(null_rows[:, None, None], 0.0)
            if self._last_attn_weights is not None:
                self._last_attn_weights = self._last_attn_weights.masked_fill(
                    null_rows[:, None, None], 0.0
                )

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
        self,
        gene_indices: torch.Tensor,
        ablate_source: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Look up and project QuantumCell embeddings for a batch of gene indices.

        Args:
            gene_indices:  (B,) LongTensor. Use -1 for unknown genes.
            ablate_source: When set to an integer 0..N_sources-1, force-masks that
                           source column so the model cannot attend to it. Used for
                           source ablation importance analysis.

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

            # Source ablation: force-mask the chosen source column for every gene
            if ablate_source is not None:
                pad_mask = pad_mask.clone()
                pad_mask[:, ablate_source] = True

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

        # Zero out KV for fully-masked genes to save computation.
        # Keep the all-True mask intact — the resulting NaN in the attention output
        # is replaced with 0 in QuantumCellCrossAttentionLayer.forward, giving a
        # pure residual pass-through without LayerNorm-bias leakage.
        if key_padding_mask is not None:
            all_masked = key_padding_mask.all(dim=-1)        # (B,) — every token is masked
            if all_masked.any():
                kv = kv.clone()
                kv[all_masked] = 0.0

        kv = self.dropout(kv)
        return kv, key_padding_mask


class CovariancePriorCrossAttention(nn.Module):
    """CIPHER-inspired covariance prior: a context-conditioned cross-attention KV source.

    Loads a precomputed covariance registry (see scripts/build_covariance_registry.py) holding,
    for each biological context c (e.g. cell type), a low-rank decomposition of the control-cell
    covariance matrix Sigma_c ~= V_c,k @ diag(Lambda_c,k) @ V_c,k^T. For a perturbed gene j in
    context c, the Sigma_c[:, j] column is a first-order template for how the transcriptome moves
    when j is perturbed (Sigma u ~= dX under the CIPHER linear-response model). Rather than feed
    the full column, we expose the gene's coordinates within the dominant covariance modes:

        f_cov(c, j) = [V_c,k[j, :], log(Lambda_c,k), Var_c(j)]
        z_cov(c, j) = MLP(f_cov(c, j))

    Unlike GeneEmbeddingCrossAttention (one static embedding per gene), the same gene gets a
    different token depending on context: z_cov(HepG2, TP53) != z_cov(Jurkat_active, TP53).

    Returns a single KV token per (context, gene) pair. Unknown context/gene pairs fall back to
    a learned [NO_COV] token (rather than being masked out), so the model can freely mix known and
    unknown covariance information across a batch.
    """

    def __init__(
        self,
        registry_path: str,
        d_model: int,
        dropout: float = 0.0,
        cov_dropout: Optional[dict] = None,
    ):
        super().__init__()
        data = np.load(registry_path, allow_pickle=True)
        V: np.ndarray = data["V"].astype(np.float32)                    # (C, G, k)
        eigenvalues: np.ndarray = data["eigenvalues"].astype(np.float32)  # (C, k)
        gene_variance: np.ndarray = data["gene_variance"].astype(np.float32)  # (C, G)
        context_names: list[str] = [str(c) for c in data["context_names"]]
        gene_symbols: list[str] = [str(g) for g in data["gene_symbols"]]
        if "n_control_cells" in data:
            n_control_cells: np.ndarray = data["n_control_cells"].astype(np.float32)  # (C,)
        else:
            n_control_cells = np.zeros((len(context_names),), dtype=np.float32)

        n_contexts, n_genes, k = V.shape
        self.n_contexts = n_contexts
        self.n_genes = n_genes
        self.k = k
        self.d_model = d_model
        self.context_names = context_names
        self.gene_symbols = gene_symbols
        self.context_name_to_idx = {name: i for i, name in enumerate(context_names)}
        self.gene_name_to_idx = {name: i for i, name in enumerate(gene_symbols)}
        # Contexts registered offline (via build_covariance_registry.py) vs. at runtime (via
        # register_context()) — reset_online_contexts() truncates back to this boundary.
        self._n_registry_contexts = n_contexts

        # Frozen buffers — the registry is precomputed offline, not trained. They CAN grow at
        # runtime (see register_context()): reassigning an nn.Module attribute that was
        # registered via register_buffer still updates the buffer in place, it isn't limited to
        # its original shape.
        self.register_buffer("V", torch.from_numpy(V))
        self.register_buffer(
            "log_eigenvalues", torch.from_numpy(np.log(np.clip(eigenvalues, 1e-8, None)))
        )
        self.register_buffer("gene_variance", torch.from_numpy(gene_variance))
        self.register_buffer("n_control_cells", torch.from_numpy(n_control_cells))

        feat_dim = 2 * k + 1  # V[j,:] (k) + log(Lambda_c) (k) + Var_c(j) (1)
        self.encoder = nn.Sequential(
            nn.Linear(feat_dim, d_model * 4),
            nn.GELU(),
            nn.LayerNorm(d_model * 4),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model),
        )
        # Learned fallback for unknown (context, gene) pairs — PDF's [NO_COV] token.
        self.no_cov_token = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)

        # Covariance-dropout mismatch-training probabilities (train-time only). Keys:
        # "none" (fall back to [NO_COV]), "wrong_context" (random other context),
        # "shuffled_gene" (random other gene in the same context). Remaining probability
        # mass keeps the correct (context, gene) pair. See PDF "Covariance dropout and
        # mismatch training".
        self.cov_dropout = cov_dropout or {}

    def _apply_cov_dropout(
        self, context_idx: torch.Tensor, gene_idx: torch.Tensor, known: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly corrupt (context, gene) pairs during training to prevent over-reliance."""
        if not self.cov_dropout:
            return context_idx, gene_idx, known

        p_none = float(self.cov_dropout.get("none", 0.0))
        p_wrong_context = float(self.cov_dropout.get("wrong_context", 0.0))
        p_shuffled_gene = float(self.cov_dropout.get("shuffled_gene", 0.0))

        device = context_idx.device
        B = context_idx.shape[0]
        u = torch.rand(B, device=device)

        none_mask = u < p_none
        wrong_context_mask = (u >= p_none) & (u < p_none + p_wrong_context)
        shuffled_gene_mask = (u >= p_none + p_wrong_context) & (
            u < p_none + p_wrong_context + p_shuffled_gene
        )

        context_idx = context_idx.clone()
        gene_idx = gene_idx.clone()
        known = known & ~none_mask  # "none" forces the [NO_COV] fallback

        if wrong_context_mask.any() and self.n_contexts > 1:
            rand_ctx = torch.randint(0, self.n_contexts, (B,), device=device)
            # Ensure it's actually a *different* context where possible.
            same = rand_ctx == context_idx
            rand_ctx = torch.where(same, (rand_ctx + 1) % self.n_contexts, rand_ctx)
            context_idx = torch.where(wrong_context_mask, rand_ctx, context_idx)

        if shuffled_gene_mask.any() and self.n_genes > 1:
            rand_gene = torch.randint(0, self.n_genes, (B,), device=device)
            same = rand_gene == gene_idx
            rand_gene = torch.where(same, (rand_gene + 1) % self.n_genes, rand_gene)
            gene_idx = torch.where(shuffled_gene_mask, rand_gene, gene_idx)

        return context_idx, gene_idx, known

    def lookup(
        self, context_idx: torch.Tensor, gene_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Look up and encode the covariance prior for a batch of (context, gene) pairs.

        Args:
            context_idx: (B,) LongTensor. Use -1 for unknown/unavailable context.
            gene_idx:    (B,) LongTensor. Use -1 for unknown gene.

        Returns:
            kv:               (B, 1, d_model) — single covariance-prior token per item.
            key_padding_mask: always None — unknown pairs use the learned [NO_COV] token
                               (substitution) rather than being masked out of attention.
        """
        B = context_idx.shape[0]
        known = (context_idx >= 0) & (gene_idx >= 0)

        if self.training:
            context_idx, gene_idx, known = self._apply_cov_dropout(context_idx, gene_idx, known)

        safe_c = context_idx.clamp(min=0)
        safe_g = gene_idx.clamp(min=0)

        v = self.V[safe_c, safe_g]                    # (B, k)
        log_lambda = self.log_eigenvalues[safe_c]      # (B, k)
        var = self.gene_variance[safe_c, safe_g].unsqueeze(-1)  # (B, 1)
        feat = torch.cat([v, log_lambda, var], dim=-1)  # (B, 2k+1)

        z = self.encoder(feat)  # (B, d_model)
        if (~known).any():
            z = torch.where(known.unsqueeze(-1), z, self.no_cov_token.expand(B, -1))

        kv = self.dropout(z).unsqueeze(1)  # (B, 1, d_model)
        return kv, None

    @torch.no_grad()
    def register_context(
        self,
        context_name: str,
        control_counts: "np.ndarray | torch.Tensor",
        gene_names: Optional[list] = None,
        kappa: float = 1000.0,
        delta: float = 0.05,
        parent_context: Optional[str] = None,
        use_global: bool = True,
    ) -> int:
        """Compute a shrunk low-rank covariance on the fly from a new context's control cells
        and add it to the registry in memory, so it's immediately usable by name in `lookup()`.

        This is the spec's inference-time "new context" workflow: given a target cell
        population's control/non-targeting cells (no perturbation labels needed), estimate
        Sigma_target and register it — no offline precompute script rerun required.

        The same shrinkage math as build_covariance_registry.py applies (own/parent/global +
        ridge, weighted by w_own = n_own/(n_own+kappa)), but "parent"/"global" here are pulled
        from *already-registered* covariances (registry entries or previously-registered online
        contexts) via the "pseudo-cell" trick in covariance_math.weighted_block_from_low_rank —
        an existing (V, Lambda) is reused as an exact shrinkage target without needing its
        original raw cell data. If this registry has a "__global__" pooled pseudo-context (see
        build_covariance_registry.py), it's used automatically unless `use_global=False`.

        Args:
            context_name: key this context will be resolved by, e.g. "{dataset}.{cell_type}" —
                matches what StateTransitionPerturbationModel._resolve_context_idx builds from
                batch["dataset_name"] + batch["cell_type"].
            control_counts: (N, G_in) raw (unnormalized) control-cell counts.
            gene_names: (G_in,) gene symbols/IDs for control_counts' columns, if not already in
                this module's `self.gene_symbols` order. Columns not found in the registry's
                vocabulary are dropped; registry genes missing from `gene_names` are left at 0
                for this context (same convention as GeneEmbeddingCrossAttention's missing-source
                handling).
            kappa: shrinkage half-scale, w_own = n_own / (n_own + kappa).
            delta: identity-ridge floor.
            parent_context: name of an existing (registry or online) context to treat as the
                "parent" shrinkage pool, e.g. the nearest known sibling cell type. Its own
                n_control_cells is used as the effective pool size for weighting.
            use_global: blend toward this registry's "__global__" pooled pseudo-context, if
                present, as the "global" shrinkage pool.

        Returns:
            The new context's index (also immediately available via
            `self.context_name_to_idx[context_name]`).
        """
        if isinstance(control_counts, torch.Tensor):
            counts_np = control_counts.detach().cpu().float().numpy()
        else:
            counts_np = np.asarray(control_counts, dtype=np.float32)

        if gene_names is not None:
            counts_np = self._align_gene_columns(counts_np, gene_names)

        n_own = counts_np.shape[0]

        parent_idx = self.context_name_to_idx.get(parent_context) if parent_context else None
        global_idx = self.context_name_to_idx.get(GLOBAL_CONTEXT_NAME) if use_global else None

        n_parent = float(self.n_control_cells[parent_idx].item()) if parent_idx is not None else 0.0
        n_global = float(self.n_control_cells[global_idx].item()) if global_idx is not None else 0.0

        weights = compute_shrinkage_weights(n_own, n_parent, n_global, kappa, delta)

        own_block, own_var = weighted_block_from_counts(counts_np, weights.own)
        blocks = [own_block]
        variance_terms = [(weights.own, own_var)]

        if parent_idx is not None:
            parent_block = weighted_block_from_low_rank(
                self.V[parent_idx].cpu().numpy(),
                self.log_eigenvalues[parent_idx].exp().cpu().numpy(),
                weights.parent,
            )
            blocks.append(parent_block)
            variance_terms.append((weights.parent, self.gene_variance[parent_idx].cpu().numpy()))

        if global_idx is not None:
            global_block = weighted_block_from_low_rank(
                self.V[global_idx].cpu().numpy(),
                self.log_eigenvalues[global_idx].exp().cpu().numpy(),
                weights.global_,
            )
            blocks.append(global_block)
            variance_terms.append((weights.global_, self.gene_variance[global_idx].cpu().numpy()))

        rank = self.k
        V_new, eigenvalues_new, gene_variance_new = stack_and_decompose(
            blocks, variance_terms, rank, weights.delta, n_genes=self.n_genes, label=context_name
        )

        device = self.V.device
        new_idx = self.n_contexts
        self.V = torch.cat([self.V, torch.from_numpy(V_new).to(device).unsqueeze(0)], dim=0)
        self.log_eigenvalues = torch.cat(
            [
                self.log_eigenvalues,
                torch.from_numpy(np.log(np.clip(eigenvalues_new, 1e-8, None))).to(device).unsqueeze(0),
            ],
            dim=0,
        )
        self.gene_variance = torch.cat(
            [self.gene_variance, torch.from_numpy(gene_variance_new).to(device).unsqueeze(0)], dim=0
        )
        self.n_control_cells = torch.cat(
            [self.n_control_cells, torch.tensor([float(n_own)], device=device)], dim=0
        )
        self.context_names.append(context_name)
        self.context_name_to_idx[context_name] = new_idx
        self.n_contexts += 1
        return new_idx

    def _align_gene_columns(self, counts: np.ndarray, gene_names: list) -> np.ndarray:
        """Reindex `counts`' columns from `gene_names` order to `self.gene_symbols` order,
        zero-filling registry genes absent from `gene_names`."""
        aligned = np.zeros((counts.shape[0], self.n_genes), dtype=np.float32)
        incoming_idx = {str(g): i for i, g in enumerate(gene_names)}
        n_matched = 0
        for j, gene in enumerate(self.gene_symbols):
            i = incoming_idx.get(gene)
            if i is not None:
                aligned[:, j] = counts[:, i]
                n_matched += 1
        if n_matched == 0:
            raise ValueError(
                "register_context: none of the provided gene_names matched this registry's "
                "gene vocabulary — check gene symbol/ID convention."
            )
        return aligned

    def reset_online_contexts(self) -> None:
        """Drop every context added via register_context(), restoring the original registry."""
        n = self._n_registry_contexts
        self.V = self.V[:n]
        self.log_eigenvalues = self.log_eigenvalues[:n]
        self.gene_variance = self.gene_variance[:n]
        self.n_control_cells = self.n_control_cells[:n]
        self.context_names = self.context_names[:n]
        self.context_name_to_idx = {name: i for i, name in enumerate(self.context_names)}
        self.n_contexts = n
