#!/usr/bin/env python3
"""
Build a covariance_registry.npz for CovariancePriorCrossAttention.

For each biological context c (a cell type within a dataset), computes a low-rank,
hierarchically-shrunk decomposition of the *control-cell* covariance matrix Sigma_c from raw
UMI counts:

    Sigma_hat_c ~= V_c,k @ diag(Lambda_c,k) @ V_c,k^T

where Sigma_hat_c = w_own * Sigma_own,c + w_parent * Sigma_parent,c + w_global * Sigma_global,c
+ delta * I ("Shrinkage estimator" in the CIPHER-inspired covariance prior spec). Three disjoint
pools of control cells feed the estimate:

    own:    this context's own control cells
    parent: control cells from *other* cell types in the *same* dataset (sibling contexts)
    global: control cells from *other* datasets entirely

so no cell is ever counted twice across pools. Weights follow "more controls -> larger own
weight, fewer controls -> lean on parent/global": w_own = n_own / (n_own + kappa), with the
remaining mass split across whichever pools are actually available (a context with no dataset
siblings and no other datasets just falls back to its own low-rank estimate + a ridge floor).

The shrinkage math (weighting, stacking, SVD) lives in state.tx.models.covariance_math and is
shared with CovariancePriorCrossAttention.register_context(), which uses the identical estimator
to shrink a brand-new context discovered at inference time — see that method's docstring for the
"pseudo-cell" trick that lets it reuse this script's output as shrinkage targets without needing
raw cell data kept around.

Only non-targeting / control cells are used at every pool level, so no perturbation-response
information leaks into the prior (critical for zero-shot evaluation).

Context is keyed as "{dataset_name}.{cell_type}", where dataset_name comes from the TOML's
[datasets] section and cell_type is read from obs[cell_type_col] if present, else inferred
from the h5ad filename stem (this matches "gold" datasets like tian2019, laid out as one
h5ad file per cell type — e.g. tian2019/ipsc.h5ad, tian2019/neuron.h5ad).

An extra "__global__" pseudo-context is appended to the registry: a pool of ALL processed
contexts' (already shrunk) covariances, combined via the same pseudo-cell trick. It isn't a real
biological context (never resolved by name at training time) — it exists purely as a shrinkage
target for CovariancePriorCrossAttention.register_context() to lean on when a brand-new context
shows up at inference with too few control cells of its own.

Output keys (consumed by CovariancePriorCrossAttention):
    V               (C, G, k) float16  — eigengene loadings per context (post-shrinkage)
    eigenvalues     (C, k)    float32  — covariance eigenvalues per context (post-shrinkage)
    gene_variance   (C, G)    float32  — exact per-gene variance (post-shrinkage)
    context_names   (C,)      str      — "{dataset_name}.{cell_type}", plus "__global__"
    gene_symbols    (G,)      str      — shared gene vocabulary (var_names)
    n_control_cells (C,)      int64    — own-context control-cell count (total pooled, for __global__)
    metadata_json   scalar    str      — per-context shrinkage weights & provenance (JSON)

Note on scale: this script loads every context's full control-cell count matrix into memory at
once (needed to build the parent/global pools). Fine for small-to-medium corpora; a very large
multi-dataset run would want streaming sufficient statistics instead.

Usage:
    python covariance_prior/build_covariance_registry.py \\
        --toml /Volumes/512ssd/perturbseq_data/gold/tian2019/tian2019.toml \\
        --out covariance_prior/tian2019_covariance_registry.npz \\
        --rank 64
"""

from __future__ import annotations

import argparse
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp

from state.tx.models.covariance_math import (
    ShrinkageWeights,
    compute_shrinkage_weights,
    stack_and_decompose,
    weighted_block_from_counts,
    weighted_block_from_low_rank,
)

GLOBAL_CONTEXT_NAME = "__global__"


@dataclass
class ContextData:
    context_id: str
    dataset_name: str
    cell_type: str
    counts: np.ndarray  # (N, G) float32 raw control-cell counts
    source_h5ad: str


def load_control_counts(h5ad_path: Path, pert_col: str, control_pert: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a dense (N_control, G) float32 raw-count matrix for control cells."""
    adata = ad.read_h5ad(h5ad_path)
    if pert_col not in adata.obs.columns:
        raise ValueError(f"{h5ad_path}: obs has no column '{pert_col}' (columns: {list(adata.obs.columns)})")

    is_control = (adata.obs[pert_col] == control_pert).to_numpy()
    n_control = int(is_control.sum())
    if n_control == 0:
        raise ValueError(f"{h5ad_path}: no cells with {pert_col}=='{control_pert}'")

    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    counts = counts[is_control]
    if sp.issparse(counts):
        counts = counts.toarray()
    counts = np.asarray(counts, dtype=np.float32)
    return counts, adata.var_names.to_numpy(), n_control


def infer_cell_type(h5ad_path: Path, adata_cell_type_col: str | None) -> str:
    if adata_cell_type_col:
        return adata_cell_type_col
    return h5ad_path.stem  # e.g. "ipsc", "neuron"


def shrunk_low_rank_covariance(
    own: np.ndarray, parent: np.ndarray | None, global_: np.ndarray | None, weights: ShrinkageWeights, rank: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the weighted own/parent/global stack from raw control-cell counts and decompose it."""
    g = own.shape[1]
    blocks = []
    variance_terms = []

    own_block, own_var = weighted_block_from_counts(own, weights.own)
    blocks.append(own_block)
    variance_terms.append((weights.own, own_var))

    if parent is not None:
        parent_block, parent_var = weighted_block_from_counts(parent, weights.parent)
        blocks.append(parent_block)
        variance_terms.append((weights.parent, parent_var))

    if global_ is not None:
        global_block, global_var = weighted_block_from_counts(global_, weights.global_)
        blocks.append(global_block)
        variance_terms.append((weights.global_, global_var))

    return stack_and_decompose(blocks, variance_terms, rank, weights.delta, n_genes=g)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--toml", type=Path, required=True, help="cell_load-style TOML with a [datasets] section")
    parser.add_argument("--out", type=Path, required=True, help="output covariance_registry.npz path")
    parser.add_argument("--rank", type=int, default=64, help="low-rank dimension k (default: 64)")
    parser.add_argument("--pert-col", default="gene", help="obs column naming the perturbed gene (default: gene)")
    parser.add_argument("--control-pert", default="non-targeting", help="control-cell label (default: non-targeting)")
    parser.add_argument(
        "--cell-type-col",
        default=None,
        help="obs column to use as cell type, if present. Falls back to the h5ad filename stem per file.",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=1000.0,
        help="shrinkage half-scale: w_own = n_own / (n_own + kappa) (default: 1000, per spec's 500-2000 range)",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=0.05,
        help="identity-ridge floor added to the (weights sum to 1-delta) shrinkage mixture (default: 0.05)",
    )
    parser.add_argument(
        "--no-shrinkage",
        action="store_true",
        help="disable hierarchical shrinkage entirely (w_own=1, w_parent=w_global=delta=0) — pure per-context estimate",
    )
    parser.add_argument(
        "--no-global-context",
        action="store_true",
        help="skip writing the '__global__' pooled pseudo-context (used for online shrinkage at inference)",
    )
    args = parser.parse_args()

    with open(args.toml, "rb") as f:
        config = tomllib.load(f)
    datasets: dict[str, str] = config.get("datasets", {})
    if not datasets:
        raise ValueError(f"{args.toml}: no [datasets] section found")

    reference_genes: np.ndarray | None = None
    contexts: list[ContextData] = []

    for dataset_name, dataset_dir in datasets.items():
        h5ad_paths = sorted(Path(dataset_dir).glob("*.h5ad"))
        if not h5ad_paths:
            raise FileNotFoundError(f"No .h5ad files found in {dataset_dir}")

        for h5ad_path in h5ad_paths:
            print(f"Loading {dataset_name}: {h5ad_path.name}")
            counts, genes, n_control = load_control_counts(h5ad_path, args.pert_col, args.control_pert)

            if reference_genes is None:
                reference_genes = genes
            elif not np.array_equal(genes, reference_genes):
                raise ValueError(
                    f"{h5ad_path}: gene vocabulary does not match the reference "
                    f"({len(genes)} vs {len(reference_genes)} genes). All h5ad files for one "
                    f"registry must share the same var_names."
                )

            cell_type_col_val = None
            if args.cell_type_col:
                adata = ad.read_h5ad(h5ad_path, backed="r")
                if args.cell_type_col in adata.obs.columns:
                    unique_vals = adata.obs[args.cell_type_col].unique()
                    if len(unique_vals) == 1:
                        cell_type_col_val = str(unique_vals[0])
            cell_type = infer_cell_type(h5ad_path, cell_type_col_val)
            context_id = f"{dataset_name}.{cell_type}"
            print(f"    context={context_id}  n_control={n_control}  n_genes={len(genes)}")

            contexts.append(
                ContextData(
                    context_id=context_id,
                    dataset_name=dataset_name,
                    cell_type=cell_type,
                    counts=counts,
                    source_h5ad=str(h5ad_path),
                )
            )

    context_names: list[str] = []
    n_control_cells: list[int] = []
    V_list: list[np.ndarray] = []
    eigenvalues_list: list[np.ndarray] = []
    gene_variance_list: list[np.ndarray] = []
    metadata: dict[str, dict] = {}

    print("\nComputing per-context shrinkage weights and low-rank estimates...")
    for ctx in contexts:
        siblings = [c for c in contexts if c.dataset_name == ctx.dataset_name and c is not ctx]
        others = [c for c in contexts if c.dataset_name != ctx.dataset_name]

        parent_pool = np.concatenate([c.counts for c in siblings], axis=0) if siblings else None
        global_pool = np.concatenate([c.counts for c in others], axis=0) if others else None
        n_parent = parent_pool.shape[0] if parent_pool is not None else 0
        n_global = global_pool.shape[0] if global_pool is not None else 0
        n_own = ctx.counts.shape[0]

        if args.no_shrinkage:
            weights = ShrinkageWeights(own=1.0, parent=0.0, global_=0.0, delta=0.0)
        else:
            weights = compute_shrinkage_weights(n_own, n_parent, n_global, args.kappa, args.delta)

        print(
            f"  {ctx.context_id}: n_own={n_own} n_parent={n_parent} n_global={n_global}  "
            f"weights(own/parent/global/delta)={weights.own:.3f}/{weights.parent:.3f}/"
            f"{weights.global_:.3f}/{weights.delta:.3f}"
        )

        V, eigenvalues, gene_variance = shrunk_low_rank_covariance(
            ctx.counts, parent_pool, global_pool, weights, args.rank
        )

        context_names.append(ctx.context_id)
        n_control_cells.append(n_own)
        V_list.append(V.astype(np.float16))
        eigenvalues_list.append(eigenvalues)
        gene_variance_list.append(gene_variance)
        metadata[ctx.context_id] = {
            "dataset_name": ctx.dataset_name,
            "cell_type": ctx.cell_type,
            "n_control_cells": n_own,
            "n_parent_pool": n_parent,
            "n_global_pool": n_global,
            "rank_requested": args.rank,
            "pert_col": args.pert_col,
            "control_pert": args.control_pert,
            "covariance_input": "raw_counts",
            "estimator": "truncated_svd_lowrank" if args.no_shrinkage else "shrinkage_lowrank",
            "shrinkage_weights": weights.as_dict(),
            "kappa": args.kappa,
            "source_h5ad": ctx.source_h5ad,
        }

    if not args.no_global_context and len(context_names) > 0:
        print(f"\nBuilding {GLOBAL_CONTEXT_NAME} pseudo-context (pooled across all processed contexts)...")
        # Reuse each context's already-shrunk (V, eigenvalues) as exact pseudo-cells (see
        # weighted_block_from_low_rank), weighted by its own control-cell count — no raw data
        # needs to be kept around for this, and it composes correctly with further shrinkage
        # later (CovariancePriorCrossAttention.register_context shrinking a *new* context toward
        # this __global__ entry is just another level of the same exact trick).
        total_n = sum(n_control_cells)
        g = len(reference_genes)
        blocks = []
        variance_terms = []
        for V, eigenvalues, gene_variance, n_c in zip(V_list, eigenvalues_list, gene_variance_list, n_control_cells):
            weight = n_c / total_n
            block = weighted_block_from_low_rank(V.astype(np.float32), eigenvalues, weight)
            blocks.append(block)
            variance_terms.append((weight, gene_variance))
        rank = eigenvalues_list[0].shape[0]
        V_global, eigenvalues_global, gene_variance_global = stack_and_decompose(
            blocks, variance_terms, rank, delta=0.0, n_genes=g, label=GLOBAL_CONTEXT_NAME
        )
        context_names.append(GLOBAL_CONTEXT_NAME)
        n_control_cells.append(total_n)
        V_list.append(V_global.astype(np.float16))
        eigenvalues_list.append(eigenvalues_global)
        gene_variance_list.append(gene_variance_global)
        metadata[GLOBAL_CONTEXT_NAME] = {
            "dataset_name": None,
            "cell_type": None,
            "n_control_cells": total_n,
            "rank_requested": rank,
            "estimator": "pooled_pseudo_cell",
            "pooled_contexts": [c for c in context_names if c != GLOBAL_CONTEXT_NAME],
        }
        print(f"    {GLOBAL_CONTEXT_NAME}: n_control(pooled)={total_n}")

    V_stack = np.stack(V_list, axis=0)                    # (C, G, k) float16
    eigenvalues_stack = np.stack(eigenvalues_list, axis=0)  # (C, k) float32
    gene_variance_stack = np.stack(gene_variance_list, axis=0)  # (C, G) float32

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        V=V_stack,
        eigenvalues=eigenvalues_stack,
        gene_variance=gene_variance_stack,
        context_names=np.array(context_names),
        gene_symbols=reference_genes,
        n_control_cells=np.array(n_control_cells, dtype=np.int64),
        metadata_json=json.dumps(metadata),
    )
    print(f"\nSaved covariance registry -> {args.out}")
    print(f"  contexts: {context_names}")
    print(f"  V shape: {V_stack.shape}, gene_variance shape: {gene_variance_stack.shape}")


if __name__ == "__main__":
    main()
