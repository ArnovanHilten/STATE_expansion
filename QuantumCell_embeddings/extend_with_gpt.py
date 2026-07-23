#!/usr/bin/env python3
"""
Extend gene_embeddings_combined.npz with GenePT and GeneGenePT sources.

Reads:
  gene_embeddings_combined.npz   (22 sources, dim 2191)
  GenePT_embeddings_v2.npz       (256-dim, mask=True for absent genes)
  GeneGenePT.npz                 (256-dim, mask=True for absent genes)

Writes:
  gene_embeddings_combined_with_gpt.npz  (24 sources, dim 2703)

The model (GeneEmbeddingCrossAttention) reads source_dims dynamically,
so no code changes are needed — just point qc_emb_path at the new file.
"""

from pathlib import Path
import numpy as np

HERE = Path(__file__).parent

COMBINED_IN  = HERE / "gene_embeddings_combined.npz"
NEW_SOURCES  = [
    HERE / "GenePT_embeddings_v2.npz",
    HERE / "GeneGenePT.npz",
]
COMBINED_OUT = HERE / "gene_embeddings_combined_with_gpt.npz"


def main() -> None:
    # ── Load existing combined ────────────────────────────────────────────────
    print(f"Loading {COMBINED_IN.name} …")
    base = np.load(COMBINED_IN, allow_pickle=True)

    embedding       = base["embedding"].astype(np.float32)   # (G, D)
    mask_per_source = base["mask_per_source"].astype(bool)   # (G, S)
    mask_any        = base["mask_any"].astype(bool)          # (G,)
    gene_ids        = np.array([str(g) for g in base["gene_ids"]])
    gene_symbols    = base["gene_symbols"]
    source_names    = list(base["source_names"])
    source_dims     = list(base["source_dims"].astype(int))

    G = len(gene_ids)
    print(f"  {len(source_names)} sources, dim {embedding.shape[1]}, {G} genes")

    # ── Append each new source ────────────────────────────────────────────────
    new_blocks = []
    new_masks  = []
    new_labels = []
    new_dims   = []

    for path in NEW_SOURCES:
        print(f"\nLoading {path.name} …")
        d = np.load(path, allow_pickle=True)

        src_emb   = d["embedding"].astype(np.float32)  # (G, dim)
        src_mask  = d["mask"].astype(bool)             # (G,) True=absent
        src_ids   = np.array([str(g) for g in d["gene_ids"]])
        src_label = str(d["source"])
        src_dim   = int(d["dim"])

        # Validate alignment
        if len(src_ids) != G:
            raise ValueError(
                f"{path.name}: {len(src_ids)} genes, expected {G}"
            )
        if not np.array_equal(src_ids, gene_ids):
            raise ValueError(
                f"{path.name}: gene_ids do not match the combined vocabulary. "
                "Ensure both were built from the same STATE gene list."
            )
        if src_emb.shape[1] != src_dim:
            raise ValueError(
                f"{path.name}: embedding dim {src_emb.shape[1]} ≠ stored dim {src_dim}"
            )

        coverage = int((~src_mask).sum())
        print(f"  label={src_label}, dim={src_dim}, coverage={coverage:,}/{G:,}")

        if src_label in source_names:
            raise ValueError(
                f"Source '{src_label}' already exists in the combined file. "
                "Rename or deduplicate before extending."
            )

        new_blocks.append(src_emb)
        new_masks.append(src_mask)
        new_labels.append(src_label)
        new_dims.append(src_dim)

    # ── Concatenate ───────────────────────────────────────────────────────────
    extended_emb  = np.concatenate([embedding] + new_blocks, axis=1)  # (G, D+)
    extended_mask = np.concatenate(
        [mask_per_source] + [m[:, None] for m in new_masks], axis=1
    )  # (G, S+)
    extended_any  = extended_mask.all(axis=1)

    # Re-L2-normalise rows (same as combine.py)
    norms = np.linalg.norm(extended_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    extended_emb = (extended_emb / norms).astype(np.float32)

    all_names = np.array(source_names + new_labels)
    all_dims  = np.array(source_dims  + new_dims, dtype=np.int32)

    print(f"\nExtended: {len(all_names)} sources, dim {extended_emb.shape[1]}")
    print(f"  dims: {' + '.join(str(d) for d in all_dims)} = {all_dims.sum()}")
    print(f"  genes present in ≥1 source: {(~extended_any).sum():,} / {G:,}")

    # ── Save ──────────────────────────────────────────────────────────────────
    np.savez_compressed(
        COMBINED_OUT,
        embedding       = extended_emb,
        mask_any        = extended_any,
        mask_per_source = extended_mask,
        gene_ids        = gene_ids,
        gene_symbols    = gene_symbols,
        source_names    = all_names,
        source_dims     = all_dims,
    )
    print(f"\nSaved → {COMBINED_OUT}")


if __name__ == "__main__":
    main()
