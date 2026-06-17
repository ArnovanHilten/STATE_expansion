#!/usr/bin/env python3
"""
Extract per-source npz files from gene_embeddings_combined.npz.

Each output file has the same format as the combined file but with a single
source, so GeneEmbeddingCrossAttention can load it directly.

Usage (on Gefion):
    python3 make_per_source_npz.py \
        --combined /dcai/users/hilarn/55_cu_0055/data/embeddings/STATE_embedddings/gene_embeddings_combined.npz \
        --out_dir  /dcai/users/hilarn/55_cu_0055/data/embeddings/STATE_embedddings/per_source
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combined", required=True, help="Path to gene_embeddings_combined.npz")
    parser.add_argument("--out_dir", required=True, help="Output directory for per-source npz files")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.combined, allow_pickle=True)
    embedding = data["embedding"]          # (N, D_total)
    source_dims = data["source_dims"]      # (S,)
    source_names = data["source_names"]    # (S,)
    mask_per_source = data["mask_per_source"]  # (N, S)
    gene_ids = data["gene_ids"]
    gene_symbols = data["gene_symbols"] if "gene_symbols" in data else gene_ids

    offsets = np.concatenate([[0], np.cumsum(source_dims)])

    for i, (name, dim) in enumerate(zip(source_names, source_dims)):
        start, end = int(offsets[i]), int(offsets[i + 1])
        src_emb = embedding[:, start:end].copy()   # (N, dim_i)
        src_mask = mask_per_source[:, i]            # (N,) True = absent

        # Re-L2-normalise rows (the combined file was normalised as a whole)
        norms = np.linalg.norm(src_emb, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        src_emb = (src_emb / norms).astype(np.float32)

        out_path = out_dir / f"{name}.npz"
        np.savez_compressed(
            out_path,
            embedding=src_emb,
            mask_per_source=src_mask[:, None].astype(bool),  # (N, 1)
            mask_any=src_mask.astype(bool),
            gene_ids=gene_ids,
            gene_symbols=gene_symbols,
            source_names=np.array([name]),
            source_dims=np.array([dim], dtype=np.int32),
        )
        coverage = int((~src_mask).sum())
        print(f"  {name:<20s}  dim={dim:4d}  coverage={coverage:,}/{len(src_mask):,}  → {out_path}")

    print(f"\nDone. {len(source_names)} files written to {out_dir}")


if __name__ == "__main__":
    main()
