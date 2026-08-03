#!/usr/bin/env python3
"""
Combine the slim set of per-source npz files into gene_embeddings_slim.npz.

Reads all *.npz files from SLIM_SOURCES_DIR (each has keys: embedding, mask,
gene_ids, source, dim), concatenates them column-wise, and writes
gene_embeddings_slim.npz in the same directory as this script.

No joint L2-normalisation is applied after concatenation: each source's slice
is already L2-normalised per gene within its own file, so their magnitudes are
independent. This keeps source ablation clean — masking source s does not
affect the scale of other sources' KV inputs.

Gene symbols are borrowed from the existing gene_embeddings_combined.npz
(same vocabulary, already resolved via mygene).

Output keys match GeneEmbeddingCrossAttention expectations:
  embedding        (19357, D_total) float32
  mask_any         (19357,)         bool
  mask_per_source  (19357, S)       bool
  gene_ids         (19357,)         str
  gene_symbols     (19357,)         str
  source_names     (S,)             str
  source_dims      (S,)             int32
"""

from pathlib import Path
import numpy as np

HERE             = Path(__file__).parent
SLIM_SOURCES_DIR = Path("/Users/arno.vh/repositories/QuantumCell_embeddings")
EXISTING_COMBINED = HERE / "gene_embeddings_combined.npz"  # for gene_symbols
OUT_FILE         = HERE / "gene_embeddings_slim.npz"

# Explicit order — determines source index used by --ablate-source
SOURCE_ORDER = [
    "GTEx.npz",
    "GWASAtlas.npz",
    "ESM-2.npz",
    "DepMap.npz",
    "CellPainting.npz",
    "GeneGenePT.npz",
    "consensus.npz",
    "neuronal_PPI.npz",
    "pathway_consensus.npz",
]


def main() -> None:
    reference_ids: np.ndarray | None = None
    blocks, masks, labels, dims = [], [], [], []

    for fname in SOURCE_ORDER:
        path = SLIM_SOURCES_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"Expected source file not found: {path}")

        d = np.load(path, allow_pickle=True)
        emb      = d["embedding"].astype(np.float32)
        mask     = d["mask"].astype(bool)
        gene_ids = np.array([str(g) for g in d["gene_ids"]])
        label    = str(d["source"])
        dim      = int(d["dim"])

        if emb.shape[1] != dim:
            raise ValueError(f"{fname}: embedding dim {emb.shape[1]} ≠ stored dim {dim}")

        if reference_ids is None:
            reference_ids = gene_ids
        elif not np.array_equal(gene_ids, reference_ids):
            raise ValueError(f"{fname}: gene_ids do not match reference vocabulary")

        coverage = int((~mask).sum())
        print(f"  {label:20s}  dim={dim:4d}  coverage={coverage:,}/{len(mask):,}")

        blocks.append(emb)
        masks.append(mask)
        labels.append(label)
        dims.append(dim)

    # Concatenate — no joint renorm so each source's slice stays at its own scale,
    # keeping ablation clean (masking source s has no effect on other sources' magnitudes).
    combined = np.concatenate(blocks, axis=1).astype(np.float32)

    mask_per_source = np.stack(masks, axis=1)          # (G, S)
    mask_any        = mask_per_source.all(axis=1)      # (G,)

    print(f"\nCombined: {len(labels)} sources, dim {combined.shape[1]}")
    print(f"  {' + '.join(str(d) for d in dims)} = {sum(dims)}")
    print(f"  genes present in ≥1 source: {(~mask_any).sum():,} / {len(reference_ids):,}")

    # Borrow gene_symbols from existing combined (same vocabulary)
    if EXISTING_COMBINED.exists():
        base = np.load(EXISTING_COMBINED, allow_pickle=True)
        gene_symbols = base["gene_symbols"]
        print(f"  gene_symbols copied from {EXISTING_COMBINED.name}")
    else:
        print("  WARNING: gene_embeddings_combined.npz not found — storing ENSG IDs as symbols")
        gene_symbols = reference_ids

    np.savez_compressed(
        OUT_FILE,
        embedding       = combined,
        mask_any        = mask_any,
        mask_per_source = mask_per_source,
        gene_ids        = reference_ids,
        gene_symbols    = gene_symbols,
        source_names    = np.array(labels),
        source_dims     = np.array(dims, dtype=np.int32),
    )
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
