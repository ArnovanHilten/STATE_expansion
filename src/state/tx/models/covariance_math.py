"""Shared low-rank / shrinkage covariance math for the covariance prior.

Used by both the offline precompute script (covariance_prior/build_covariance_registry.py,
which estimates Sigma_c from raw control-cell counts per dataset) and CovariancePriorCrossAttention
(which can register a brand-new context on the fly at inference time — see
`register_context` there). Keeping the math in one place means both paths shrink covariances
identically.

Core trick used throughout: Sigma is never formed densely. A weighted sum of covariances
    Sigma_hat = w_1*Sigma_1 + w_2*Sigma_2 + ...
can be obtained by rescaling each Sigma_i's underlying centered data by sqrt(w_i / (n_i - 1))
and stacking the results — one SVD of the stack gives Sigma_hat's eigenvectors/eigenvalues
directly (stack.T @ stack == Sigma_hat exactly). A low-rank covariance (V, Lambda) — including
an *already shrunk* one from a previous call — can itself be treated as exact "pseudo-cells" for
this same trick, since Sigma = V @ diag(Lambda) @ V.T == R.T @ R for R = sqrt(Lambda) * V.T. That
lets a brand-new context be shrunk toward already-shrunk registry contexts without ever needing
to keep raw cell data around for them (see `weighted_block_from_low_rank`).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np


@dataclass
class ShrinkageWeights:
    own: float
    parent: float
    global_: float
    delta: float

    def as_dict(self) -> dict:
        return {"own": self.own, "parent": self.parent, "global": self.global_, "delta": self.delta}


def compute_shrinkage_weights(n_own: float, n_parent: float, n_global: float, kappa: float, delta: float) -> ShrinkageWeights:
    """w_own = n_own / (n_own + kappa); remaining mass split across whichever pools exist,
    proportional to each pool's own (effective) sample count. All three weights sum to
    (1 - delta); delta is the identity-ridge floor applied afterwards.
    """
    alpha = n_own / (n_own + kappa)
    remaining = 1.0 - alpha
    pool_total = n_parent + n_global
    if pool_total > 0:
        w_parent = remaining * (n_parent / pool_total)
        w_global = remaining * (n_global / pool_total)
    else:
        w_parent = 0.0
        w_global = 0.0
        alpha = 1.0  # nothing to fall back on — all non-ridge mass stays on "own"

    scale = 1.0 - delta
    return ShrinkageWeights(own=alpha * scale, parent=w_parent * scale, global_=w_global * scale, delta=delta)


def weighted_block_from_counts(counts: np.ndarray, weight: float) -> tuple[np.ndarray | None, np.ndarray]:
    """Mean-center raw `counts` and rescale rows so block.T @ block == weight * Sigma(counts).

    Returns (scaled_block_or_None, exact_gene_variance). scaled_block is None when weight <= 0
    or there are too few cells (<2) to form a covariance.
    """
    n = counts.shape[0]
    mean = counts.mean(axis=0, keepdims=True)
    centered = counts - mean
    variance = centered.var(axis=0, ddof=1).astype(np.float32) if n > 1 else np.zeros(counts.shape[1], np.float32)
    if weight <= 0 or n < 2:
        return None, variance
    scaled = (centered * np.sqrt(weight / (n - 1))).astype(np.float32)
    return scaled, variance


def weighted_block_from_low_rank(V: np.ndarray, eigenvalues: np.ndarray, weight: float) -> np.ndarray | None:
    """Turn an existing low-rank covariance (V, eigenvalues) into a (k, G) pseudo-data block
    such that block.T @ block == weight * (V @ diag(eigenvalues) @ V.T) *exactly*.

    This lets an already-computed (possibly already-shrunk) covariance — e.g. a registry entry —
    be reused as a shrinkage target without ever needing the raw cell data it was built from.
    """
    if weight <= 0:
        return None
    safe_eigenvalues = np.clip(eigenvalues, 0.0, None)
    pseudo_rows = (V * np.sqrt(safe_eigenvalues)[None, :]).T  # (k, G); rows.T @ rows == V diag(eig) V.T
    return (pseudo_rows * np.sqrt(weight)).astype(np.float32)


def stack_and_decompose(
    blocks: list[np.ndarray],
    variance_terms: list[tuple[float, np.ndarray]],
    rank: int,
    delta: float,
    n_genes: int,
    label: str = "",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack pre-weighted blocks (from either weighted_block_from_counts or
    weighted_block_from_low_rank), SVD once, and return the shrunk low-rank estimate.

    Args:
        blocks: list of (n_i, G) arrays, already rescaled so sum(block.T @ block) == the
            desired weighted covariance sum (excluding the delta*I ridge).
        variance_terms: list of (weight, exact_variance) pairs — summed directly (exact, no
            low-rank truncation) to get the final per-gene variance, then the ridge floor added.
        rank: target low-rank dimension k.
        delta: identity-ridge floor added post-hoc (shifts every eigenvalue by a constant, so
            it's applied after the SVD rather than as another stacked block).
        n_genes: G, needed to build the zero-padded output when fewer than `rank` blocks/rows
            are available.
        label: optional string used only in the padding warning message.

    Returns:
        V:             (G, rank) float32 — zero-padded past the number of nontrivial components
        eigenvalues:   (rank,)   float32 — includes the `delta` ridge floor
        gene_variance: (G,)      float32 — exact weighted variance, includes the ridge floor
    """
    gene_variance = np.zeros(n_genes, dtype=np.float32)
    for weight, variance in variance_terms:
        gene_variance += weight * variance
    gene_variance += delta

    usable_blocks = [b for b in blocks if b is not None]
    if not usable_blocks:
        raise ValueError("No usable blocks (all weight<=0 or too few rows) — cannot estimate covariance.")

    stacked = np.concatenate(usable_blocks, axis=0)
    k_eff = min(rank, stacked.shape[0] - 1 if stacked.shape[0] > 1 else stacked.shape[0], n_genes)
    k_eff = max(k_eff, 0)
    if k_eff < 1:
        raise ValueError(f"Need at least 1 usable row across blocks for a rank-1 estimate, got {stacked.shape[0]}")

    _u, s, vt = np.linalg.svd(stacked, full_matrices=False)
    eigenvalues_eff = (s[:k_eff] ** 2) + delta
    v_eff = vt[:k_eff, :].T  # (G, k_eff)

    V = np.zeros((n_genes, rank), dtype=np.float32)
    eigenvalues = np.full((rank,), delta + 1e-8, dtype=np.float32)
    V[:, :k_eff] = v_eff
    eigenvalues[:k_eff] = eigenvalues_eff

    if k_eff < rank:
        print(
            f"    WARNING{f' ({label})' if label else ''}: only {stacked.shape[0]} usable rows across blocks; "
            f"requested rank={rank} but only {k_eff} nontrivial components exist. "
            f"Remaining {rank - k_eff} components zero-padded.",
            file=sys.stderr,
        )
    return V, eigenvalues, gene_variance
