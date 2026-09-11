# AI-Assisted Work Summary: Covariance Prior

This document summarizes work done by Claude Code (Anthropic) on branch `feature/qc-cross-attention`
implementing a CIPHER-inspired, context-conditioned covariance prior for the `state` perturbation
model. It's meant as a standalone reference for anyone reviewing this work — what was built, why,
how it was validated, and what's intentionally left out.

For hands-on usage instructions (CLI commands, registry format, code snippets), see
[`covariance_prior/README.md`](covariance_prior/README.md). This document is the higher-level
narrative: motivation, design decisions, what was checked, and what's still open.

## Origin

The feature follows a design spec ("cipher_inspired_covariance.pdf") describing how to add a
cell-type- and state-specific covariance prior to a perturbation-prediction model, inspired by
the CIPHER framework's linear-response approximation `ΔX ≈ Σu`. The spec's core principle:

> The covariance prior is a property of the gene *within a cellular context*, not a universal
> property of the gene.

i.e. unlike the model's existing static priors (STRING, ESM-2, GenePT, DepMap, Cell Painting —
one fixed embedding per gene, loaded via `GeneEmbeddingCrossAttention`), this prior is computed
fresh from each target cell population's own unperturbed control cells, so the same perturbed
gene produces a different signal depending on cell type/state.

## What was built

### 1. Core mechanism — context-conditioned cross-attention token

- **`CovariancePriorCrossAttention`** (`src/state/tx/models/cross_attention.py`): loads a
  precomputed registry of per-context low-rank covariance decompositions and looks up
  `z_cov(context, gene)` for each cell-set in a batch, returning one cross-attention KV token.
  Falls back to a learned `[NO_COV]` token for unknown context/gene pairs (substitution, not
  masking).
- **Feature representation**: `f_cov(c,j) = [V_c,k[j,:], log(Λ_c,k), Var_c(j)]` — the spec's
  "eigengene loading vector" option, the cheapest of the representations proposed (avoids
  per-gene `O(G)` column reconstruction).
- **Model integration** (`src/state/tx/models/state_transition.py`): the token is concatenated
  onto the existing QuantumCell static-prior tokens (if enabled) and fed through the *same*
  cross-attention layers already interleaved with the transformer backbone — no new attention
  mechanism was introduced, just another KV source. Context is resolved per cell-set from
  `batch["cell_type"]` + `batch["dataset_name"]`, which are already present in every `cell_load`
  batch (no data-schema changes were needed).
- **Config**: `use_cov_cross_attn` / `cov_registry_path` / `cov_dropout` kwargs added to
  `src/state/configs/model/state.yaml`, usable independently of or alongside the existing
  `use_qc_cross_attn`.

### 2. Offline precompute pipeline

- **`covariance_prior/build_covariance_registry.py`**: reads a `cell_load`-style TOML, takes
  each context's control/non-targeting cells, computes a low-rank covariance decomposition via
  economy SVD (the dense `G×G` covariance matrix is never formed — sizes here reach 33,752×33,752),
  and writes a `covariance_registry.npz`.
- **Hierarchical shrinkage estimator**: raw per-context covariance is unreliable when
  `N_control << G` (a classic high-dimensional estimation problem — hundreds of control cells vs.
  tens of thousands of genes means the sample covariance is mostly noise). Each context's
  estimate is shrunk toward two disjoint pools — sibling cell types in the same dataset
  ("parent"), and other datasets entirely ("global") — weighted by
  `w_own = n_own / (n_own + κ)`, with remaining mass split by pool size. No cell is ever counted
  in more than one pool, and the whole weighted sum is computed via a *single* SVD of stacked,
  pre-scaled data blocks — never a dense matrix.
- **`src/state/tx/models/covariance_math.py`**: the shrinkage math (weighting, block
  construction, stack+SVD) was extracted into a shared module so the offline script and the
  runtime registration path below use identical logic.

### 3. On-the-fly context registration (inference-time, no offline rebuild)

The spec explicitly calls out live covariance computation for brand-new contexts as "the most
powerful use case" (test-time adaptation on a target cell population without needing
perturbation labels). Implemented as:

- **`CovariancePriorCrossAttention.register_context(...)`**: given raw control cells for a new
  context, computes the same shrinkage estimate live and grows the module's buffers so the
  context is immediately usable — no registry file rebuild, no model reload.
- **The key trick**: a low-rank covariance `(V, Λ)` — including one *already shrunk* and sitting
  in the registry — can be treated as exact "pseudo-cells" for further shrinkage, since
  `Σ = VΛVᵀ = (√Λ·Vᵀ)ᵀ(√Λ·Vᵀ)`. This means a new context can be shrunk toward an existing
  sibling context or a pooled global reference without ever needing that reference's original
  raw cell data — just its already-computed factors.
- **`build_covariance_registry.py`** now also writes a `"__global__"` pooled pseudo-context into
  every registry (all processed contexts combined via the same pseudo-cell trick), purely as a
  ready-made shrinkage target for online registration — it's never resolved as a real training
  context.
- Convenience wrappers on the model itself: `model.register_covariance_context(...)` /
  `model.reset_covariance_contexts()`.
- Gene-column realignment: `register_context` accepts a `gene_names` list and reindexes a new
  context's columns to the registry's gene order, so a differently-ordered or partial var_names
  from a new dataset still works.

### 4. Training-time robustness

- **Covariance dropout**: `CovariancePriorCrossAttention` randomly corrupts `(context, gene)`
  pairs during training (`none` → `[NO_COV]` fallback, `wrong_context` → random other context,
  `shuffled_gene` → random other gene in the same context) so the model doesn't over-rely on the
  prior and its benefit can be measured.

## Validation

All validation ran against a small real perturb-seq dataset (`tian2019`: 115 iPSC + 480 neuron
control cells, 33,752 genes, 2 perturbed genes) used purely as an integration/smoke test — far
too small to draw any scientific conclusion from, but sufficient to prove every part of the
pipeline is wired correctly before pointing it at a real training corpus:

- **Design principle confirmed empirically**: `z_cov(ipsc, PPP2R1A)` vs. `z_cov(neuron, PPP2R1A)`
  have cosine similarity ≈ 0.17 post-shrinkage — meaningfully different despite heavy shrinkage
  (`w_own ≈ 0.10` for `ipsc`, borrowing ≈ 0.85 from the `neuron` sibling pool).
- **Real data pipeline integration**: instantiated the actual `cell_load.PerturbationDataModule`
  against the dataset's TOML and confirmed batch fields (`cell_type`, `dataset_name`) resolve to
  exactly the registry's context keys with zero glue code.
- **Real forward passes and training steps**: ran real optimizer steps (Adam, energy loss,
  covariance dropout active) through `StateTransitionPerturbationModel` using real dataloader
  batches — gradients flow into `cov_module` every step, no NaNs, output differs by context for
  the same perturbed gene.
- **Shrinkage math correctness**: reconstructed the full dense covariance from the low-rank
  output and compared it to the manually-computed weighted sum of dense covariances — matches to
  `1e-3` at full rank (`tests/test_covariance_shrinkage.py`).
- **On-the-fly registration, end-to-end on real data**: carved out a held-out slice of
  `tian2019` neuron cells, registered it live as a pretend brand-new context
  (`"brandnew.neuronlike"`, not in the offline registry), shrunk toward `tian2019.ipsc` as parent
  plus the `__global__` pool, and confirmed a real forward pass referencing it produces output
  distinct from the `[NO_COV]` fallback.

## Testing

69 tests across 4 files, all passing:

| File | Coverage |
|---|---|
| `tests/test_covariance_cross_attn.py` | Unit tests for `CovariancePriorCrossAttention` + model integration (synthetic registry) |
| `tests/test_covariance_shrinkage.py` | Shrinkage-math correctness (dense-covariance reconstruction) |
| `tests/test_covariance_online_registration.py` | On-the-fly registration: math, gene realignment, parent/global blending, reset, model integration |
| `tests/test_qc_cross_attn.py` | Pre-existing QuantumCell cross-attention tests (unaffected; one pre-existing unrelated stale-mock failure found and fixed along the way) |

## Known limitations / not implemented

Deliberately out of scope for this pass — none block using the feature as implemented, but worth
being explicit about what the spec describes that isn't here yet:

- **Only the "eigengene loading vector" representation.** The spec's response-template variant,
  the summary-statistics hybrid (positive/negative covariance mass, entropy, sparse
  top-neighbor tokens), and the sparse covariance-neighbor-token alternative aren't implemented.
- **No normalization ablation.** Locked to raw-UMI-count covariance; the spec's proposed sweep
  (log1p CP10K, Pearson correlation, residualized-raw, low-rank-log) hasn't been run.
- **Covariance dropout covers 3 of the spec's 5 categories** (none / wrong-context /
  shuffled-gene — no separate "global-covariance" substitution, partially subsumed by shrinkage
  already blending global signal into the registry itself).
- **No `context_match_level` / confidence metadata** fed to the model alongside the token.
- **No ablation test harness or eval metrics** (same-context vs. wrong-context vs. shuffled vs.
  diagonal-only vs. rank-`k` sweep, Pearson Δ / DE direction match / etc.) — the spec's own
  validation plan for proving the gain comes from real context-specific structure, not just
  extra parameters.
- **No CIPHER-alignment auxiliary loss or InfoNCE context-mismatch objective** — both explicitly
  deferred by the spec itself until the base cross-attention-token version proves useful.
- **No automatic "nearest context" selection.** `register_context`'s `parent_context` must be
  named explicitly; the spec's `Σ_nearest_context` implies automatically finding the closest
  existing context (e.g. by mean-expression similarity).

## Files touched

```
src/state/tx/models/cross_attention.py       CovariancePriorCrossAttention (+ register_context, reset_online_contexts)
src/state/tx/models/covariance_math.py       new — shared shrinkage math
src/state/tx/models/state_transition.py      wiring, context resolution, register_covariance_context wrapper
src/state/configs/model/state.yaml           use_cov_cross_attn / cov_registry_path / cov_dropout kwargs
covariance_prior/build_covariance_registry.py  new — offline precompute script
covariance_prior/README.md                   new — usage documentation
covariance_prior/tian2019_covariance_registry.npz  new — real registry built for validation (test-only dataset)
tests/test_covariance_cross_attn.py          new
tests/test_covariance_shrinkage.py           new
tests/test_covariance_online_registration.py new
tests/test_qc_cross_attn.py                  one-line fix to a stale test mock (unrelated pre-existing bug)
```

---

🤖 This summary and the work it describes were generated with [Claude Code](https://claude.com/claude-code).
