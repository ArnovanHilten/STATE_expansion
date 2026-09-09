# Covariance Prior

A CIPHER-inspired, context-conditioned covariance prior for the `state` perturbation model.

Unlike the existing static per-gene priors (QuantumCell cross-attention: STRING, ESM-2, GenePT,
DepMap, Cell Painting, ...), this prior is **not** a fixed embedding of a gene. It's computed
from the target cell population's own unperturbed control cells, so the same gene gets a
different token depending on cellular context:

```
z_cov(HepG2, TP53)         !=  z_cov(Jurkat_active, TP53)
z_cov(resting_T_cell, CD3E) !=  z_cov(activated_T_cell, CD3E)
```

The idea: in the CIPHER linear-response model, a perturbation's effect is approximated as
`ΔX ≈ Σu`. For a single-gene perturbation of gene `j`, the `j`-th column of the *unperturbed*
population's covariance matrix is a first-order template for how the transcriptome moves when
`j` is perturbed, in that specific cellular context.

## Status

Core mechanism implemented and validated end-to-end on real data (a small real perturb-seq
dataset was used purely as an integration smoke test — see [Validation](#validation)). See
[Known limitations](#known-limitations) for what's intentionally deferred or still open.

## How it works

```
Σ_c = Cov(X_control, c)              # control/non-targeting cells only, never perturbed cells
z_cov(c, j) = CovEncoder(Σ_c[:, j])  # one cross-attention token per (context, gene)
```

1. **Offline precompute** ([`build_covariance_registry.py`](build_covariance_registry.py)):
   for each context (a cell type within a dataset), take its control cells, compute a
   hierarchically-shrunk low-rank decomposition of the covariance matrix, and write a
   `covariance_registry.npz`.
2. **Runtime lookup** (`CovariancePriorCrossAttention` in
   [`../src/state/tx/models/cross_attention.py`](../src/state/tx/models/cross_attention.py)):
   given `(context, gene)` for each cell-set in a batch, look up that context's eigengene
   loadings for that gene, encode them with a small MLP, and hand back one KV token.
3. **Model integration** ([`../src/state/tx/models/state_transition.py`](../src/state/tx/models/state_transition.py)):
   the token is concatenated onto the existing QuantumCell static-prior tokens (if enabled) and
   fed through the same cross-attention layers already interleaved with the transformer backbone
   — no new attention mechanism, just another KV source.

Context is resolved per cell-set from `batch["cell_type"]` + `batch["dataset_name"]` (already
present in every `cell_load` batch, no data-schema changes needed), combined as
`"{dataset_name}.{cell_type}"` to match the registry's keys.

### Feature representation

For gene `j` in context `c`, with `V_c,k` / `Λ_c,k` the top-`k` eigenvectors/eigenvalues of the
(shrunk) covariance matrix:

```
f_cov(c, j) = [ V_c,k[j, :],  log(Λ_c,k),  Var_c(j) ]     # 2k+1 dims
z_cov(c, j) = LayerNorm(MLP(f_cov(c, j)))
```

This is the spec's "Option 1: eigengene loading vector" — the simplest, cheapest representation
(no per-gene `O(G)` column reconstruction). The `[NO_COV]` fallback (a learned token) is
substituted whenever the context or gene isn't in the registry, so unknown contexts degrade
gracefully instead of crashing or masking.

### Shrinkage

Raw per-context covariance is unreliable when `N_control << G` (a classic high-dimensional
covariance estimation problem — with hundreds of control cells and tens of thousands of genes,
the sample covariance is mostly noise). Each context's estimate is shrunk toward two disjoint,
broader pools:

```
Σ_hat_c = w_own * Σ_own,c + w_parent * Σ_parent + w_global * Σ_global + δ * I

own:    this context's own control cells
parent: control cells from OTHER cell types in the SAME dataset (sibling contexts)
global: control cells from OTHER datasets entirely

w_own = n_own / (n_own + κ);  remaining mass split across parent/global by pool size
```

No cell is ever counted in more than one pool. The whole weighted sum is computed via a single
SVD of the stacked, pre-scaled control-cell blocks — the dense `G x G` covariance matrix is
never materialized at any point, for any context. See `shrunk_low_rank_covariance()` and its
math-correctness tests in [`../tests/test_covariance_shrinkage.py`](../tests/test_covariance_shrinkage.py).

### Covariance dropout

To keep the model from over-relying on the prior, `CovariancePriorCrossAttention` randomly
corrupts `(context, gene)` pairs during training (`cov_dropout` kwarg):

- `none`: substitute the `[NO_COV]` fallback token
- `wrong_context`: substitute a random *different* context's covariance
- `shuffled_gene`: substitute a random *different* gene's covariance in the same context

Whatever probability mass isn't spent on these keeps the correct pair.

### On-the-fly context registration (new contexts at inference)

The offline registry doesn't need to cover every context the model will ever see. A brand-new
context's covariance can be computed live from its own control cells and registered into a
running model, with the same hierarchical shrinkage as the offline path — this is the spec's
"Case 2/3: new context" and zero-shot test-time-adaptation workflow:

```python
model.register_covariance_context(
    "newdataset.newcelltype",     # matches "{dataset_name}.{cell_type}" batches will resolve to
    control_counts,               # (N, G_in) raw control-cell counts, no perturbation labels needed
    gene_names=adata.var_names,   # only needed if column order/vocab differs from the registry's
    parent_context="tian2019.ipsc",  # optional: nearest known sibling to shrink toward
    use_global=True,              # blend toward the registry's __global__ pooled pseudo-context
)
```

Once registered, any subsequent batch whose `cell_type`/`dataset_name` resolves to that context
name uses it immediately — no registry file rebuild, no model reload. `model.reset_covariance_contexts()`
drops everything registered this way, restoring the offline registry (useful between different
target datasets in a TTA loop).

**The key trick making this cheap**: a low-rank covariance `(V, Λ)` — including an *already
shrunk* one already sitting in the registry — can be treated as exact "pseudo-cells" for further
shrinkage, since `Σ = VΛVᵀ = (√Λ·Vᵀ)ᵀ(√Λ·Vᵀ)`. So blending a new context toward an existing
sibling context or the global pool never needs that existing context's original raw cell data —
just its already-computed `(V, Λ)`. Both the offline precompute path (`build_covariance_registry.py`)
and this runtime path (`CovariancePriorCrossAttention.register_context` /
`weighted_block_from_low_rank`) share the exact same math in
[`covariance_math.py`](../src/state/tx/models/covariance_math.py), so a context registered live
and one baked into the registry are computed identically.

`build_covariance_registry.py` writes a `"__global__"` pooled pseudo-context into every registry
specifically to make this available — it's a real usable shrinkage target, not a normal
biological context (it's never resolved by name during training).

## Files

| File | Purpose |
|---|---|
| [`build_covariance_registry.py`](build_covariance_registry.py) | Offline precompute script: TOML → `covariance_registry.npz` |
| `../src/state/tx/models/covariance_math.py` | Shared shrinkage math (weighting, pseudo-cell trick, stack+SVD) — used by both the offline script and online registration |
| `../src/state/tx/models/cross_attention.py` | `CovariancePriorCrossAttention` module, incl. `register_context` / `reset_online_contexts` |
| `../src/state/tx/models/state_transition.py` | Wiring into the model's forward pass; `register_covariance_context` / `reset_covariance_contexts` wrappers |
| `../src/state/configs/model/state.yaml` | `use_cov_cross_attn` / `cov_registry_path` / `cov_dropout` kwargs |
| `../tests/test_covariance_cross_attn.py` | Unit + model-integration tests (synthetic registry) |
| `../tests/test_covariance_shrinkage.py` | Offline shrinkage-math correctness tests |
| `../tests/test_covariance_online_registration.py` | On-the-fly context registration tests |

## Registry format

`covariance_registry.npz`:

| Key | Shape | Description |
|---|---|---|
| `V` | `(C, G, k)` float16 | Eigengene loadings per context (post-shrinkage) |
| `eigenvalues` | `(C, k)` float32 | Covariance eigenvalues per context (post-shrinkage) |
| `gene_variance` | `(C, G)` float32 | Exact per-gene variance (post-shrinkage) |
| `context_names` | `(C,)` str | `"{dataset_name}.{cell_type}"`, plus one `"__global__"` pooled pseudo-context |
| `gene_symbols` | `(G,)` str | Shared gene vocabulary (`var_names`) |
| `n_control_cells` | `(C,)` int64 | Own-context control-cell count (total pooled, for `__global__`) |
| `metadata_json` | scalar str | Per-context shrinkage weights & provenance (JSON) |

The `"__global__"` entry exists purely as a shrinkage target for online registration (see
below) — it's never resolved as a real training context.

## Usage

### 1. Build a registry

```bash
python covariance_prior/build_covariance_registry.py \
  --toml /path/to/dataset/dataset.toml \
  --out covariance_prior/my_covariance_registry.npz \
  --rank 64 \
  --pert-col gene \
  --control-pert non-targeting \
  --kappa 1000 \
  --delta 0.05
```

`--kappa` controls how many control cells "earn" full trust in a context's own estimate before
shrinkage kicks in (spec suggests 500–2000). `--delta` is the identity-ridge floor. Pass
`--no-shrinkage` to fall back to the pure per-context estimator.

### 2. Enable it in training

```bash
state tx train \
  data.kwargs.toml_config_path=/path/to/dataset.toml \
  data.kwargs.cell_type_key=cell_type \
  model=state \
  model.kwargs.use_cov_cross_attn=true \
  model.kwargs.cov_registry_path=/abs/path/to/my_covariance_registry.npz \
  ...
```

Can be enabled independently of, or alongside, `use_qc_cross_attn` — both sources share the
same cross-attention layers.

## Validation

Ran end-to-end against a real perturb-seq dataset (`tian2019`: 115 iPSC + 480 neuron control
cells, 33,752 genes) purely as an integration test:

- Confirmed the design principle empirically: `z_cov(ipsc, PPP2R1A)` vs. `z_cov(neuron,
  PPP2R1A)` have cosine similarity ≈ 0.17 (post-shrinkage) — meaningfully different despite
  heavy shrinkage (`w_own` ≈ 0.10 for `ipsc`, borrowing ≈ 0.85 from the `neuron` sibling pool).
- Instantiated the real `cell_load.PerturbationDataModule` against the dataset's TOML and
  confirmed batch fields (`cell_type`, `dataset_name`) resolve to exactly the registry's context
  keys with no glue code.
- Ran real forward passes and real optimizer steps (Adam, energy loss, covariance dropout
  active) through `StateTransitionPerturbationModel` using real dataloader batches — gradients
  flow into `cov_module`, no NaNs, output differs by context for the same perturbed gene.
- Registered a genuinely new context on the fly (a held-out slice of `neuron` control cells,
  labeled as if it were an unseen `"brandnew.neuronlike"` context not in the registry), shrunk
  toward `tian2019.ipsc` as parent plus the `__global__` pool, and confirmed a real forward pass
  referencing it produces output distinct from the `[NO_COV]` fallback.

This dataset is far too small (2 perturbed genes total) to draw any scientific conclusion from —
it only proves the mechanism is wired correctly before pointing it at a real training corpus.

## Known limitations

Deliberately out of scope for this first pass (see the spec for full detail on each):

- **Only the "eigengene loading vector" representation** (spec's Option 1). The response-template
  variant, the summary-statistics hybrid (positive/negative covariance mass, entropy, sparse
  top-neighbor tokens), and the sparse covariance-neighbor-token alternative are not implemented.
- **No normalization ablation.** Locked to raw-UMI-count covariance; the spec's proposed sweep
  (log1p CP10K, Pearson correlation, residualized-raw, low-rank-log) hasn't been run.
- **Covariance dropout covers 3 of the spec's 5 categories** (none / wrong-context /
  shuffled-gene, no separate "global-covariance" substitution — partially subsumed by shrinkage
  already blending global signal into the registry itself).
- **No `context_match_level` / confidence metadata** fed to the model alongside the token.
- **No ablation test harness or eval metrics** (same-context vs. wrong-context vs. shuffled vs.
  diagonal-only vs. rank-`k` sweep, Pearson Δ / DE direction match / etc.) — the spec's own
  validation plan for proving the gain comes from real context-specific structure.
- **No CIPHER-alignment auxiliary loss or InfoNCE context-mismatch objective** — both explicitly
  deferred by the spec itself until the base cross-attention-token version proves useful.
- **No automatic "nearest context" selection.** `register_context`'s `parent_context` must be
  named explicitly by the caller; the spec's `Σ_nearest_context` implies automatically finding
  the closest existing context (e.g. by mean-expression similarity) rather than requiring it as
  an argument.

None of these block using the feature as implemented; they're the natural next increments.
