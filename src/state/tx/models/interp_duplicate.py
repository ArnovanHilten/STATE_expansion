import hashlib
import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from .base import PerturbationModel

logger = logging.getLogger(__name__)


class HalfSplitter:
    """
    Deterministic, balanced, seeded split of the cells of each (context, perturbation) group into a
    ground-truth half (GT) and a technical-duplicate half (TD).

    Cells of a group are paired in arrival order; a hash of (seed, group, pair index) decides which
    cell of each pair goes to TD. Halves therefore differ in size by at most one cell, and the same
    seed and loader order always reproduce the same split, so every model/baseline evaluated with
    the same seed is scored against the same ground-truth cells.
    """

    def __init__(self, seed: int = 0):
        self.seed = int(seed)
        self._counters: Dict[tuple, int] = defaultdict(int)

    def assign(self, key: tuple, n: int) -> np.ndarray:
        """Return a boolean mask of length n; True marks cells assigned to the technical duplicate."""
        start = self._counters[key]
        mask = np.empty(n, dtype=bool)
        for j in range(n):
            c = start + j
            bit = hashlib.blake2b(f"{self.seed}|{key[0]}|{key[1]}|{c // 2}".encode(), digest_size=1).digest()[0] & 1
            first_of_pair_is_td = bit == 0
            mask[j] = first_of_pair_is_td if c % 2 == 0 else not first_of_pair_is_td
        self._counters[key] = start + n
        return mask


def _benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0.0, 1.0)
    return out


def welch_overestim_var_pvalues(
    mean1: np.ndarray, var1: np.ndarray, n1: int, mean2: np.ndarray, var2: np.ndarray
) -> np.ndarray:
    """
    Per-gene two-sided Welch t-test p-values using scanpy's 't-test_overestim_var' convention:
    the reference group's sample size is set to the group's own size n1, which overestimates the
    variance of the difference and keeps p-values conservative for small groups.
    """
    from scipy import stats

    if n1 < 2:
        return np.ones_like(mean1)
    se2 = (var1 + var2) / n1
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (mean1 - mean2) / np.sqrt(se2)
        df = se2**2 / (((var1 / n1) ** 2 + (var2 / n1) ** 2) / (n1 - 1))
        p = 2.0 * stats.t.sf(np.abs(t), df)
    return np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)


def interpolated_duplicate(
    td_mean: np.ndarray,
    mean_baseline: np.ndarray,
    td_sum: np.ndarray,
    td_sumsq: np.ndarray,
    td_n: int,
    rest_sum: np.ndarray,
    rest_sumsq: np.ndarray,
    rest_n: int,
) -> np.ndarray:
    """
    Interpolated duplicate for one perturbation: alpha * technical_duplicate + (1 - alpha) * mean_baseline,
    with per-gene alpha = 1 - BH-adjusted p-value of the technical-duplicate half vs. all other perturbed cells.
    """
    if td_n < 2 or rest_n < 2:
        return mean_baseline.astype(np.float64)
    m1 = td_sum / td_n
    v1 = np.maximum((td_sumsq - td_n * m1**2) / (td_n - 1), 0.0)
    m2 = rest_sum / rest_n
    v2 = np.maximum((rest_sumsq - rest_n * m2**2) / (rest_n - 1), 0.0)
    padj = _benjamini_hochberg(welch_overestim_var_pvalues(m1, v1, td_n, m2, v2))
    alpha = 1.0 - padj
    return alpha * td_mean + (1.0 - alpha) * mean_baseline


def apply_interpolated_duplicate(groups: List[dict], control_pert: str) -> None:
    """
    Rewrite each non-control group's ``pred_sum`` in place so that ``pred_sum / count`` is its
    interpolated duplicate.

    Each group dict needs: context, pert_name, count (ground-truth half), pred_sum (model prediction
    summed over the ground-truth half; for this baseline that is the mean baseline), td_sum, td_sumsq,
    td_n (technical-duplicate half) and all_sum, all_sumsq, all_n (both halves).
    """
    totals: Dict[str, list] = {}
    for g in groups:
        if g["pert_name"] == control_pert:
            continue
        t = totals.setdefault(g["context"], [0.0, 0.0, 0])
        t[0] = t[0] + g["all_sum"]
        t[1] = t[1] + g["all_sumsq"]
        t[2] += g["all_n"]

    for g in groups:
        if g["pert_name"] == control_pert or g["count"] == 0:
            continue
        tot_sum, tot_sumsq, tot_n = totals[g["context"]]
        mean_baseline = g["pred_sum"] / g["count"]
        interp = interpolated_duplicate(
            td_mean=g["td_sum"] / max(g["td_n"], 1),
            mean_baseline=mean_baseline,
            td_sum=g["td_sum"],
            td_sumsq=g["td_sumsq"],
            td_n=g["td_n"],
            rest_sum=tot_sum - g["all_sum"],
            rest_sumsq=tot_sumsq - g["all_sumsq"],
            rest_n=tot_n - g["all_n"],
        )
        g["pred_sum"] = interp * g["count"]


class InterpDuplicatePerturbationModel(PerturbationModel):
    """
    Positive-control baseline from "Deep learning perturbation models can outperform baselines on
    calibrated metrics" (Miller, Mejia et al., Nat. Biotechnol. 2026): the *interpolated duplicate*.

    For every held-out (context, perturbation), the cells are split in half. One half (TD) is used as
    the "prediction", the other (GT) as ground truth. The TD half is blended gene by gene with the
    mean baseline, using alpha = 1 - adjusted p-value of the TD half vs. all other perturbed cells,
    so affected genes follow the duplicate and unaffected genes follow the mean baseline.

    This is an oracle: it reads held-out cells, so it is a reference for calibrating metrics and
    contextualising models, not a deployable predictor. Only the mean baseline is learned here
    (in ``on_fit_start``, same style as the other baselines). The split and interpolation happen in
    ``state tx predict --pseudobulk``, which scores the baseline against the GT half only.

    Mean baseline: for each cell type, the average over training perturbations of their mean
    expression (every perturbation weighted equally, controls excluded); cell types unseen in
    training fall back to the same average over all training perturbations.
    """

    is_interp_duplicate = True

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        pert_dim: int,
        dropout: float = 0.0,
        lr: float = 1e-3,
        loss_fn=nn.MSELoss(),
        embed_key: str = None,
        output_space: str = "gene",
        gene_names=None,
        **kwargs,
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            pert_dim=pert_dim,
            dropout=dropout,
            lr=lr,
            loss_fn=loss_fn,
            embed_key=embed_key,
            output_space=output_space,
            gene_names=gene_names,
            **kwargs,
        )
        self.celltype_means: Dict[str, torch.Tensor] = {}
        self.global_mean: torch.Tensor = torch.zeros(self.output_dim)
        # Dummy parameter so that Lightning sees something to "optimize"
        self.dummy_param = nn.Parameter(torch.zeros(1, requires_grad=True))

    def _output_key(self) -> str:
        if (self.embed_key and self.embed_key != "X_hvg" and self.output_space == "gene") or (
            self.embed_key and self.output_space == "all"
        ):
            return "pert_cell_counts"
        return "pert_cell_emb"

    def on_fit_start(self):
        """Compute the mean baseline from the training dataloader."""
        super().on_fit_start()

        train_loader = self.trainer.datamodule.train_dataloader()
        if train_loader is None:
            logger.warning("No train dataloader found. Cannot compute mean baseline.")
            return

        # (cell type, perturbation) -> running sum / count
        sums = defaultdict(lambda: defaultdict(lambda: {"sum": torch.zeros(self.output_dim), "count": 0}))
        with torch.no_grad():
            for batch in train_loader:
                X = batch[self._output_key()].float().cpu()
                pert_names = batch["pert_name"]
                cell_types = batch["cell_type"]
                for i in range(len(X)):
                    p_name = str(pert_names[i])
                    if p_name == self.control_pert:
                        continue
                    entry = sums[str(cell_types[i])][p_name]
                    entry["sum"] += X[i]
                    entry["count"] += 1

        all_pert_means = []
        for ct_name, pert_dict in sums.items():
            pert_means = [e["sum"] / e["count"] for e in pert_dict.values() if e["count"] > 0]
            if not pert_means:
                continue
            self.celltype_means[ct_name] = torch.stack(pert_means).mean(0)
            all_pert_means.extend(pert_means)

        if all_pert_means:
            self.global_mean = torch.stack(all_pert_means).mean(0)
        else:
            logger.warning("No perturbed cells found. Mean baseline set to zeros.")
        logger.info(
            "InterpDuplicate: computed mean baseline for %d cell types from %d (cell type, perturbation) pseudobulks.",
            len(self.celltype_means),
            len(all_pert_means),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Controls are copied through (as in the context-mean baseline). Perturbed cells get the mean
        baseline of their cell type; the oracle interpolation is applied at prediction time.
        """
        B = len(batch["pert_name"])
        device = self.dummy_param.device
        output_key = self._output_key()
        pred_out = torch.zeros((B, self.output_dim), device=device)
        for i in range(B):
            if str(batch["pert_name"][i]) == self.control_pert:
                pred_out[i] = batch[output_key][i].to(device)
            else:
                mean = self.celltype_means.get(str(batch["cell_type"][i]), self.global_mean)
                pred_out[i] = mean.to(device)
        return pred_out

    def configure_optimizers(self):
        if len(list(self.parameters())) > 0:
            return torch.optim.Adam(self.parameters(), lr=self.lr)
        return None

    def training_step(self, batch, batch_idx):
        pred = self(batch)
        loss = self.loss_fn(pred, batch[self._output_key()])
        self.log("train_loss", loss, prog_bar=True)
        return None

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["celltype_means"] = {ct: m.cpu().numpy() for ct, m in self.celltype_means.items()}
        checkpoint["global_mean"] = self.global_mean.cpu().numpy()
        logger.info("InterpDuplicate: saved mean baseline to checkpoint.")

    def on_load_checkpoint(self, checkpoint):
        super().on_load_checkpoint(checkpoint)
        self.celltype_means = {
            ct: torch.tensor(m, dtype=torch.float32) for ct, m in checkpoint.get("celltype_means", {}).items()
        }
        if "global_mean" in checkpoint:
            self.global_mean = torch.tensor(checkpoint["global_mean"], dtype=torch.float32)
        else:
            logger.warning("InterpDuplicate: no global_mean in checkpoint. Using zero vector.")
            self.global_mean = torch.zeros(self.output_dim)
        logger.info("InterpDuplicate: loaded mean baseline for %d cell types.", len(self.celltype_means))

    def encode_perturbation(self, pert: torch.Tensor) -> torch.Tensor:
        return pert

    def encode_basal_expression(self, expr: torch.Tensor) -> torch.Tensor:
        return expr

    def perturb(self, pert: torch.Tensor, basal: torch.Tensor) -> torch.Tensor:
        return basal

    def _build_networks(self):
        pass
