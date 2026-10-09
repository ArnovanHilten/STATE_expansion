import numpy as np
from scipy import stats

from state.tx.models.interp_duplicate import (
    HalfSplitter,
    _benjamini_hochberg,
    apply_interpolated_duplicate,
    welch_overestim_var_pvalues,
)


def test_half_split_balanced_deterministic_and_streaming_consistent():
    key = ("ctx", "GENE1")
    full = HalfSplitter(seed=3).assign(key, 101)
    assert abs(int(full.sum()) - (101 - int(full.sum()))) <= 1

    # same seed -> same split; different seed -> different split
    assert np.array_equal(full, HalfSplitter(seed=3).assign(key, 101))
    assert not np.array_equal(full, HalfSplitter(seed=4).assign(key, 101))

    # cells arriving across several batches get the same assignment as in one go
    s = HalfSplitter(seed=3)
    chunks = np.concatenate([s.assign(key, n) for n in (7, 1, 50, 43)])
    assert np.array_equal(full, chunks)


def test_welch_overestim_matches_scipy_with_equal_n():
    rng = np.random.default_rng(0)
    n = 20
    a = rng.normal(1.0, 1.0, size=(n, 5))
    b = rng.normal(0.0, 2.0, size=(n, 5))
    ours = welch_overestim_var_pvalues(a.mean(0), a.var(0, ddof=1), n, b.mean(0), b.var(0, ddof=1))
    ref = stats.ttest_ind_from_stats(
        a.mean(0), a.std(0, ddof=1), n, b.mean(0), b.std(0, ddof=1), n, equal_var=False
    ).pvalue
    np.testing.assert_allclose(ours, ref)


def test_benjamini_hochberg_matches_scipy():
    p = np.random.default_rng(1).uniform(size=200) ** 3
    np.testing.assert_allclose(_benjamini_hochberg(p), stats.false_discovery_control(p))


def _group(ctx, pert, cells, td_mask, mean_baseline):
    gt, td = cells[~td_mask], cells[td_mask]
    return {
        "context": ctx,
        "pert_name": pert,
        "count": len(gt),
        "pred_sum": np.tile(mean_baseline, (len(gt), 1)).sum(0),
        "real_sum": gt.sum(0),
        "td_sum": td.sum(0),
        "td_sumsq": (td**2).sum(0),
        "td_n": len(td),
        "all_sum": cells.sum(0),
        "all_sumsq": (cells**2).sum(0),
        "all_n": len(cells),
    }


def test_interpolation_follows_duplicate_on_affected_genes_and_mean_elsewhere():
    rng = np.random.default_rng(0)
    n_genes, n_cells, affected = 50, 40, 3
    mean_baseline = np.zeros(n_genes)
    groups = []
    for i in range(30):
        cells = rng.normal(0.0, 1.0, size=(n_cells, n_genes))
        if i == 0:
            cells[:, :affected] += 5.0  # strong effect on the first 3 genes of perturbation 0
        td_mask = HalfSplitter(0).assign(("c", f"P{i}"), n_cells)
        groups.append(_group("c", f"P{i}", cells, td_mask, mean_baseline))
    groups.append(_group("c", "ctrl", rng.normal(size=(n_cells, n_genes)), np.zeros(n_cells, bool), mean_baseline))
    ctrl_before = groups[-1]["pred_sum"].copy()
    td_means = [g["td_sum"] / max(g["td_n"], 1) for g in groups]

    apply_interpolated_duplicate(groups, control_pert="ctrl")

    pred0 = groups[0]["pred_sum"] / groups[0]["count"]
    np.testing.assert_allclose(pred0[:affected], td_means[0][:affected], atol=0.05)  # follows the duplicate
    assert np.abs(pred0[affected:]).max() < np.abs(td_means[0][affected:]).max()  # shrunk toward the mean
    np.testing.assert_array_equal(groups[-1]["pred_sum"], ctrl_before)  # controls untouched


def test_tiny_groups_fall_back_to_mean_baseline():
    cells = np.ones((1, 4))
    g = _group("c", "P", cells, np.array([True]), np.full(4, 2.0))
    g["count"], g["pred_sum"] = 1, np.full(4, 2.0)
    apply_interpolated_duplicate([g], control_pert="ctrl")
    np.testing.assert_allclose(g["pred_sum"], np.full(4, 2.0))
