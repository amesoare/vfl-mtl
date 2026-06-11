"""
tests/check_model_reliability.py - Model reliability checks for PRISM (VFL-MTL).

Four unified figures saved to figures/reliability/:

  calibration_curves.png   - 3 panels (tasks), one line per epsilon level including no-DP.
                             Requires checkpoints/.
  confusion_matrices.png   - rows = epsilon subset [no-DP, eps=2, eps=0.5], cols = tasks.
                             Requires checkpoints/.
  seed_stability.png       - 3 panels (tasks), x = epsilon levels, dots per seed + mean.
                             Reads results/test_results_dp.csv.
  train_test_gap.png       - 3 panels (tasks), x = epsilon levels, val vs. test AUC.
                             Reads results/privacy_utility_combined.csv + test_results_dp.csv.

Usage:
    python tests/check_model_reliability.py --root /home/asoare/vfl_mlt
    python tests/check_model_reliability.py --root . --skip_model   # CSV checks only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.dataset import build_site_loaders
from fl.client import VFLClient
from fl.server import VFLServer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEEDS     = [42, 123, 7]
DP_SEEDS  = [42, 123, 7, 17, 99]
EPS_LEVELS = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
EPS_LABELS = {0.5: "eps=0.5", 1.0: "eps=1", 2.0: "eps=2",
              5.0: "eps=5", 10.0: "eps=10", float("inf"): "No DP"}
EPS_X_TICKS = ["0.5", "1", "2", "5", "10", "No DP"]
UTILITY_FLOORS = {"IHM": 0.75, "Decomp": 0.70, "Pheno": 0.65}
TASKS = [("IHM", "ihm"), ("Decomp", "decomp"), ("Pheno", "pheno")]

# Shared palette — matches figures/resilience_variance.py and plot_baselines.py
_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]
TASK_COLORS = {"IHM": _C[1], "Decomp": _C[2], "Pheno": _C[3]}

# Alpha per epsilon: darkest = no DP (most utility), lightest = lowest epsilon (most noise)
EPS_ALPHAS = {0.5: 0.25, 1.0: 0.40, 2.0: 0.55, 5.0: 0.70, 10.0: 0.85, float("inf"): 1.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _out_dir(root: Path) -> Path:
    d = root / "figures" / "reliability"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _dp_ckpt_path(ckpt_dir: Path, eps: float, seed: int) -> Path:
    if eps == float("inf"):
        return ckpt_dir / f"best_VFL-MTL_seed{seed}.pt"
    eps_str = str(eps)
    return ckpt_dir / f"best_DP-uniform-eps{eps_str}-seed{seed}_seed{seed}.pt"


def _load_vfl_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    proj = ckpt["client_A"]["projection.weight"]
    lstm = ckpt["client_A"]["lstm.weight_ih_l0"]
    embed_dim  = int(proj.shape[0])
    hidden_dim = int(lstm.shape[0] // 4)
    dev = torch.device(device)
    clients = {
        s: VFLClient(input_dim=d, hidden_dim=hidden_dim, embed_dim=embed_dim,
                     lr=1e-3, device=dev)
        for s, d in [("A", 7), ("B", 4), ("C", 3)]
    }
    server = VFLServer(embed_dim=embed_dim, device=dev)
    clients["A"].encoder.load_state_dict(ckpt["client_A"])
    clients["B"].encoder.load_state_dict(ckpt["client_B"])
    clients["C"].encoder.load_state_dict(ckpt["client_C"])
    server.model.load_state_dict(ckpt["server"])
    return clients, server


@torch.no_grad()
def _get_predictions(clients, server, loaders):
    for c in clients.values():
        c.encoder.eval()
    server.model.eval()
    preds  = {"ihm": [], "decomp": [], "pheno": []}
    labels = {"ihm": [], "decomp": [], "pheno": []}
    for bA, bB, bC in zip(loaders["A"], loaders["B"], loaders["C"]):
        xA, mA, yI = bA; xB, mB, yD = bB; xC, mC, yP = bC
        embs = {
            "A": clients["A"].eval_forward(xA, mA),
            "B": clients["B"].eval_forward(xB, mB),
            "C": clients["C"].eval_forward(xC, mC),
        }
        out = server.predict(embs)
        preds["ihm"].append(out["ihm"].squeeze(-1).cpu().numpy())
        preds["decomp"].append(out["decomp"].squeeze(-1).cpu().numpy())
        preds["pheno"].append(out["pheno"].cpu().numpy())
        labels["ihm"].append(yI.numpy())
        labels["decomp"].append(yD.numpy())
        labels["pheno"].append(yP.numpy())
    return (
        {k: np.concatenate(v) for k, v in preds.items()},
        {k: np.concatenate(v) for k, v in labels.items()},
    )


def _load_eps_predictions(ckpt_dir: Path, loaders, device: str,
                          eps_list: list, seeds: list) -> dict[float, dict]:
    """Load and pool predictions for each epsilon level across seeds."""
    eps_preds = {}
    for eps in eps_list:
        p_all = {"ihm": [], "decomp": [], "pheno": []}
        l_all = {"ihm": [], "decomp": [], "pheno": []}
        found = 0
        for seed in seeds:
            ckpt = _dp_ckpt_path(ckpt_dir, eps, seed)
            if not ckpt.exists():
                continue
            clients, server = _load_vfl_checkpoint(ckpt, device)
            p, l = _get_predictions(clients, server, loaders)
            for k in p:
                p_all[k].append(p[k]); l_all[k].append(l[k])
            found += 1
        if found == 0:
            print(f"  [SKIP eps={EPS_LABELS[eps]}] no checkpoints found")
            continue
        eps_preds[eps] = {
            "preds":  {k: np.concatenate(v) for k, v in p_all.items()},
            "labels": {k: np.concatenate(v) for k, v in l_all.items()},
            "n_seeds": found,
        }
        print(f"  eps={EPS_LABELS[eps]}: {found} seeds pooled")
    return eps_preds


def _parse_eps(val) -> float:
    try:
        f = float(val)
        return float("inf") if f > 100 else f
    except (ValueError, TypeError):
        return float("inf")


# ---------------------------------------------------------------------------
# Figure 1 — calibration_curves.png
# ---------------------------------------------------------------------------

def make_calibration_figure(root: Path, device: str = "cpu") -> bool:
    """
    3 panels (one per task), one line per epsilon level.
    Darker = more noise (lower eps). Dashed grey = perfect calibration.
    """
    ckpt_dir = root / "checkpoints"
    if not ckpt_dir.exists():
        print("[Calibration] SKIP — checkpoints/ not found.")
        return False

    print("[Calibration] Loading test data ...")
    loaders = build_site_loaders(root, "test", batch_size=256)
    # no-DP uses SEEDS [42,123,7]; DP levels use DP_SEEDS [42,123,7,17,99]
    dp_levels = [e for e in EPS_LEVELS if e != float("inf")]
    eps_preds = _load_eps_predictions(ckpt_dir, loaders, device, dp_levels, DP_SEEDS)
    eps_preds.update(_load_eps_predictions(ckpt_dir, loaders, device, [float("inf")], SEEDS))
    if not eps_preds:
        print("[Calibration] SKIP — no checkpoints found.")
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    for ax, (task_name, task_key) in zip(axes, TASKS):
        color = TASK_COLORS[task_name]
        for eps in EPS_LEVELS:
            if eps not in eps_preds:
                continue
            p = eps_preds[eps]["preds"][task_key]
            y = eps_preds[eps]["labels"][task_key]
            alpha = EPS_ALPHAS[eps]
            label = f"PRISM ({EPS_LABELS[eps]})"
            try:
                if task_key == "pheno":
                    fp_list, mp_list = [], []
                    for i in range(y.shape[1]):
                        if y[:, i].sum() > 0:
                            fp, mp = calibration_curve(y[:, i], p[:, i],
                                                       n_bins=10, strategy="uniform")
                            fp_list.append(fp); mp_list.append(mp)
                    if fp_list:
                        mp_mean = np.mean([np.interp(np.linspace(0, 1, 10), m, f)
                                           for m, f in zip(mp_list, fp_list)], axis=0)
                        ax.plot(np.linspace(0, 1, 10), mp_mean, "o-",
                                color=color, alpha=alpha, linewidth=1.5,
                                markersize=4, label=label)
                else:
                    frac_pos, mean_pred = calibration_curve(y, p, n_bins=10,
                                                             strategy="uniform")
                    ax.plot(mean_pred, frac_pos, "o-",
                            color=color, alpha=alpha, linewidth=1.5,
                            markersize=4, label=label)
            except ValueError:
                pass

        ax.plot([0, 1], [0, 1], color="#888888", linestyle="--",
                linewidth=1.2, label="Perfect calibration")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        suffix = " (mean over 25 labels)" if task_key == "pheno" else ""
        ax.set_title(f"Calibration: {task_name}{suffix}")
        ax.legend(fontsize=7)

    path = _out_dir(root) / "calibration_curves.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")
    return True


# ---------------------------------------------------------------------------
# Figure 2 — confusion_matrices.png
# ---------------------------------------------------------------------------

def _youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Return threshold maximising sensitivity + specificity - 1 (Youden index)."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    idx = np.argmax(tpr - fpr)
    return float(thresholds[idx])


def _youden_thresholds(val_data: dict) -> dict[str, float | np.ndarray]:
    """Compute per-task Youden thresholds from validation predictions."""
    thresholds = {}
    for task_key in ("ihm", "decomp"):
        thresholds[task_key] = _youden_threshold(
            val_data["labels"][task_key], val_data["preds"][task_key]
        )
    # per-label thresholds for phenotyping
    y = val_data["labels"]["pheno"]
    p = val_data["preds"]["pheno"]
    per_label = np.full(y.shape[1], 0.5)
    for i in range(y.shape[1]):
        if y[:, i].sum() > 0:
            per_label[i] = _youden_threshold(y[:, i], p[:, i])
    thresholds["pheno"] = per_label
    return thresholds


def make_confusion_figure(root: Path, device: str = "cpu") -> bool:
    """
    Single row: no-DP model only, one panel per task.
    Task-specific colormaps (white to task color). Cell values shown.
    """
    ckpt_dir = root / "checkpoints"
    if not ckpt_dir.exists():
        print("[Confusion] SKIP — checkpoints/ not found.")
        return False

    print("[Confusion] Loading val data for Youden thresholds ...")
    val_loaders  = build_site_loaders(root, "val",  batch_size=256)
    test_loaders = build_site_loaders(root, "test", batch_size=256)

    val_preds  = _load_eps_predictions(ckpt_dir, val_loaders,  device, [float("inf")], SEEDS)
    test_preds = _load_eps_predictions(ckpt_dir, test_loaders, device, [float("inf")], SEEDS)

    if float("inf") not in test_preds:
        print("[Confusion] SKIP — no-DP checkpoints not found.")
        return False

    thresholds = _youden_thresholds(val_preds[float("inf")])
    data = test_preds[float("inf")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)

    for ax, (task_name, task_key) in zip(axes, TASKS):
        color = TASK_COLORS[task_name]
        cmap  = LinearSegmentedColormap.from_list(task_name, ["#ffffff", color])
        p = data["preds"][task_key]
        y = data["labels"][task_key]

        if task_key == "pheno":
            per_label_thr = thresholds["pheno"]
            tpr_list = []
            for i in range(y.shape[1]):
                if y[:, i].sum() > 0:
                    thr_i = per_label_thr[i]
                    cm_i = confusion_matrix(y[:, i], (p[:, i] >= thr_i).astype(int))
                    tn, fp, fn, tp = cm_i.ravel() if cm_i.size == 4 else (0, 0, 0, 0)
                    tpr_list.append(tp / (tp + fn) if (tp + fn) > 0 else 0.0)
                else:
                    tpr_list.append(float("nan"))
            tpr_grid = np.array(tpr_list).reshape(5, 5)
            im = ax.imshow(tpr_grid, cmap=cmap, vmin=0, vmax=1)
            for i in range(5):
                for j in range(5):
                    v = tpr_grid[i, j]
                    ax.text(j, i, f"{v:.2f}" if not np.isnan(v) else "n/a",
                            ha="center", va="center", fontsize=7, color="black")
            median_thr = float(np.median(per_label_thr))
            ax.set_title(f"{task_name}: TPR per label (No DP, Youden thr≈{median_thr:.2f})")
            ax.set_xlabel("Label index (column)")
            ax.set_ylabel("Label index (row)")
            plt.colorbar(im, ax=ax, label="TPR")
        else:
            thr = thresholds[task_key]
            cm = confusion_matrix(y, (p >= thr).astype(int))
            ax.imshow(cm, cmap=cmap)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color="black", fontsize=12)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(["Predicted 0", "Predicted 1"])
            ax.set_yticklabels(["True 0", "True 1"])
            ax.set_title(f"{task_name}: No DP (Youden thr={thr:.2f})")

    path = _out_dir(root) / "confusion_matrices.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")
    return True


# ---------------------------------------------------------------------------
# Figure 3 — seed_stability.png
# ---------------------------------------------------------------------------

def make_seed_stability_figure(root: Path) -> bool:
    """
    3 panels (one per task).
    x-axis: epsilon levels [0.5, 1, 2, 5, 10, No DP].
    Dots = individual seeds, line = mean, shading = std, dashed = clinical floor.
    Source: results/test_results_dp.csv.
    """
    csv = root / "results" / "test_results_dp.csv"
    if not csv.exists():
        print("[Seed Stability] SKIP — test_results_dp.csv not found.")
        return False

    df = pd.read_csv(csv)
    df["eps_float"] = df["epsilon_level"].apply(_parse_eps)

    task_cols = {"IHM": "ihm_auroc", "Decomp": "decomp_auroc", "Pheno": "pheno_macro_auroc"}
    seeds_present = sorted(df["seed"].unique())
    seed_markers = {s: m for s, m in zip(seeds_present, ["o", "s", "^", "D", "v"])}
    seed_grays   = {s: g for s, g in zip(seeds_present,
                    ["#222222", "#555555", "#888888", "#aaaaaa", "#cccccc"])}
    print("\n[Seed Stability] Test AUC across epsilon levels")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for ax, (task_name, _) in zip(axes, TASKS):
        col = task_cols[task_name]
        if col not in df.columns:
            continue
        color = TASK_COLORS[task_name]
        means, stds, x_positions = [], [], []

        for xi, eps in enumerate(EPS_LEVELS):
            eps_sub = df[df["eps_float"] == eps]
            if eps_sub.empty or col not in eps_sub.columns:
                continue
            vals = eps_sub[col].dropna()
            if vals.empty:
                continue
            # one dot per seed, distinguished by marker + gray shade
            for _, row in eps_sub.dropna(subset=[col]).iterrows():
                seed = int(row["seed"])
                ax.scatter(xi, row[col],
                           marker=seed_markers.get(seed, "o"),
                           color=seed_grays.get(seed, "#555555"),
                           edgecolors=color, linewidths=1.2,
                           s=55, alpha=0.9, zorder=3)
            means.append(vals.mean()); stds.append(vals.std()); x_positions.append(xi)

        if means:
            means = np.array(means); stds = np.array(stds)
            ax.plot(x_positions, means, color=color, linewidth=2, zorder=4, label="PRISM mean")
            ax.fill_between(x_positions, means - stds, means + stds,
                            color=color, alpha=0.15, zorder=2)

        floor = UTILITY_FLOORS[task_name]
        ax.axhline(floor, color="#888888", linestyle="--", linewidth=1.2,
                   label=f"Clinical floor ({floor})")

        # seed legend entries (shown once in first panel only)
        if ax is axes[0]:
            for seed in seeds_present:
                ax.scatter([], [], marker=seed_markers[seed],
                           color=seed_grays[seed], edgecolors="#444444",
                           linewidths=1.0, s=45, label=f"seed {seed}")

        ax.set_xticks(range(len(EPS_LEVELS)))
        ax.set_xticklabels(EPS_X_TICKS, fontsize=8)
        ax.set_xlabel("Privacy budget (epsilon)")
        ax.set_ylabel("AUC-ROC")
        all_shown = [df[df["eps_float"] == e][col].dropna().values
                     for e in EPS_LEVELS if not df[df["eps_float"] == e][col].dropna().empty]
        if all_shown:
            ymin = max(0, np.concatenate(all_shown).min() - 0.05)
            ymax = min(1, np.concatenate(all_shown).max() + 0.05)
            ax.set_ylim(ymin, ymax)
        ax.set_title(f"Seed Stability: {task_name}")
        ax.legend(fontsize=8)
        print(f"  {task_name}: " + " | ".join(
            f"eps={EPS_LABELS[e]} mean={df[df['eps_float']==e][col].mean():.3f}"
            for e in EPS_LEVELS if not df[df['eps_float']==e][col].dropna().empty))

    path = _out_dir(root) / "seed_stability.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")
    return True


# ---------------------------------------------------------------------------
# Figure 4 — train_test_gap.png
# ---------------------------------------------------------------------------

def make_train_test_gap_figure(root: Path) -> bool:
    """
    3 panels (one per task).
    x-axis: epsilon levels [0.5, 1, 2, 5, 10, No DP].
    Two lines: validation AUC (mean +- std) and test AUC (mean +- std).
    Val source: results/privacy_utility_combined.csv (last round per seed/eps).
    Test source: results/test_results_dp.csv.
    """
    val_csv  = root / "results" / "privacy_utility_combined.csv"
    test_csv = root / "results" / "test_results_dp.csv"

    if not val_csv.exists() or not test_csv.exists():
        print("[Train/Test Gap] SKIP — privacy_utility_combined.csv or test_results_dp.csv not found.")
        return False

    val_df  = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    val_df["eps_float"]  = val_df["epsilon_level"].apply(_parse_eps)
    test_df["eps_float"] = test_df["epsilon_level"].apply(_parse_eps)

    val_task_cols  = {"IHM": "val_ihm_auroc",         "Decomp": "val_decomp_auroc",
                      "Pheno": "val_pheno_macro_auroc"}
    test_task_cols = {"IHM": "ihm_auroc",              "Decomp": "decomp_auroc",
                      "Pheno": "pheno_macro_auroc"}

    print("\n[Train/Test Gap] Validation vs. test AUC across epsilon levels")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)

    for ax, (task_name, _) in zip(axes, TASKS):
        color = TASK_COLORS[task_name]
        vcol  = val_task_cols[task_name]
        tcol  = test_task_cols[task_name]

        val_means, val_stds, test_means, test_stds, x_pos = [], [], [], [], []

        for xi, eps in enumerate(EPS_LEVELS):
            # Val: last round per seed for this eps
            v_sub = val_df[val_df["eps_float"] == eps]
            if vcol in v_sub.columns and not v_sub.empty:
                last_per_seed = v_sub.groupby("seed")[vcol].last().dropna()
                if not last_per_seed.empty:
                    val_means.append(last_per_seed.mean())
                    val_stds.append(last_per_seed.std())
                else:
                    val_means.append(np.nan); val_stds.append(0)
            else:
                val_means.append(np.nan); val_stds.append(0)

            # Test
            t_sub = test_df[test_df["eps_float"] == eps]
            if tcol in t_sub.columns and not t_sub.empty:
                t_vals = t_sub[tcol].dropna()
                test_means.append(t_vals.mean()); test_stds.append(t_vals.std())
            else:
                test_means.append(np.nan); test_stds.append(0)
            x_pos.append(xi)

        val_means  = np.array(val_means);  val_stds  = np.array(val_stds)
        test_means = np.array(test_means); test_stds = np.array(test_stds)
        x_pos = np.array(x_pos)

        ax.plot(x_pos, val_means,  color=color, linewidth=2, label="PRISM (validation)")
        ax.fill_between(x_pos, val_means - val_stds, val_means + val_stds,
                        color=color, alpha=0.15)
        ax.plot(x_pos, test_means, color=color, linewidth=2, linestyle="--",
                label="PRISM (test)")
        ax.fill_between(x_pos, test_means - test_stds, test_means + test_stds,
                        color=color, alpha=0.08)

        floor = UTILITY_FLOORS[task_name]
        ax.axhline(floor, color="#888888", linestyle=":", linewidth=1.2,
                   label=f"Clinical floor ({floor})")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(EPS_X_TICKS, fontsize=8)
        ax.set_xlabel("Privacy budget (epsilon)")
        ax.set_ylabel("AUC-ROC")
        all_v = np.concatenate([val_means[~np.isnan(val_means)],
                                test_means[~np.isnan(test_means)], [floor]])
        if len(all_v):
            ax.set_ylim(max(0, all_v.min() - 0.05), min(1, all_v.max() + 0.05))
        ax.set_title(f"Validation vs. Test AUC: {task_name}")
        ax.legend(fontsize=8)

    path = _out_dir(root) / "train_test_gap.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  Saved: {path}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root",        default=".", help="Project root directory")
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--skip_model",     action="store_true",
                        help="Skip figures that require checkpoints")
    parser.add_argument("--confusion_only", action="store_true",
                        help="Only regenerate confusion_matrices.png (fast)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    print(f"Root: {root}")
    print("=" * 60)

    if args.confusion_only:
        make_confusion_figure(root, device=args.device)
    else:
        if not args.skip_model:
            make_calibration_figure(root, device=args.device)
            make_confusion_figure(root, device=args.device)
        make_seed_stability_figure(root)
        make_train_test_gap_figure(root)

    print("\nAll checks complete. Figures saved to:", root / "figures" / "reliability")


if __name__ == "__main__":
    main()
