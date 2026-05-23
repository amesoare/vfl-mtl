"""
figures/plot_confusion_matrices.py

Confusion matrices for VFL-MTL on the test set, one panel per task:
  IHM         (binary, threshold 0.5)
  Decomp      (binary, threshold 0.5)
  Phenotyping (multi-label: sum of per-label 2×2 CMs across 25 phenotypes)

CMs are averaged across 3 seeds and rounded to integers.

Usage (real data — run from project root on Snellius):
    python figures/plot_confusion_matrices.py --root /home/asoare/vfl_mlt

Usage (synthetic smoke test):
    python figures/plot_confusion_matrices.py --use_synthetic

Output: figures/confusion_matrices.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent))

from fl.client import VFLClient
from fl.server import VFLServer

SEEDS = [42, 123, 7]
THRESHOLD = 0.5

_C = ["#9d7b78", "#6a4c7a", "#2f283d", "#8a3c48", "#3d3527", "#b8c7d6", "#2f4a6d"]


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _real_loaders(root: str, batch_size: int = 256) -> dict:
    from data_prep.dataset import build_site_loaders
    return build_site_loaders(Path(root), "test", batch_size, num_workers=0)


def _synthetic_loaders(seed: int, batch_size: int = 64) -> dict:
    N, T = 256, 48
    g = torch.Generator()
    g.manual_seed(seed + 999)
    return {
        "A": DataLoader(TensorDataset(
            torch.randn(N, T, 7), torch.ones(N, T),
            torch.randint(0, 2, (N,), generator=g).float()), batch_size=batch_size),
        "B": DataLoader(TensorDataset(
            torch.randn(N, T, 4), torch.ones(N, T),
            torch.randint(0, 2, (N,), generator=g).float()), batch_size=batch_size),
        "C": DataLoader(TensorDataset(
            torch.randn(N, T, 3), torch.ones(N, T),
            torch.randint(0, 2, (N, 25), generator=g).float()), batch_size=batch_size),
    }


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def _infer_vfl_dims(ckpt: dict) -> tuple[int, int]:
    proj = ckpt["client_A"]["projection.weight"]
    lstm = ckpt["client_A"]["lstm.weight_ih_l0"]
    return int(proj.shape[0]), int(lstm.shape[0] // 4)


@torch.no_grad()
def collect_predictions(ckpt_path: Path, loaders: dict, device: torch.device
                        ) -> dict[str, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    ed, hd = _infer_vfl_dims(ckpt)

    clients = {s: VFLClient(input_dim=d, hidden_dim=hd, embed_dim=ed, lr=1e-3, device=device)
               for s, d in [("A", 7), ("B", 4), ("C", 3)]}
    server = VFLServer(embed_dim=ed, device=device)
    clients["A"].encoder.load_state_dict(ckpt["client_A"])
    clients["B"].encoder.load_state_dict(ckpt["client_B"])
    clients["C"].encoder.load_state_dict(ckpt["client_C"])
    server.model.load_state_dict(ckpt["server"])
    for c in clients.values():
        c.encoder.eval()
    server.model.eval()

    p_ihm, p_dec, p_phn = [], [], []
    y_ihm, y_dec, y_phn = [], [], []

    for bA, bB, bC in zip(loaders["A"], loaders["B"], loaders["C"]):
        xA, mA, yI = bA
        xB, mB, yD = bB
        xC, mC, yP = bC
        embs = {
            "A": clients["A"].eval_forward(xA.to(device), mA.to(device)),
            "B": clients["B"].eval_forward(xB.to(device), mB.to(device)),
            "C": clients["C"].eval_forward(xC.to(device), mC.to(device)),
        }
        out = server.predict(embs)
        p_ihm.append(out["ihm"].squeeze(-1).cpu().numpy())
        p_dec.append(out["decomp"].squeeze(-1).cpu().numpy())
        p_phn.append(out["pheno"].cpu().numpy())
        y_ihm.append(yI.numpy())
        y_dec.append(yD.numpy())
        y_phn.append(yP.numpy())

    return {
        "p_ihm":   np.concatenate(p_ihm),
        "p_dec":   np.concatenate(p_dec),
        "p_phn":   np.concatenate(p_phn),   # (N, 25)
        "y_ihm":   np.concatenate(y_ihm),
        "y_dec":   np.concatenate(y_dec),
        "y_phn":   np.concatenate(y_phn),   # (N, 25)
    }


# ---------------------------------------------------------------------------
# Confusion matrix helpers
# ---------------------------------------------------------------------------

def binary_cm(y_true: np.ndarray, y_prob: np.ndarray,
              threshold: float = THRESHOLD) -> np.ndarray:
    """Returns 2×2 CM [[TN, FP], [FN, TP]]."""
    y_pred = (y_prob >= threshold).astype(int)
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])


def pheno_cm(y_true: np.ndarray, y_prob: np.ndarray,
             threshold: float = THRESHOLD) -> np.ndarray:
    """Sum of per-label 2×2 CMs across all 25 phenotypes → single 2×2 CM."""
    cm = np.zeros((2, 2), dtype=int)
    for k in range(y_true.shape[1]):
        cm += binary_cm(y_true[:, k], y_prob[:, k], threshold)
    return cm


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def draw_cm(ax, cm: np.ndarray, title: str, accent: str, total_label: str):
    """Draw a single confusion matrix panel."""
    total = cm.sum()
    pct   = cm / total if total > 0 else cm.astype(float)

    # Background heatmap (percentage-normalised, one colour family)
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=pct.max() * 1.4, aspect="auto")

    cell_names = [["TN", "FP"], ["FN", "TP"]]
    labels = ["Neg (0)", "Pos (1)"]
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            p     = pct[i, j]
            name  = cell_names[i][j]
            cell_label = f"{name}\n{count:,}\n({p:.1%})"
            text_color = "white" if p > 0.45 else "#1a1a1a"
            ax.text(j, i, cell_label, ha="center", va="center",
                    fontsize=10, fontweight="bold", color=text_color)

    ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=9, rotation=90, va="center")
    ax.set_xlabel("Predicted", fontsize=10, labelpad=6)
    ax.set_ylabel("Actual",    fontsize=10, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", color=accent, pad=8)
    ax.tick_params(length=0)

    # Metric annotations below the matrix
    tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1   = (2 * ppv * sens / (ppv + sens)
            if (ppv + sens) > 0 else float("nan"))
    footnote = (f"Sens {sens:.3f}  ·  Spec {spec:.3f}  ·  "
                f"PPV {ppv:.3f}  ·  F1 {f1:.3f}  ·  n={total_label}")
    ax.set_xlabel(f"Predicted\n\n{footnote}", fontsize=8.5,
                  labelpad=4, color="#444444")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",          default=".")
    parser.add_argument("--ckpt_dir",      default="checkpoints")
    parser.add_argument("--output",        default="figures/confusion_matrices.png")
    parser.add_argument("--threshold",     type=float, default=THRESHOLD)
    parser.add_argument("--use_synthetic", action="store_true",
                        help="Use synthetic data instead of real MIMIC-III test set")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt_dir)

    cms_ihm, cms_dec, cms_phn = [], [], []

    for seed in SEEDS:
        ckpt_path = ckpt_dir / f"best_VFL-MTL_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {ckpt_path.name} not found")
            continue

        loaders = (_synthetic_loaders(seed) if args.use_synthetic
                   else _real_loaders(args.root))

        print(f"  Running inference: seed={seed} ({'synthetic' if args.use_synthetic else 'real'}) ...")
        preds = collect_predictions(ckpt_path, loaders, device)

        cms_ihm.append(binary_cm(preds["y_ihm"], preds["p_ihm"], args.threshold))
        cms_dec.append(binary_cm(preds["y_dec"], preds["p_dec"], args.threshold))
        cms_phn.append(pheno_cm (preds["y_phn"], preds["p_phn"], args.threshold))

    if not cms_ihm:
        print("No checkpoints found — nothing to plot.")
        return

    cm_ihm = np.round(np.mean(cms_ihm, axis=0)).astype(int)
    cm_dec = np.round(np.mean(cms_dec, axis=0)).astype(int)
    cm_phn = np.round(np.mean(cms_phn, axis=0)).astype(int)

    n_seeds = len(cms_ihm)
    n_ihm   = f"{cm_ihm.sum():,} ({n_seeds} seeds avg)"
    n_dec   = f"{cm_dec.sum():,} ({n_seeds} seeds avg)"
    n_phn   = f"{cm_phn.sum():,} label-instances ({n_seeds} seeds avg)"

    plt.rcParams.update({
        "figure.dpi":       150,
        "font.size":        10,
        "font.family":      "sans-serif",
        "axes.linewidth":   0.8,
    })

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        f"PRISM — Test-set Confusion Matrices  (threshold={args.threshold})",
        fontsize=12, fontweight="bold", y=1.02,
    )

    draw_cm(axes[0], cm_ihm, "In-hospital Mortality\n(IHM)",
            accent=_C[0], total_label=n_ihm)
    draw_cm(axes[1], cm_dec, "Decompensation\n(Decomp)",
            accent=_C[1], total_label=n_dec)
    draw_cm(axes[2], cm_phn, "Phenotyping\n(summed over 25 labels)",
            accent=_C[3], total_label=n_phn)

    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved: {args.output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
