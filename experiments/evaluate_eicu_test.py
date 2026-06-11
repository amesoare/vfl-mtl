
"""
experiments/evaluate_eicu_test.py — Final test-set evaluation on eICU.

Mirrors evaluate_test.py but for eICU:
  - Site B: input_dim=3, task=RLOS regression
  - Decomp metric: MAE/RMSE instead of AUC-ROC/AUC-PR
  - Checkpoint dir: checkpoints/eicu/
  - Checkpoint names: best_eICU_{model}_seed{N}.pt (VFL/ST variants)
                      best_local_{site}_seed{N}.pt  (local baselines)
                      best_centralized_seed{N}.pt   (centralized)

Output: results/eicu_test_results.csv
  columns: model, seed, ihm_auc_roc, ihm_auc_pr, rlos_mae, rlos_rmse,
           pheno_macro_auc, pheno_micro_auc

Usage:
    python experiments/evaluate_eicu_test.py --root /home/asoare/vfl_mlt
    python experiments/evaluate_eicu_test.py --root . --use_synthetic
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.dataset import build_site_loaders
from experiments.metrics import ihm_metrics, pheno_metrics, rlos_metrics
from fl.client import VFLClient
from fl.server import VFLServer
from model.encoder import SiteEncoder
from baselines.local_only import _SITE_CFG_EICU, _LocalHead
from baselines.centralized import (
    CentralizedDatasetEICU, CentralizedEncoder, _collate, _EMBED_DIM,
)
from model.mmoe import MMoEServer

SEEDS      = [42, 123, 7]
TASK_TYPES = {"decomp": "regression"}

# eICU site input dims
_EICU_DIMS = {"A": 7, "B": 3, "C": 3}


# ---------------------------------------------------------------------------
# Test loaders
# ---------------------------------------------------------------------------

def _real_test_loaders(root: str, batch_size: int, num_workers: int = 0) -> dict:
    return build_site_loaders(Path(root), "test", batch_size, num_workers,
                              dataset="eicu")


def _synthetic_test_loaders(batch_size: int, seed: int) -> dict:
    N = 64
    g = torch.Generator(); g.manual_seed(seed + 999)
    return {
        "A": DataLoader(TensorDataset(
            torch.randn(N, 1, 7), torch.ones(N, 1),
            torch.randint(0, 2, (N,), generator=g).float()), batch_size=batch_size),
        "B": DataLoader(TensorDataset(
            torch.randn(N, 1, 3), torch.ones(N, 1),
            torch.rand(N, generator=g) * 10.0), batch_size=batch_size),
        "C": DataLoader(TensorDataset(
            torch.randn(N, 1, 3), torch.ones(N, 1),
            torch.randint(0, 2, (N, 25), generator=g).float()), batch_size=batch_size),
    }


# ---------------------------------------------------------------------------
# VFL-MTL / ST-* evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def _eval_vfl(clients: dict, server: VFLServer, loaders: dict) -> dict[str, float]:
    for c in clients.values(): c.encoder.eval()
    server.model.eval()
    preds  = {"ihm": [], "rlos": [], "pheno": []}
    labels = {"ihm": [], "rlos": [], "pheno": []}
    for bA, bB, bC in zip(loaders["A"], loaders["B"], loaders["C"]):
        xA, mA, yI = bA; xB, mB, yD = bB; xC, mC, yP = bC
        embs = {s: clients[s].eval_forward(x, m)
                for s, (x, m) in zip("ABC", [(xA, mA), (xB, mB), (xC, mC)])}
        out = server.predict(embs)
        preds["ihm"].append(out["ihm"].squeeze(-1).cpu().numpy())
        preds["rlos"].append(out["decomp"].squeeze(-1).cpu().numpy())
        preds["pheno"].append(out["pheno"].cpu().numpy())
        labels["ihm"].append(yI.numpy())
        labels["rlos"].append(yD.numpy())
        labels["pheno"].append(yP.numpy())

    p_ihm  = np.concatenate(preds["ihm"]);  y_ihm  = np.concatenate(labels["ihm"])
    p_rlos = np.concatenate(preds["rlos"]); y_rlos = np.concatenate(labels["rlos"])
    p_phn  = np.concatenate(preds["pheno"]); y_phn = np.concatenate(labels["pheno"])
    return {
        **{f"ihm_{k}":   v for k, v in ihm_metrics(y_ihm, p_ihm).items()},
        **{f"rlos_{k}":  v for k, v in rlos_metrics(y_rlos, p_rlos).items()},
        **{f"pheno_{k}": v for k, v in pheno_metrics(y_phn, p_phn).items()},
    }


def _infer_vfl_dims(ckpt: dict) -> tuple[int, int]:
    proj = ckpt["client_A"]["projection.weight"]
    lstm = ckpt["client_A"]["lstm.weight_ih_l0"]
    return int(proj.shape[0]), int(lstm.shape[0] // 4)


def eval_vfl_mtl(ckpt_dir: Path, loaders: dict, device: str = "cpu") -> list[dict]:
    rows = []
    dev  = torch.device(device)
    for model_name in ["eICU_VFL-MTL", "eICU_ST-IHM", "eICU_ST-RLOS", "eICU_ST-Pheno"]:
        for seed in SEEDS:
            ckpt_path = ckpt_dir / f"best_{model_name}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"  [SKIP] {ckpt_path.name}")
                continue
            ckpt    = torch.load(ckpt_path, map_location=device, weights_only=True)
            ed, hd  = _infer_vfl_dims(ckpt)
            clients = {s: VFLClient(input_dim=d, hidden_dim=hd, embed_dim=ed,
                                    lr=1e-3, device=dev)
                       for s, d in _EICU_DIMS.items()}
            server  = VFLServer(embed_dim=ed, device=dev, task_types=TASK_TYPES)
            for s in "ABC":
                clients[s].encoder.load_state_dict(ckpt[f"client_{s}"])
            server.model.load_state_dict(ckpt["server"])
            m = _eval_vfl(clients, server, loaders)
            rows.append({"model": model_name, "seed": seed, **m})
            print(f"  {model_name} seed={seed}: IHM={m['ihm_auc_roc']:.4f} "
                  f"RLOS_MAE={m['rlos_mae']:.4f} Pheno={m['pheno_macro_auc']:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Local-only evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_local(ckpt_dir: Path, loaders: dict, device: str = "cpu") -> list[dict]:
    rows = []
    dev  = torch.device(device)
    for site in ["A", "B", "C"]:
        cfg = _SITE_CFG_EICU[site]
        for seed in SEEDS:
            ckpt_path = ckpt_dir / f"best_local_{site}_seed{seed}.pt"
            if not ckpt_path.exists():
                print(f"  [SKIP] {ckpt_path.name}")
                continue
            ckpt       = torch.load(ckpt_path, map_location=device, weights_only=True)
            embed_dim  = ckpt.get("embed_dim",  192)
            hidden_dim = ckpt.get("hidden_dim", 128)
            encoder = SiteEncoder(cfg["input_dim"], hidden_dim, embed_dim=embed_dim).to(dev)
            head    = _LocalHead(embed_dim, cfg["task_type"]).to(dev)
            encoder.load_state_dict(ckpt["encoder"])
            head.load_state_dict(ckpt["head"])
            encoder.eval(); head.eval()

            all_p, all_y = [], []
            for x, mask, y in loaders[site]:
                emb  = encoder(x.to(dev), mask.to(dev))
                pred = head(emb).cpu()
                all_p.append(pred); all_y.append(y)
            p = torch.cat(all_p); y = torch.cat(all_y)

            nan = float("nan")
            row = {"model": f"local_{site}", "seed": seed,
                   "ihm_auc_roc": nan, "ihm_auc_pr": nan,
                   "rlos_mae": nan, "rlos_rmse": nan,
                   "pheno_macro_auc": nan, "pheno_micro_auc": nan}

            if site == "A":
                m = ihm_metrics(y.numpy(), p.squeeze(-1).numpy())
                row.update({"ihm_auc_roc": m["auc_roc"], "ihm_auc_pr": m["auc_pr"]})
            elif site == "B":
                m = rlos_metrics(y.numpy(), p.squeeze(-1).numpy())
                row.update({"rlos_mae": m["mae"], "rlos_rmse": m["rmse"]})
            else:
                m = pheno_metrics(y.numpy(), p.numpy())
                row.update({"pheno_macro_auc": m["macro_auc"],
                            "pheno_micro_auc": m["micro_auc"]})

            rows.append(row)
            print(f"  local_{site} seed={seed}: {m}")
    return rows


# ---------------------------------------------------------------------------
# Centralized evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_centralized(ckpt_dir: Path, root: str, batch_size: int,
                     use_synthetic: bool, device: str = "cpu") -> list[dict]:
    dev  = torch.device(device)
    rows = []

    if use_synthetic:
        def _loader(seed):
            N = 64
            g = torch.Generator(); g.manual_seed(seed + 777)
            ds = TensorDataset(
                torch.randn(N, 1, 13), torch.ones(N, 1),
                torch.randint(0, 2, (N,),    generator=g).float(),
                torch.rand(N,               generator=g) * 10.0,
                torch.randint(0, 2, (N, 25), generator=g).float())
            return DataLoader(ds, batch_size=batch_size)
    else:
        _ds     = CentralizedDatasetEICU(root, "test")
        _shared = DataLoader(_ds, batch_size=batch_size,
                             collate_fn=_collate, shuffle=False)
        _loader = lambda seed: _shared  # noqa: E731

    for seed in SEEDS:
        ckpt_path = ckpt_dir / f"best_centralized_seed{seed}.pt"
        if not ckpt_path.exists():
            print(f"  [SKIP] {ckpt_path.name}")
            continue
        ckpt       = torch.load(ckpt_path, map_location=device, weights_only=True)
        hidden_dim = ckpt.get("hidden_dim", 128)
        encoder = CentralizedEncoder(input_dim=13, hidden_dim=hidden_dim).to(dev)
        mmoe    = MMoEServer(input_dim=_EMBED_DIM, task_types=TASK_TYPES).to(dev)
        encoder.load_state_dict(ckpt["encoder"])
        mmoe.load_state_dict(ckpt["mmoe"])
        encoder.eval(); mmoe.eval()

        ihm_p, ihm_l = [], []
        rlos_p, rlos_l = [], []
        phn_p, phn_l   = [], []
        for x, mask, yi, yd, yp in _loader(seed):
            emb = encoder(x.to(dev), mask.to(dev))
            out = mmoe(emb)
            ihm_p.append(out["ihm"].squeeze(-1).cpu());    ihm_l.append(yi)
            rlos_p.append(out["decomp"].squeeze(-1).cpu()); rlos_l.append(yd)
            phn_p.append(out["pheno"].cpu());              phn_l.append(yp)

        m_ihm  = ihm_metrics(torch.cat(ihm_l).numpy(),   torch.cat(ihm_p).numpy())
        m_rlos = rlos_metrics(torch.cat(rlos_l).numpy(), torch.cat(rlos_p).numpy())
        m_phn  = pheno_metrics(torch.cat(phn_l).numpy(), torch.cat(phn_p).numpy())
        rows.append({"model": "centralized_oracle", "seed": seed,
                     "ihm_auc_roc": m_ihm["auc_roc"], "ihm_auc_pr": m_ihm["auc_pr"],
                     "rlos_mae": m_rlos["mae"], "rlos_rmse": m_rlos["rmse"],
                     "pheno_macro_auc": m_phn["macro_auc"],
                     "pheno_micro_auc": m_phn["micro_auc"]})
        print(f"  centralized seed={seed}: IHM={m_ihm['auc_roc']:.4f} "
              f"RLOS_MAE={m_rlos['mae']:.4f} Pheno={m_phn['macro_auc']:.4f}")
    return rows


_EPS_LEVELS = [float("inf"), 10.0, 5.0, 2.0, 1.0, 0.5]
_EPS_LABEL  = {float("inf"): "inf", 10.0: "10.0", 5.0: "5.0",
               2.0: "2.0", 1.0: "1.0", 0.5: "0.5"}


def eval_dp_privacy_curves(ckpt_dir: Path, loaders: dict,
                           device: str = "cpu") -> list[dict]:
    rows = []
    dev  = torch.device(device)

    def _run(ckpt_path: Path, eps_label: str, mode: str, seed: int) -> None:
        if not ckpt_path.exists():
            print(f"  [SKIP] {ckpt_path.name}")
            return
        ckpt    = torch.load(ckpt_path, map_location=device, weights_only=True)
        ed, hd  = _infer_vfl_dims(ckpt)
        clients = {s: VFLClient(input_dim=d, hidden_dim=hd, embed_dim=ed,
                                lr=1e-3, device=dev)
                   for s, d in _EICU_DIMS.items()}
        server  = VFLServer(embed_dim=ed, device=dev, task_types=TASK_TYPES)
        for s in "ABC":
            clients[s].encoder.load_state_dict(ckpt[f"client_{s}"])
        server.model.load_state_dict(ckpt["server"])
        m = _eval_vfl(clients, server, loaders)
        rows.append({"epsilon_level": eps_label, "mode": mode, "seed": seed, **m})
        print(f"  {mode} ε={eps_label} seed={seed}: "
              f"IHM={m['ihm_auc_roc']:.4f}  "
              f"RLOS_MAE={m['rlos_mae']:.4f}  "
              f"Pheno={m['pheno_macro_auc']:.4f}")

    print("\n── Uniform σ sweep ──")
    for eps in _EPS_LEVELS:
        label = _EPS_LABEL[eps]
        for seed in SEEDS:
            _run(ckpt_dir / f"best_eicu-DP-uniform-eps{label}-seed{seed}_seed{seed}.pt",
                 label, "uniform", seed)

    print("\n── Stratified σ (ε_total=5) ──")
    for seed in SEEDS:
        _run(ckpt_dir / f"best_eicu-DP-stratified-eps5-seed{seed}_seed{seed}.pt",
             "5.0", "stratified", seed)

    return rows


def _print_dp_summary(rows: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    df["epsilon_level"] = pd.to_numeric(df["epsilon_level"], errors="coerce").fillna(float("inf"))
    metrics = ["ihm_auc_roc", "rlos_mae", "pheno_macro_auc"]
    grp    = df.groupby(["mode", "epsilon_level"])
    header = f"{'mode':<12} {'ε':>8}  " + "  ".join(f"{m:>18}" for m in metrics)
    sep    = "─" * len(header)
    print(f"\n{sep}\n{header}\n{sep}")
    for (mode, eps), g in grp:
        vals    = [f"{g[m].dropna().mean():.4f}±{g[m].dropna().std():.4f}"
                   if not g[m].dropna().empty else "—" for m in metrics]
        eps_str = "inf" if eps == float("inf") else str(eps)
        print(f"{mode:<12} {eps_str:>8}  " + "  ".join(f"{v:>18}" for v in vals))
    print(sep)


def _print_summary(rows: list[dict]) -> None:
    import pandas as pd
    df = pd.DataFrame(rows)
    metrics = ["ihm_auc_roc", "rlos_mae", "pheno_macro_auc"]
    header  = f"{'Model':<28}" + "".join(f"{m:>20}" for m in metrics)
    print("\n" + "─" * len(header))
    print(header)
    print("─" * len(header))
    for model, grp in df.groupby("model", sort=False):
        vals = []
        for m in metrics:
            col = grp[m].dropna()
            vals.append(f"{col.mean():.4f}±{col.std():.4f}" if not col.empty else "—")
        print(f"{model:<28}" + "".join(f"{v:>20}" for v in vals))
    print("─" * len(header))


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--root",          default=".")
    p.add_argument("--ckpt_dir",      default="checkpoints/eicu")
    p.add_argument("--batch_size",    type=int, default=64)
    p.add_argument("--device",        default="cpu")
    p.add_argument("--output",        default="results/eicu_test_results.csv")
    p.add_argument("--use_synthetic", action="store_true")
    p.add_argument("--dp",            action="store_true",
                   help="Evaluate DP privacy-curve checkpoints; writes to results/eicu_test_dp.csv")
    args = p.parse_args()

    ckpt_dir     = Path(args.root) / args.ckpt_dir
    site_loaders = (_synthetic_test_loaders(args.batch_size, seed=42)
                    if args.use_synthetic
                    else _real_test_loaders(args.root, args.batch_size))

    print("=" * 60)
    print("FINAL TEST-SET EVALUATION — eICU" + (" (DP)" if args.dp else ""))
    print("=" * 60)

    if args.dp:
        all_rows = eval_dp_privacy_curves(ckpt_dir, site_loaders, args.device)
        if not all_rows:
            print("\n[WARNING] No DP checkpoints found.")
            return
        _print_dp_summary(all_rows)
        output = Path(args.root) / "results/eicu_test_dp.csv"
        fields = ["epsilon_level", "mode", "seed",
                  "ihm_auc_roc", "ihm_auc_pr",
                  "rlos_mae", "rlos_rmse",
                  "pheno_macro_auc", "pheno_micro_auc"]
    else:
        all_rows = []
        print("\n── VFL-MTL & ST variants ──")
        all_rows.extend(eval_vfl_mtl(ckpt_dir, site_loaders, args.device))
        print("\n── Local-only ──")
        all_rows.extend(eval_local(ckpt_dir, site_loaders, args.device))
        print("\n── Centralized oracle ──")
        all_rows.extend(eval_centralized(
            ckpt_dir, args.root, args.batch_size, args.use_synthetic, args.device))
        if not all_rows:
            print("\n[WARNING] No checkpoints found.")
            return
        _print_summary(all_rows)
        output = Path(args.output)
        fields = ["model", "seed", "ihm_auc_roc", "ihm_auc_pr",
                  "rlos_mae", "rlos_rmse", "pheno_macro_auc", "pheno_micro_auc"]

    if args.use_synthetic:
        output = output.parent / f"smoketest_{output.name}"

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nTest results → {output}")


if __name__ == "__main__":
    main()
