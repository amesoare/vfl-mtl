"""
baselines/centralized.py — Centralized oracle baseline (no privacy, no FL).

Single LSTM on all 14 features → single embedding → MMoE multi-task head.
No site partitioning, no client/server split, no embedding exchange.
The MMoE module is reused from model/mmoe.py purely as a multi-task head
(same role as in VFL-MTL, different input source).

Upper bound: gap to VFL-MTL quantifies the privacy/communication cost.

Usage:
    python baselines/centralized.py --use_synthetic --n_epochs 3
    python baselines/centralized.py \
        --root /home/asoare/vfl_mlt --n_epochs 50 \
        --seeds 42 123 7 --output results/centralized.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_prep.dataset import (
    VFLSiteDataset, collate_fn,
    SITE_A_FEATURES, SITE_B_FEATURES, SITE_C_FEATURES, PHENO_LABEL_COLS,
    EICU_SITE_A_FEATURES, EICU_SITE_B_FEATURES, EICU_SITE_C_FEATURES,
    EICU_PHENO_LABEL_COLS,
)
from model.mmoe import MMoEServer
from experiments.metrics import ihm_metrics, decomp_metrics, pheno_metrics, rlos_metrics


_ALL_FEATURES      = SITE_A_FEATURES + SITE_B_FEATURES + SITE_C_FEATURES
_ALL_FEATURES_EICU = EICU_SITE_A_FEATURES + EICU_SITE_B_FEATURES + EICU_SITE_C_FEATURES
assert len(_ALL_FEATURES) == 14
assert len(_ALL_FEATURES_EICU) == 13

_EMBED_DIM = 192  # matches VFL-MTL server input (3 sites × 64); fair capacity comparison



# Encoder


class CentralizedEncoder(nn.Module):
    """Single LSTM on all 14 features → embed_dim."""

    def __init__(self, input_dim: int = 14, hidden_dim: int = 128,
                 num_layers: int = 2, embed_dim: int = 192,
                 dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.proj = nn.Linear(hidden_dim, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        from torch.nn.utils.rnn import pack_padded_sequence
        lengths = mask.sum(1).long().clamp(min=1).cpu()
        packed  = pack_padded_sequence(x, lengths, batch_first=True,
                                       enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.norm(self.proj(h_n[-1]))   # (B, embed_dim)



# Dataset — all 14 features + all three labels


class CentralizedDataset(torch.utils.data.Dataset):
    """Joins all three site VFLSiteDatasets by aligned patient order (stay-level intersection)."""

    def __init__(self, root: str | Path, split: str, max_seq_len: int = 48):
        root_p     = Path(root)
        splits_dir = root_p / "data" / "vertical_splits"
        bench_dir  = root_p / "data" / "mimic3-benchmarks" / "data"
        aligned    = splits_dir / "aligned_patient_ids.csv"

        self._a = VFLSiteDataset(
            site_csv=splits_dir / "site_A_vitals.csv",
            feature_cols=SITE_A_FEATURES, label_col="y_ihm",
            split=split, aligned_ids_csv=aligned,
            timeseries_root=bench_dir / "in-hospital-mortality",
            max_seq_len=max_seq_len, task_type="binary")
        self._b = VFLSiteDataset(
            site_csv=splits_dir / "site_B_labs.csv",
            feature_cols=SITE_B_FEATURES, label_col="y_decomp",
            split=split, aligned_ids_csv=aligned,
            timeseries_root=bench_dir / "decompensation",
            max_seq_len=max_seq_len, task_type="binary")
        self._c = VFLSiteDataset(
            site_csv=splits_dir / "site_C_composite.csv",
            feature_cols=SITE_C_FEATURES, label_col=PHENO_LABEL_COLS,
            split=split, aligned_ids_csv=aligned,
            timeseries_root=bench_dir / "phenotyping",
            max_seq_len=max_seq_len, task_type="multilabel")

        common = sorted(set(self._a.stays) & set(self._b.stays) & set(self._c.stays))
        self._idx_a = {s: i for i, s in enumerate(self._a.stays)}
        self._idx_b = {s: i for i, s in enumerate(self._b.stays)}
        self._idx_c = {s: i for i, s in enumerate(self._c.stays)}
        self._common_stays = common

    def __len__(self): return len(self._common_stays)

    def __getitem__(self, idx):
        stay = self._common_stays[idx]
        x_a, mask, y_ihm    = self._a[self._idx_a[stay]]
        x_b, _,    y_decomp = self._b[self._idx_b[stay]]
        x_c, _,    y_pheno  = self._c[self._idx_c[stay]]
        return torch.cat([x_a, x_b, x_c], dim=-1), mask, y_ihm, y_decomp, y_pheno


def _collate(batch):
    x, mask, yi, yl, yp = zip(*batch)
    return (torch.stack(x), torch.stack(mask), torch.stack(yi),
            torch.stack(yl), torch.stack(yp))


class CentralizedDatasetEICU(torch.utils.data.Dataset):
    """Tabular join of three eICU site CSVs on patientunitstayid."""

    def __init__(self, root: str | Path, split: str):
        splits_dir = Path(root) / "data" / "eicu_vertical_splits"
        aligned    = pd.read_csv(splits_dir / "aligned_patient_ids_eicu.csv")
        pids       = set(aligned[aligned["split"] == split]["patientunitstayid"])

        def _filt(fname):
            df = pd.read_csv(splits_dir / fname)
            return df[df["patientunitstayid"].isin(pids) & (df["split"] == split)]

        a = _filt("site_A_eicu.csv")
        b = _filt("site_B_eicu.csv")
        c = _filt("site_C_eicu.csv")

        m = (a[["patientunitstayid", "y_ihm"] + EICU_SITE_A_FEATURES]
               .merge(b[["patientunitstayid", "y_rlos"] + EICU_SITE_B_FEATURES],
                      on="patientunitstayid")
               .merge(c[["patientunitstayid"] + EICU_SITE_C_FEATURES + EICU_PHENO_LABEL_COLS],
                      on="patientunitstayid"))

        self._x      = m[_ALL_FEATURES_EICU].values.astype(np.float32)
        self._y_ihm  = m["y_ihm"].values.astype(np.float32)
        self._y_rlos = m["y_rlos"].values.astype(np.float32)
        self._y_pheno = m[EICU_PHENO_LABEL_COLS].values.astype(np.float32)

    def __len__(self):
        return len(self._x)

    def __getitem__(self, idx):
        x    = torch.from_numpy(self._x[idx:idx+1])  # (1, 13) — tabular single timestep
        mask = torch.ones(1)
        return (x, mask,
                torch.tensor(self._y_ihm[idx]),
                torch.tensor(self._y_rlos[idx]),
                torch.from_numpy(self._y_pheno[idx]))



# Synthetic loaders


def _synthetic_loaders(batch_size: int, seed: int, dataset: str = "mimic") -> dict:
    g = torch.Generator(); g.manual_seed(seed)
    n_feat = 13 if dataset == "eicu" else 14
    T      = 1  if dataset == "eicu" else 48
    def _make(n):
        x   = torch.randn(n, T, n_feat)
        m   = torch.ones(n, T)
        yi  = torch.randint(0, 2, (n,),    generator=g).float()
        yd  = (torch.rand(n, generator=g) * 10.0 if dataset == "eicu"
               else torch.randint(0, 2, (n,), generator=g).float())
        yp  = torch.randint(0, 2, (n, 25), generator=g).float()
        return DataLoader(TensorDataset(x, m, yi, yd, yp), batch_size=batch_size)
    return {"train": _make(256), "val": _make(64)}



# Training


def _weighted_bce(pred: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    w = pos_weight * target + (1.0 - target)
    return nn.functional.binary_cross_entropy(pred, target, weight=w)


def train_centralized(root: str, n_epochs: int, lr: float, batch_size: int,
                      seed: int, use_synthetic: bool, num_workers: int = 0,
                      hidden_dim: int = 128, prebuilt_loaders: dict = None,
                      patience: int = 10, ckpt_dir: str | None = None,
                      decomp_pos_weight: float = 0.0,
                      dataset: str = "mimic") -> list[dict]:
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    _eicu = (dataset == "eicu")

    # Compute pos_weight for MIMIC decompensation (skip for eICU regression)
    _decomp_pw = 1.0
    if not _eicu:
        if use_synthetic or decomp_pos_weight > 0.0:
            _decomp_pw = 1.0 if use_synthetic else decomp_pos_weight
        else:
            splits_dir = Path(root) / "data" / "vertical_splits"
            _b = pd.read_csv(splits_dir / "site_B_labs.csv", usecols=["y_decomp", "split"])
            pos_rate   = float(_b[_b["split"] == "train"]["y_decomp"].mean())
            _decomp_pw = (1.0 - pos_rate) / pos_rate
            print(f"[centralized] decomp pos_weight={_decomp_pw:.1f}  (pos_rate={pos_rate:.3%})")

    n_feat   = 13 if _eicu else 14
    encoder  = CentralizedEncoder(input_dim=n_feat, hidden_dim=hidden_dim).to(device)
    mmoe     = MMoEServer(input_dim=_EMBED_DIM,
                          task_types={"decomp": "regression"} if _eicu else None).to(device)
    opt      = torch.optim.Adam(
        list(encoder.parameters()) + list(mmoe.parameters()), lr=lr)
    ihm_fn   = nn.BCELoss()
    pheno_fn = nn.BCELoss()

    if prebuilt_loaders is not None:
        loaders = prebuilt_loaders
    elif use_synthetic:
        loaders = _synthetic_loaders(batch_size, seed, dataset)
    else:
        _ds_cls = CentralizedDatasetEICU if _eicu else CentralizedDataset
        loaders = {
            split: DataLoader(_ds_cls(root, split),
                              batch_size=batch_size, shuffle=(split == "train"),
                              collate_fn=_collate, num_workers=num_workers,
                              pin_memory=True)
            for split in ("train", "val")}

    # early stopping state — monitor mean AUC across all three tasks
    best_score, no_improve = -1.0, 0

    rows = []
    for epoch in range(1, n_epochs + 1):
        encoder.train(); mmoe.train()
        t0 = time.perf_counter(); total_loss = 0.0; nb = 0

        for x, mask, y_ihm, y_decomp, y_pheno in loaders["train"]:
            x        = x.to(device);        mask     = mask.to(device)
            y_ihm    = y_ihm.to(device);    y_decomp = y_decomp.to(device)
            y_pheno  = y_pheno.to(device)

            emb  = encoder(x, mask)
            out  = mmoe(emb)
            if _eicu:
                decomp_loss = torch.nn.functional.mse_loss(
                    out["decomp"].squeeze(-1), y_decomp.float())
            else:
                decomp_loss = _weighted_bce(
                    out["decomp"].squeeze(-1), y_decomp.float(), _decomp_pw)
            loss = ihm_fn(out["ihm"].squeeze(-1), y_ihm) + decomp_loss + pheno_fn(out["pheno"], y_pheno)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1

        encoder.eval(); mmoe.eval()
        ihm_p,    ihm_l    = [], []
        decomp_p, decomp_l = [], []
        pheno_p,  pheno_l  = [], []
        with torch.no_grad():
            for x, mask, y_ihm, y_decomp, y_pheno in loaders["val"]:
                emb = encoder(x.to(device), mask.to(device))
                out = mmoe(emb)
                ihm_p.append(out["ihm"].squeeze(-1).cpu());    ihm_l.append(y_ihm)
                decomp_p.append(out["decomp"].squeeze(-1).cpu()); decomp_l.append(y_decomp)
                pheno_p.append(out["pheno"].cpu());            pheno_l.append(y_pheno)

        m_ihm   = ihm_metrics(torch.cat(ihm_l).numpy(), torch.cat(ihm_p).numpy())
        m_pheno = pheno_metrics(torch.cat(pheno_l).numpy(), torch.cat(pheno_p).numpy())
        if _eicu:
            m_decomp = rlos_metrics(torch.cat(decomp_l).numpy(), torch.cat(decomp_p).numpy())
            score    = (m_ihm["auc_roc"] + m_pheno["macro_auc"]) / 2
        else:
            m_decomp = decomp_metrics(torch.cat(decomp_l).numpy(), torch.cat(decomp_p).numpy())
            score    = (m_ihm["auc_roc"] + m_decomp["auc_roc"] + m_pheno["macro_auc"]) / 3
        if score > best_score:
            best_score = score
            no_improve = 0
            if ckpt_dir is not None:
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                ckpt_path = Path(ckpt_dir) / f"best_centralized_seed{seed}.pt"
                torch.save(
                    {"encoder": encoder.state_dict(),
                     "mmoe":    mmoe.state_dict(),
                     "seed": seed, "hidden_dim": hidden_dim},
                    ckpt_path,
                )
                print(f"  [ckpt] saved → {ckpt_path}  (score={score:.4f})")
        else:
            no_improve += 1

        decomp_prefix = "val_rlos_" if _eicu else "val_decomp_"
        rows.append({"model": "centralized_oracle", "epoch": epoch,
                     "train_loss": total_loss / max(nb, 1),
                     "elapsed_s": time.perf_counter() - t0, "seed": seed,
                     **{f"val_ihm_{k}":       v for k, v in m_ihm.items()},
                     **{f"{decomp_prefix}{k}": v for k, v in m_decomp.items()},
                     **{f"val_pheno_{k}":     v for k, v in m_pheno.items()}})

        if no_improve >= patience:
            print(f"  Early stop at epoch {epoch} (best mean AUC={best_score:.4f})")
            break

    return rows



# Entry point


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root",          default=".")
    p.add_argument("--n_epochs",      type=int,   default=50)
    p.add_argument("--lr",            type=float, default=1e-3)
    p.add_argument("--batch_size",    type=int,   default=64)
    p.add_argument("--seeds",         type=int,   nargs="+", default=[42, 123, 7])
    p.add_argument("--num_workers",   type=int,   default=0)
    p.add_argument("--output",        default=None)
    p.add_argument("--use_synthetic", action="store_true")
    p.add_argument("--patience",      type=int,   default=10)
    p.add_argument("--ckpt_dir",      default="checkpoints")
    p.add_argument("--dataset",       default="mimic", choices=["mimic", "eicu"])
    args = p.parse_args()
    _prefix = "eicu_" if args.dataset == "eicu" else ""
    if args.output is None:
        args.output = f"results/{_prefix}centralized.csv"
    if args.use_synthetic:
        _p = Path(args.output); args.output = str(_p.parent / f"smoketest_{_p.name}")

    _ds_cls = CentralizedDatasetEICU if args.dataset == "eicu" else CentralizedDataset
    prebuilt = (None if args.use_synthetic else {
        split: DataLoader(_ds_cls(args.root, split),
                          batch_size=args.batch_size, shuffle=(split == "train"),
                          collate_fn=_collate, num_workers=args.num_workers,
                          pin_memory=True)
        for split in ("train", "val")})

    all_rows = []
    for seed in args.seeds:
        rows = train_centralized(args.root, args.n_epochs, args.lr,
                                 args.batch_size, seed, args.use_synthetic,
                                 args.num_workers, prebuilt_loaders=prebuilt,
                                 patience=args.patience, ckpt_dir=args.ckpt_dir,
                                 dataset=args.dataset)
        all_rows.extend(rows)
        print(f"seed={seed}: " + str({k: round(v, 4) for k, v in rows[-1].items()
                                      if k.startswith("val_")}))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader(); w.writerows(all_rows)
    print(f"Saved {len(all_rows)} rows → {args.output}")


if __name__ == "__main__":
    main()
