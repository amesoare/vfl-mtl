"""
train.py — Round-based VFL-MTL training loop.

Usage
-----
  python train.py --root . --rounds 50 --seed 42
  python train.py --root . --rounds 50 --device cuda --fedprox-mu 0.01 --seed 42
  python train.py --root . --rounds 50 --resume checkpoints/ckpt_round0020_seed42.pt

Seeds reported in the paper: 42, 123, 7
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader, TensorDataset

from data_prep.dataset import build_site_loaders
from fl.client import VFLClient
from fl.fedavg import fedavg_aggregate
from fl.fedprox import fedprox_penalty
from fl.server import VFLServer


@dataclass
class TrainConfig:
    """All hyperparameters and flags for a single VFL-MTL training run."""
    splits_dir:         str   = "data/vertical_splits"
    n_rounds:           int   = 50
    batch_size:         int   = 64
    lr:                 float = 1e-3
    seed:               int   = 42
    device:             str   = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    hidden_dim:         int   = 128
    embed_dim:          int   = 64
    num_experts:        int   = 4
    task_weights:       dict  = field(default_factory=lambda: {"ihm": 1.0, "decomp": 1.0, "pheno": 1.0})
    site_input_dims:    dict  = field(default_factory=lambda: {"A": 7, "B": 4, "C": 3})
    n_sites:            int   = 3
    use_fedavg:         bool  = True
    fedavg_every:       int   = 5
    fedprox_mu:         float = 0.0
    use_synthetic:      bool  = False
    n_synthetic:        int   = 256
    num_workers:        int   = 0
    max_seq_len:        int   = 48
    eval_every:         int   = 1
    model_name:         str   = "VFL-MTL"   # used for checkpoint filename
    ckpt_dir:           str   = "checkpoints"
    # 0.0 = auto-compute from site_B_labs.csv train split at run start.
    # Set explicitly only for synthetic runs (use 1.0) or to override.
    decomp_pos_weight:  float = 0.0
    # Rounds without mean-AUROC improvement before stopping (0 = disabled).
    patience:           int   = 15
    use_mmoe:           bool  = True   # False = shared-bottom MLP (Abl 1)
    uniform_gating:     bool  = False  # True  = fixed equal expert weights (Abl 4)
    uncertainty_weighting: bool = False  # Kendall et al. (2018) homoscedastic loss weighting
    # 0 = disabled. When > 0, compute per-task gradient cosine similarity (Yu et al. 2020)
    # at the first batch of every N-th round and log to the results CSV.
    grad_sim_every:     int   = 0
    # When set, shuffle Sites B and C patient ordering (breaks PSI alignment). Abl 2.
    random_align_seed:  int | None = None
    # Dataset selection and per-task type overrides.
    # dataset='eicu' switches data paths; task_types overrides any head's loss/metric.
    # Example for eICU: task_types={"decomp": "regression"}
    dataset:            str        = "mimic"
    task_types:         dict | None = None
    # Paper 2 — Differential privacy configuration.
    # None = no DP (default; Paper 1 behaviour unchanged).
    # Uniform:     {'mode': 'uniform',     'sigma': 1.0, 'max_grad_norm': 1.0, 'delta': 1e-5}
    # Stratified:  {'mode': 'stratified',  'sigma_ihm': 0.5, 'sigma_decomp': 1.0,
    #               'sigma_pheno': 1.5, 'max_grad_norm': 1.0, 'delta': 1e-5}
    privacy_config:     dict | None = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_synthetic_loaders(
    batch_size: int,
    seq_len: int,
    n_batches: int = 10,
    task_types: dict | None = None,
) -> dict[str, DataLoader]:
    """Random tensors matching build_site_loaders() shapes. Use --use-synthetic for smoke tests."""
    _tt = task_types or {}
    N   = batch_size * n_batches
    T   = seq_len

    def _loader(n_feat: int, y: torch.Tensor) -> DataLoader:
        x    = torch.randn(N, T, n_feat)
        mask = torch.ones(N, T)
        ds   = TensorDataset(x, mask, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    n_feat_B = 3 if _tt.get("decomp") == "regression" else 4
    y_B      = (torch.rand(N) * 10.0 if _tt.get("decomp") == "regression"
                else torch.randint(0, 2, (N,)).float())

    loaders = {
        "A": _loader(7,       torch.randint(0, 2, (N,)).float()),
        "B": _loader(n_feat_B, y_B),
        "C": _loader(3,       torch.randint(0, 2, (N, 25)).float()),
    }
    return loaders



def save_checkpoint(
    path: Path,
    round_idx: int,
    clients: dict[str, VFLClient],
    server: VFLServer,
) -> None:
    ckpt = {"round": round_idx, "server": server.model.state_dict(),
            "server_opt": server.optimizer.state_dict()}
    for site, client in clients.items():
        ckpt[f"client_{site}"] = client.encoder.state_dict()
    torch.save(
        ckpt,
        path,
    )


def load_checkpoint(
    path: Path,
    clients: dict[str, VFLClient],
    server: VFLServer,
) -> int:
    """Load checkpoint and return the round it was saved at."""
    ckpt = torch.load(path, weights_only=True)
    clients["A"].encoder.load_state_dict(ckpt["client_A"])
    clients["B"].encoder.load_state_dict(ckpt["client_B"])
    clients["C"].encoder.load_state_dict(ckpt["client_C"])
    server.model.load_state_dict(ckpt["server"])
    server.optimizer.load_state_dict(ckpt["server_opt"])
    return int(ckpt["round"])


# ---------------------------------------------------------------------------
# Programmatic training entry point (used by experiment scripts)
# ---------------------------------------------------------------------------

def run_training(
    cfg: TrainConfig,
    prebuilt_loaders: dict | None = None,
) -> list[dict]:
    """
    Run a full VFL-MTL training loop from a TrainConfig and return per-round metrics.

    Parameters
    ----------
    prebuilt_loaders : optional dict with keys 'train' and 'val', each a
        dict[site → DataLoader].  When provided, data loading is skipped —
        the experiment script pre-builds loaders once and passes them here,
        avoiding repeated GPFS reads across (config, seed) pairs.

    Returns
    -------
    list of dicts, one per round, each containing:
        round, train_loss, ihm_loss, decomp_loss, pheno_loss, elapsed_s
        + val metrics (val_ihm_auroc, val_ihm_auprc, val_decomp_auroc,
          val_decomp_auprc, val_pheno_macro_auroc) on eval rounds
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    print(f"[train] device={device}  seed={cfg.seed}  rounds={cfg.n_rounds}")
    if device.type == "cuda":
        print(f"[train] GPU: {torch.cuda.get_device_name(device)}")

    # ---- Data ----
    _all_dims    = {"A": 7, "B": 3 if cfg.dataset == "eicu" else 4, "C": 3}
    # cfg.site_input_dims defaults to MIMIC values; cap at actual feature count so
    # eICU (B=3) is never overridden by the MIMIC default (B=4).
    _site_dims   = {s: min(cfg.site_input_dims.get(s, _all_dims[s]), _all_dims[s])
                    for s in _all_dims}
    active_sites = list(_all_dims.keys())[:cfg.n_sites]
    _tt          = cfg.task_types or {}

    if prebuilt_loaders is not None:
        train_loaders     = {s: prebuilt_loaders["train"][s] for s in active_sites}
        val_loaders       = {s: prebuilt_loaders["val"][s]   for s in active_sites}
        decomp_pos_weight = prebuilt_loaders.get("decomp_pos_weight", 1.0)
    elif cfg.use_synthetic:
        decomp_pos_weight = 1.0
        n_batches = max(1, cfg.n_synthetic // cfg.batch_size)
        train_loaders = make_synthetic_loaders(cfg.batch_size, cfg.max_seq_len, n_batches, _tt)
        val_loaders   = make_synthetic_loaders(cfg.batch_size, cfg.max_seq_len, max(1, n_batches // 4), _tt)
        train_loaders = {s: train_loaders[s] for s in active_sites}
        val_loaders   = {s: val_loaders[s]   for s in active_sites}
    else:
        if _tt.get("decomp") == "regression":
            decomp_pos_weight = 1.0  # MSE loss; pos_weight unused
        elif cfg.decomp_pos_weight > 0.0:
            decomp_pos_weight = cfg.decomp_pos_weight
        else:
            site_b_csv = Path(cfg.splits_dir) / "site_B_labs.csv"
            _b = pd.read_csv(site_b_csv, usecols=["y_decomp", "split"])
            pos_rate = float(_b[_b["split"] == "train"]["y_decomp"].mean())
            assert 0.0 < pos_rate < 1.0, f"Degenerate decomp positive rate: {pos_rate}"
            decomp_pos_weight = (1.0 - pos_rate) / pos_rate
            print(f"[train] Decomp pos_weight auto-computed: {decomp_pos_weight:.1f}  "
                  f"(pos_rate={pos_rate:.3%})")
            if pos_rate < 0.005 or pos_rate > 0.5:
                print(f"[train] WARNING: decomp pos_rate={pos_rate:.3%} is outside expected "
                      f"5–50% range — check site_B_labs.csv alignment.")

        project_root = Path(cfg.splits_dir).parents[1]
        train_loaders = build_site_loaders(
            project_root, "train", cfg.batch_size, cfg.num_workers, cfg.max_seq_len,
            dataset=cfg.dataset,
        )
        val_loaders = build_site_loaders(
            project_root, "val", cfg.batch_size, cfg.num_workers, cfg.max_seq_len,
            dataset=cfg.dataset,
        )
        train_loaders = {s: train_loaders[s] for s in active_sites}
        val_loaders   = {s: val_loaders[s]   for s in active_sites}

    # ---- Differential Privacy setup (Paper 2) ----
    _dp_enabled = cfg.privacy_config is not None
    _dp_mechanism = None
    _dp_accountant = None
    _dp_delta = 1e-5
    _dp_sample_rate = 1.0
    _dp_sigma_map: dict[str, float] = {}

    if _dp_enabled:
        from privacy.adaptive_dpsgd import AdaptiveDPSGD, DPVFLClient  # noqa: F401
        from privacy.renyi_accountant import RenyiAccountant

        pc = cfg.privacy_config
        _dp_delta = float(pc.get("delta", 1e-5))
        _max_grad_norm = float(pc.get("max_grad_norm", 1.0))
        _dp_mechanism = AdaptiveDPSGD(max_grad_norm=_max_grad_norm)

        if pc["mode"] == "uniform":
            _sigma = float(pc["sigma"])
            _dp_mechanism.set_uniform(_sigma)
            _dp_sigma_map = {t: _sigma for t in ("ihm", "decomp", "pheno")}
        elif pc["mode"] == "stratified":
            _dp_mechanism.set_stratified(
                sigma_ihm=float(pc["sigma_ihm"]),
                sigma_decomp=float(pc["sigma_decomp"]),
                sigma_pheno=float(pc["sigma_pheno"]),
            )
            _dp_sigma_map = {
                "ihm":    float(pc["sigma_ihm"]),
                "decomp": float(pc["sigma_decomp"]),
                "pheno":  float(pc["sigma_pheno"]),
            }
        else:
            raise ValueError(f"Unknown privacy mode: {pc['mode']!r}. Expected 'uniform' or 'stratified'.")

        _dp_accountant = RenyiAccountant()
        # sample_rate = batch_size / N_train ≈ 1 / n_batches_per_loader
        _n_train_batches = len(train_loaders[active_sites[0]])
        _dp_sample_rate = 1.0 / max(_n_train_batches, 1)
        print(
            f"[train] DP enabled: mode={pc['mode']}  "
            f"max_grad_norm={_max_grad_norm}  delta={_dp_delta}  "
            f"sample_rate={_dp_sample_rate:.5f}  sigma={_dp_sigma_map}"
        )

    # ---- Models ----
    clients: dict[str, VFLClient] = {}
    for s in active_sites:
        _client_kwargs = dict(
            input_dim=_site_dims[s],
            hidden_dim=cfg.hidden_dim,
            embed_dim=cfg.embed_dim,
            lr=cfg.lr,
            device=device,
        )
        if _dp_enabled:
            from privacy.adaptive_dpsgd import DPVFLClient
            clients[s] = DPVFLClient(dp_mechanism=_dp_mechanism, site=s, **_client_kwargs)
        else:
            clients[s] = VFLClient(**_client_kwargs)
    server = VFLServer(
        embed_dim=cfg.embed_dim,
        num_experts=cfg.num_experts,
        lr=cfg.lr,
        device=device,
        task_weights=cfg.task_weights,
        n_sites=cfg.n_sites,
        decomp_pos_weight=decomp_pos_weight,
        use_mmoe=cfg.use_mmoe,
        uniform_gating=cfg.uniform_gating,
        uncertainty_weighting=cfg.uncertainty_weighting,
        task_types=cfg.task_types,
    )

    ckpt_dir = Path(cfg.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_ihm = -1.0
    # Early stopping: track mean AUROC over tasks with nonzero weight.
    _active_auroc_keys = (
        (["ihm_auroc"]         if cfg.task_weights.get("ihm",   0) > 0 else []) +
        (["decomp_auroc"]      if cfg.task_weights.get("decomp",0) > 0
                                  and _tt.get("decomp", "binary") == "binary" else []) +
        (["pheno_macro_auroc"] if cfg.task_weights.get("pheno", 0) > 0 else [])
    )
    best_mean_auroc = -1.0
    no_improve      = 0
    results = []

    for rnd in range(cfg.n_rounds):
        t_round = time.time()

        # Snapshot for FedProx
        global_encoder_params: dict[str, dict] | None = None
        if cfg.fedprox_mu > 0.0:
            global_encoder_params = {s: c.get_encoder_params() for s, c in clients.items()}

        # ---- Train one round ----
        round_train = {s: train_loaders[s] for s in active_sites}
        _compute_grad_sim = (
            cfg.grad_sim_every > 0 and rnd % cfg.grad_sim_every == 0
        )
        train_losses = _train_one_round_sites(
            clients, server, round_train, active_sites,
            _site_dims, _all_dims,
            cfg.fedprox_mu, global_encoder_params,
            compute_grad_sim=_compute_grad_sim,
            task_types=_tt,
        )

        # ---- FedAvg ----
        # Only aggregate encoders with matching architectures (same input_dim).
        # In our heterogeneous VFL setup each site has a unique input_dim, so
        # FedAvg across sites is skipped — aggregation would require identical shapes.
        if cfg.use_fedavg and (rnd + 1) % cfg.fedavg_every == 0:
            dims = [_site_dims[s] for s in active_sites]
            if len(set(dims)) == 1:
                n_batches_per_site = len(list(train_loaders.values())[0])
                global_params = fedavg_aggregate(
                    [clients[s].get_encoder_params() for s in active_sites],
                    weights=[n_batches_per_site] * len(active_sites),
                )
                for s in active_sites:
                    clients[s].set_encoder_params(global_params)

        # ---- Validate ----
        val_metrics: dict[str, float] = {}
        if (rnd + 1) % cfg.eval_every == 0:
            round_val = {s: val_loaders[s] for s in active_sites}
            val_metrics = _evaluate_sites(clients, server, round_val, active_sites,
                                          _site_dims, _all_dims, _tt)

        elapsed = time.time() - t_round
        row = {
            "round":        rnd + 1,
            "train_loss":   train_losses["total_loss"],
            "ihm_loss":     train_losses["ihm_loss"],
            "decomp_loss":  train_losses["decomp_loss"],
            "pheno_loss":   train_losses["pheno_loss"],
            "elapsed_s":    round(elapsed, 1),
        }
        for _gs_key in ("grad_sim_ihm_decomp", "grad_sim_ihm_pheno", "grad_sim_decomp_pheno"):
            if _gs_key in train_losses:
                row[_gs_key] = train_losses[_gs_key]
        # Step DP accountant and log per-task ε (Paper 2).
        if _dp_enabled and _dp_accountant is not None:
            _gs_payload = {
                k: train_losses[k]
                for k in ("grad_sim_ihm_decomp", "grad_sim_ihm_pheno", "grad_sim_decomp_pheno")
                if k in train_losses
            }
            if _gs_payload:
                _dp_accountant.log_grad_sim(_gs_payload)
            _n_batches_this_round = len(round_train[active_sites[0]])
            if cfg.privacy_config["mode"] == "uniform":
                _dp_accountant.step(
                    noise_multiplier=_dp_sigma_map.get("ihm", 0.0),
                    sample_rate=_dp_sample_rate,
                    num_steps=_n_batches_this_round,
                )
            else:
                _dp_accountant.step_stratified(
                    sigma_map=_dp_sigma_map,
                    sample_rate=_dp_sample_rate,
                    num_steps=_n_batches_this_round,
                )
            _eps = _dp_accountant.get_epsilon(delta=_dp_delta)
            row["epsilon_ihm"]    = _eps.get("ihm",    float("nan"))
            row["epsilon_decomp"] = _eps.get("decomp", float("nan"))
            row["epsilon_pheno"]  = _eps.get("pheno",  float("nan"))
        # Log learned σ values when uncertainty weighting is active.
        if cfg.uncertainty_weighting and server.log_vars is not None:
            for t, lv in server.log_vars.items():
                row[f"sigma_{t}"] = float(torch.exp(0.5 * lv).item())
        if val_metrics:
            row["val_ihm_auroc"]         = val_metrics.get("ihm_auroc",         float("nan"))
            row["val_ihm_auprc"]         = val_metrics.get("ihm_auprc",         float("nan"))
            row["val_decomp_auroc"]      = val_metrics.get("decomp_auroc",      float("nan"))
            row["val_decomp_auprc"]      = val_metrics.get("decomp_auprc",      float("nan"))
            row["val_rlos_mae"]          = val_metrics.get("rlos_mae",           float("nan"))
            row["val_rlos_rmse"]         = val_metrics.get("rlos_rmse",          float("nan"))
            row["val_pheno_macro_auroc"] = val_metrics.get("pheno_macro_auroc", float("nan"))
            # Save best checkpoint (keyed by model_name so each config gets its own file)
            score = val_metrics.get("ihm_auroc", -1.0)
            if score > best_val_ihm:
                best_val_ihm = score
                save_checkpoint(
                    ckpt_dir / f"best_{cfg.model_name}_seed{cfg.seed}.pt",
                    rnd + 1, clients, server,
                )
            # Early stopping: mean AUROC across active tasks.
            # When uncertainty_weighting is on, weight each task's AUROC by its
            # learned Kendall precision exp(-s_i) so that high-σ tasks (decomp)
            # count less in checkpoint selection — consistent with the training objective.
            if cfg.patience > 0 and _active_auroc_keys:
                present = [val_metrics[k] for k in _active_auroc_keys if k in val_metrics]
                if present:
                    if cfg.uncertainty_weighting and server.log_vars is not None:
                        _task_map = {"ihm_auroc": "ihm", "decomp_auroc": "decomp",
                                     "pheno_macro_auroc": "pheno"}
                        _prec = {t: float(torch.exp(-server.log_vars[t]).item())
                                 for t in server.log_vars}
                        _w = [_prec.get(_task_map.get(k, k), 1.0)
                              for k in _active_auroc_keys if k in val_metrics]
                        _total = sum(_w) or 1.0
                        mean_auroc = float(sum(
                            w * v for w, v in zip(_w, present)
                        ) / _total)
                    else:
                        mean_auroc = float(np.mean(present))
                    if mean_auroc > best_mean_auroc:
                        best_mean_auroc = mean_auroc
                        no_improve = 0
                    else:
                        no_improve += 1
                    if no_improve >= cfg.patience:
                        results.append(row)
                        print(f"[train] Early stop at round {rnd+1} "
                              f"(best mean AUROC={best_mean_auroc:.4f}, "
                              f"no improvement for {cfg.patience} rounds)")
                        return results

        results.append(row)

    return results


def _train_one_round_sites(
    clients, server, loaders, active_sites,
    site_input_dims, all_dims,
    fedprox_mu, global_encoder_params,
    compute_grad_sim: bool = False,
    task_types: dict | None = None,
) -> dict[str, float]:
    """One training round over active_sites with feature truncation and optional FedProx/grad-sim."""
    task_loss_sums = {"ihm": 0.0, "decomp": 0.0, "pheno": 0.0}
    total_loss_sum = 0.0
    n_batches = 0
    _grad_sim: dict[str, float] = {}

    loader_iters = [loaders[s] for s in active_sites]

    for batches in zip(*loader_iters):
        embeddings = {}
        labels = {}
        for site, (x, mask, y) in zip(active_sites, batches):
            dim = site_input_dims.get(site, all_dims[site])
            x = x[..., :dim]
            embeddings[site] = clients[site].forward(x, mask)
            if site == "A":
                labels["ihm"] = y
            elif site == "B":
                labels["decomp"] = y
            elif site == "C":
                labels["pheno"] = y

        # Pad missing tasks with zero labels if n_sites < 3
        if "decomp" not in labels:
            labels["decomp"] = torch.zeros(list(embeddings.values())[0].shape[0], dtype=torch.float)
        if "pheno" not in labels:
            labels["pheno"] = torch.zeros(list(embeddings.values())[0].shape[0], 25)

        server.aggregate_embeddings(embeddings)
        total_loss, task_losses = server.forward_and_loss(labels)

        # ---- Fail-fast checks on first batch ----
        if n_batches == 0:
            for t, loss in task_losses.items():
                v = loss.item()
                assert not (v != v), f"NaN loss at step 0 for task '{t}'"
                assert v != float("inf"), f"Inf loss at step 0 for task '{t}'"
                assert v > 0.0, f"Zero loss at step 0 for task '{t}' — constant predictor?"
            if "decomp" in labels and (task_types or {}).get("decomp", "binary") == "binary":
                y_d = labels["decomp"]
                assert set(y_d.unique().tolist()).issubset({0.0, 1.0}), \
                    f"Decomp labels contain values outside {{0,1}}: {y_d.unique().tolist()}"
            for site, emb in embeddings.items():
                assert not torch.isnan(emb).any(), f"NaN embedding from site {site} at step 0"
            print(f"[train] Step-0 losses: " +
                  " ".join(f"{t}={v.item():.4f}" for t, v in task_losses.items()))

        server.backward_and_step(total_loss)

        if compute_grad_sim and n_batches == 0:
            _grad_sim = server.compute_task_gradient_similarity(labels)

        grads = server.get_embedding_gradients()
        for site in active_sites:
            clients[site].receive_gradient(grads[site])

        if fedprox_mu > 0.0 and global_encoder_params is not None:
            for site in active_sites:
                penalty = fedprox_penalty(clients[site].encoder, global_encoder_params[site], mu=fedprox_mu)
                clients[site].optimizer.zero_grad()
                penalty.backward()
                clients[site].optimizer.step()

        for t, loss in task_losses.items():
            task_loss_sums[t] += loss.item()
        total_loss_sum += total_loss.item()
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Training loaders returned no batches.")

    return {
        "total_loss":  total_loss_sum / n_batches,
        "ihm_loss":    task_loss_sums["ihm"]    / n_batches,
        "decomp_loss": task_loss_sums["decomp"] / n_batches,
        "pheno_loss":  task_loss_sums["pheno"]  / n_batches,
        **_grad_sim,
    }


@torch.no_grad()
def _evaluate_sites(clients, server, loaders, active_sites, site_input_dims, all_dims,
                    task_types: dict | None = None) -> dict[str, float]:
    """Evaluation over active_sites with feature truncation; handles binary and regression tasks."""
    all_preds:  dict[str, list] = {"ihm": [], "decomp": [], "pheno": []}
    all_labels: dict[str, list] = {"ihm": [], "decomp": [], "pheno": []}

    loader_iters = [loaders[s] for s in active_sites]

    for batches in zip(*loader_iters):
        embeddings = {}
        for site, (x, mask, y) in zip(active_sites, batches):
            dim = site_input_dims.get(site, all_dims[site])
            x = x[..., :dim]
            embeddings[site] = clients[site].eval_forward(x, mask)
            if site == "A":
                all_labels["ihm"].append(y.numpy())
            elif site == "B":
                all_labels["decomp"].append(y.numpy())
            elif site == "C":
                all_labels["pheno"].append(y.numpy())

        preds = server.predict(embeddings)
        all_preds["ihm"].append(preds["ihm"].squeeze(-1).cpu().numpy())
        all_preds["decomp"].append(preds["decomp"].squeeze(-1).cpu().numpy())
        all_preds["pheno"].append(preds["pheno"].cpu().numpy())

    # Fall back gracefully if a task has no labels (n_sites < 3)
    metrics: dict[str, float] = {}
    if all_labels["ihm"]:
        p = np.concatenate(all_preds["ihm"])
        y = np.concatenate(all_labels["ihm"])
        metrics["ihm_auroc"] = float(roc_auc_score(y, p))
        metrics["ihm_auprc"] = float(average_precision_score(y, p))
    if all_labels["decomp"]:
        p  = np.concatenate(all_preds["decomp"])
        y  = np.concatenate(all_labels["decomp"])
        tt = (task_types or {}).get("decomp", "binary")
        if tt == "regression":
            metrics["rlos_mae"]  = float(np.mean(np.abs(p - y)))
            metrics["rlos_rmse"] = float(np.sqrt(np.mean((p - y) ** 2)))
        else:
            metrics["decomp_auroc"] = float(roc_auc_score(y, p))
            metrics["decomp_auprc"] = float(average_precision_score(y, p))
    if all_labels["pheno"]:
        p = np.concatenate(all_preds["pheno"])
        y = np.concatenate(all_labels["pheno"])
        per_label = [roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1]) if y[:, i].sum() > 0]
        metrics["pheno_macro_auroc"] = float(np.mean(per_label)) if per_label else float("nan")

    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VFL-MTL training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--root",        default=".",  help="Project root directory")
    p.add_argument("--dataset",     default="mimic", choices=["mimic", "eicu"],
                   help="Dataset to train on; 'eicu' uses tabular vertical splits")
    p.add_argument("--max-seq-len", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")

    # Training
    p.add_argument("--rounds",     type=int,   default=50,   help="Total training rounds")
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3, help="Adam lr for all components")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="cuda" if torch.cuda.is_available() else "cpu")

    # Architecture
    p.add_argument("--hidden-dim",  type=int, default=128, help="LSTM hidden size")
    p.add_argument("--embed-dim",   type=int, default=64,  help="Cut-layer embedding size")
    p.add_argument("--num-experts", type=int, default=4,   help="MMoE shared expert count")

    # Loss weights
    p.add_argument("--w-ihm",    type=float, default=1.0, help="IHM task loss weight")
    p.add_argument("--w-decomp", type=float, default=1.0, help="Decompensation task loss weight")
    p.add_argument("--w-pheno",  type=float, default=1.0, help="Pheno task loss weight")

    # Regularisation
    p.add_argument("--fedprox-mu", type=float, default=0.0,
                   help="FedProx proximal coefficient (0 = disabled)")

    # I/O
    p.add_argument("--save-dir",    default="checkpoints")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--save-every",  type=int, default=5,
                   help="Save a checkpoint every N rounds (best is always saved)")
    p.add_argument("--eval-every",  type=int, default=1,
                   help="Evaluate on val set every N rounds")
    p.add_argument("--resume",      default=None, help="Path to checkpoint to resume from")

    # Smoke-test mode
    p.add_argument("--use-synthetic", action="store_true",
                   help="Use random synthetic data instead of real CSVs (for smoke testing)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root        = Path(args.root)
    save_dir    = root / args.save_dir
    results_dir = root / args.results_dir
    save_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[train] device={device}  seed={args.seed}  rounds={args.rounds}")

    # ---- Data ----
    _eicu = args.dataset == "eicu"
    _task_types = {"decomp": "regression"} if _eicu else {}

    if args.use_synthetic:
        print("[train] Using synthetic data (smoke-test mode)")
        train_loaders = make_synthetic_loaders(args.batch_size, args.max_seq_len,
                                               task_types=_task_types)
        val_loaders   = make_synthetic_loaders(args.batch_size, args.max_seq_len,
                                               n_batches=4, task_types=_task_types)
    else:
        print("[train] Building data loaders...")
        train_loaders = build_site_loaders(
            root, "train", args.batch_size, args.num_workers, args.max_seq_len,
            dataset=args.dataset,
        )
        val_loaders = build_site_loaders(
            root, "val", args.batch_size, args.num_workers, args.max_seq_len,
            dataset=args.dataset,
        )
    print(
        f"[train] Train batches — A: {len(train_loaders['A'])}, "
        f"B: {len(train_loaders['B'])}, C: {len(train_loaders['C'])}"
    )

    # ---- Models ----
    site_input_dims = {"A": 7, "B": 3 if _eicu else 4, "C": 3}
    clients: dict[str, VFLClient] = {
        site: VFLClient(
            input_dim=dim,
            hidden_dim=args.hidden_dim,
            embed_dim=args.embed_dim,
            lr=args.lr,
            device=device,
        )
        for site, dim in site_input_dims.items()
    }
    server = VFLServer(
        embed_dim=args.embed_dim,
        num_experts=args.num_experts,
        lr=args.lr,
        device=device,
        task_weights={"ihm": args.w_ihm, "decomp": args.w_decomp, "pheno": args.w_pheno},
        task_types=_task_types,
    )

    # ---- Resume ----
    start_round = 0
    if args.resume:
        start_round = load_checkpoint(Path(args.resume), clients, server)
        print(f"[train] Resumed from round {start_round}")

    # ---- Metrics CSV (append so resumed runs extend the same file) ----
    metrics_path = results_dir / f"metrics_seed{args.seed}.csv"
    csv_fields = [
        "round", "split",
        "total_loss", "ihm_loss", "decomp_loss", "pheno_loss",
        "ihm_auroc", "ihm_auprc", "decomp_auroc", "decomp_auprc", "pheno_macro_auroc",
        "elapsed_s",
    ]
    csv_file = open(metrics_path, "a", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=csv_fields, extrasaction="ignore")
    if metrics_path.stat().st_size == 0:
        writer.writeheader()

    # ---- Training loop ----
    best_ihm_auroc = 0.0
    t0 = time.time()

    for rnd in range(start_round, args.rounds):
        t_round = time.time()

        # Snapshot encoder params at round start for FedProx (if enabled)
        global_encoder_params: dict[str, dict] | None = None
        if args.fedprox_mu > 0.0:
            global_encoder_params = {
                site: client.get_encoder_params()
                for site, client in clients.items()
            }

        # ---- Train ----
        train_losses = _train_one_round_sites(
            clients, server, train_loaders,
            active_sites=["A", "B", "C"],
            site_input_dims=site_input_dims,
            all_dims=site_input_dims,
            fedprox_mu=args.fedprox_mu,
            global_encoder_params=global_encoder_params,
            task_types=_task_types,
        )

        # ---- Validate ----
        val_metrics: dict[str, float] = {}
        if (rnd + 1) % args.eval_every == 0:
            val_metrics = _evaluate_sites(
                clients, server, val_loaders,
                active_sites=["A", "B", "C"],
                site_input_dims=site_input_dims,
                all_dims=site_input_dims,
                task_types=_task_types,
            )

        elapsed = time.time() - t_round

        # ---- Log ----
        row = {
            "round":    rnd + 1,
            "split":    "val",
            "elapsed_s": f"{elapsed:.1f}",
            **train_losses,
            **val_metrics,
        }
        writer.writerow(row)
        csv_file.flush()

        val_str = ""
        if val_metrics:
            val_str = (
                f"| val IHM={val_metrics.get('ihm_auroc', float('nan')):.4f} "
                f"Decomp={val_metrics.get('decomp_auroc', float('nan')):.4f} "
                f"Pheno={val_metrics.get('pheno_macro_auroc', float('nan')):.4f} "
            )
        print(
            f"[round {rnd+1:3d}/{args.rounds}]  "
            f"loss {train_losses['total_loss']:.4f} "
            f"(ihm={train_losses['ihm_loss']:.4f} "
            f"decomp={train_losses['decomp_loss']:.4f} "
            f"pheno={train_losses['pheno_loss']:.4f})  "
            f"{val_str}[{elapsed:.0f}s]"
        )

        # ---- Checkpoint ----
        if (rnd + 1) % args.save_every == 0:
            ckpt_path = save_dir / f"ckpt_round{rnd+1:04d}_seed{args.seed}.pt"
            save_checkpoint(ckpt_path, rnd + 1, clients, server)

        if val_metrics and val_metrics.get("ihm_auroc", 0.0) > best_ihm_auroc:
            best_ihm_auroc = val_metrics["ihm_auroc"]
            best_path = save_dir / f"best_seed{args.seed}.pt"
            save_checkpoint(best_path, rnd + 1, clients, server)
            print(f"  -> new best IHM AUC={best_ihm_auroc:.4f}  saved {best_path.name}")

    # Save final checkpoint regardless of --save-every
    save_checkpoint(
        save_dir / f"final_seed{args.seed}.pt",
        args.rounds, clients, server,
    )

    csv_file.close()
    total_time = time.time() - t0
    print(f"\n[train] Done.  Total time: {total_time/60:.1f} min")
    print(f"[train] Metrics : {metrics_path}")
    print(f"[train] Best ckpt: {save_dir}/best_seed{args.seed}.pt")


if __name__ == "__main__":
    main()
