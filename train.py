"""
train.py — Round-based VFL-MTL training loop.

Each "round" is one pass over the training dataset (epoch). Per-batch protocol:
  1. Each VFLClient runs its LSTM encoder → detached embedding (cut layer)
  2. VFLServer concatenates embeddings, forward through MMoE, computes weighted loss
  3. VFLServer backpropagates, slices embedding gradients per site
  4. Each VFLClient receives its gradient slice, backpropagates into LSTM, updates weights

Validation runs after every --eval-every rounds and logs per-task metrics to
results/metrics_seed{N}.csv.

Usage
-----
  # Basic run:
  python train.py --root . --rounds 50 --seed 42

  # With GPU and FedProx regularisation:
  python train.py --root . --rounds 50 --device cuda --fedprox-mu 0.01 --seed 42

  # Resume from checkpoint:
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
import torch
from sklearn.metrics import (
    average_precision_score,
    cohen_kappa_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader, TensorDataset

from data_prep.dataset import build_site_loaders
from fl.client import VFLClient
from fl.fedavg import fedavg_aggregate
from fl.fedprox import fedprox_penalty
from fl.server import VFLServer


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All hyperparameters and flags for a single VFL-MTL training run."""
    splits_dir:      str   = "data/vertical_splits"
    n_rounds:        int   = 50
    batch_size:      int   = 64
    lr:              float = 1e-3
    seed:            int   = 42
    device:          str   = "cpu"
    hidden_dim:      int   = 128
    embed_dim:       int   = 64
    num_experts:     int   = 4
    task_weights:    dict  = field(default_factory=lambda: {"ihm": 1.0, "los": 1.0, "pheno": 1.0})
    site_input_dims: dict  = field(default_factory=lambda: {"A": 7, "B": 4, "C": 3})
    n_sites:         int   = 3
    use_fedavg:      bool  = True
    fedavg_every:    int   = 5
    fedprox_mu:      float = 0.0
    use_synthetic:   bool  = False
    n_synthetic:     int   = 256
    num_workers:     int   = 0
    max_seq_len:     int   = 48
    eval_every:      int   = 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Synthetic data (smoke-test mode)
# ---------------------------------------------------------------------------

def make_synthetic_loaders(
    batch_size: int,
    seq_len: int,
    n_batches: int = 10,
) -> dict[str, DataLoader]:
    """
    Generate random tensors matching the shape contract of build_site_loaders().
    Used with --use-synthetic to verify the training loop without real data.

    Shapes per batch:
      Site A: x (B, T, 7)  mask (B, T)  y_ihm  (B,) float32
      Site B: x (B, T, 4)  mask (B, T)  y_los  (B,) int64
      Site C: x (B, T, 3)  mask (B, T)  y_pheno (B, 25) float32
    """
    N = batch_size * n_batches
    T = seq_len

    def _loader(n_feat: int, y: torch.Tensor) -> DataLoader:
        x    = torch.randn(N, T, n_feat)
        mask = torch.ones(N, T)
        ds   = TensorDataset(x, mask, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    loaders = {
        "A": _loader(7,  torch.randint(0, 2, (N,)).float()),
        "B": _loader(4,  torch.randint(0, 10, (N,)).long()),
        "C": _loader(3,  torch.randint(0, 2, (N, 25)).float()),
    }
    return loaders


# ---------------------------------------------------------------------------
# Per-task metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    all_preds: dict[str, list],
    all_labels: dict[str, list],
) -> dict[str, float]:
    """
    IHM  : AUC-ROC, AUC-PR              (Harutyunyan et al. 2019, primary metrics)
    LOS  : Cohen's kappa (bin predicted vs. bin true)
    Pheno: macro-AUC-ROC across 25 labels (skip labels with no positives in split)
    """
    metrics: dict[str, float] = {}

    # IHM — binary probabilities
    p_ihm = np.concatenate(all_preds["ihm"])   # (N,)
    y_ihm = np.concatenate(all_labels["ihm"])  # (N,)
    metrics["ihm_auroc"] = float(roc_auc_score(y_ihm, p_ihm))
    metrics["ihm_auprc"] = float(average_precision_score(y_ihm, p_ihm))

    # LOS — argmax of logits → predicted bin index
    p_los = np.concatenate(all_preds["los"])   # (N, 10)
    y_los = np.concatenate(all_labels["los"])  # (N,)
    metrics["los_kappa"] = float(cohen_kappa_score(y_los, p_los.argmax(axis=1), weights="quadratic"))

    # Pheno — per-label AUC averaged over labels that have at least one positive
    p_pheno = np.concatenate(all_preds["pheno"])  # (N, 25)
    y_pheno = np.concatenate(all_labels["pheno"]) # (N, 25)
    per_label_aucs = [
        roc_auc_score(y_pheno[:, i], p_pheno[:, i])
        for i in range(y_pheno.shape[1])
        if y_pheno[:, i].sum() > 0
    ]
    metrics["pheno_macro_auroc"] = float(np.mean(per_label_aucs)) if per_label_aucs else float("nan")

    return metrics


# ---------------------------------------------------------------------------
# One training round
# ---------------------------------------------------------------------------

def train_one_round(
    clients: dict[str, VFLClient],
    server: VFLServer,
    loaders: dict,
    fedprox_mu: float,
    global_encoder_params: dict[str, dict] | None,
) -> dict[str, float]:
    """
    Iterate over aligned batches from all three site loaders.

    FedProx: if fedprox_mu > 0 and global_encoder_params are provided,
    a proximal correction step is applied to each client encoder after the
    normal VFL backward pass. This keeps local encoders close to the
    round-start parameters, improving convergence in heterogeneous settings.

    Returns averaged loss values for logging.
    """
    task_loss_sums = {"ihm": 0.0, "los": 0.0, "pheno": 0.0}
    total_loss_sum = 0.0
    n_batches = 0

    # All three loaders cover the same PSI-aligned ICU stays; zip stops
    # at the shortest loader. Lengths should be equal but zip is safe.
    for batch_A, batch_B, batch_C in zip(loaders["A"], loaders["B"], loaders["C"]):
        x_A, mask_A, y_ihm   = batch_A
        x_B, mask_B, y_los   = batch_B
        x_C, mask_C, y_pheno = batch_C

        # Step 1 — local forward passes (returns detached embeddings)
        emb_A = clients["A"].forward(x_A, mask_A)
        emb_B = clients["B"].forward(x_B, mask_B)
        emb_C = clients["C"].forward(x_C, mask_C)

        # Step 2 — server: aggregate, forward MMoE, compute loss
        server.aggregate_embeddings({"A": emb_A, "B": emb_B, "C": emb_C})
        total_loss, task_losses = server.forward_and_loss({
            "ihm":   y_ihm,
            "los":   y_los,
            "pheno": y_pheno,
        })

        # Step 3 — server backward + weight update
        server.backward_and_step(total_loss)

        # Step 4 — distribute embedding gradients back to clients
        grads = server.get_embedding_gradients()
        clients["A"].receive_gradient(grads["A"])
        clients["B"].receive_gradient(grads["B"])
        clients["C"].receive_gradient(grads["C"])

        # Optional FedProx correction: one extra gradient step per client
        # toward the round-start (global) encoder parameters.
        if fedprox_mu > 0.0 and global_encoder_params is not None:
            for site, client in clients.items():
                penalty = fedprox_penalty(
                    client.encoder,
                    global_encoder_params[site],
                    mu=fedprox_mu,
                )
                client.optimizer.zero_grad()
                penalty.backward()
                client.optimizer.step()

        for t, loss in task_losses.items():
            task_loss_sums[t] += loss.item()
        total_loss_sum += total_loss.item()
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Training loaders returned no batches — check data paths.")

    return {
        "total_loss": total_loss_sum / n_batches,
        "ihm_loss":   task_loss_sums["ihm"]   / n_batches,
        "los_loss":   task_loss_sums["los"]   / n_batches,
        "pheno_loss": task_loss_sums["pheno"] / n_batches,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    clients: dict[str, VFLClient],
    server: VFLServer,
    loaders: dict,
) -> dict[str, float]:
    """Run inference on val or test split; return per-task metrics."""
    all_preds: dict[str, list]  = {"ihm": [], "los": [], "pheno": []}
    all_labels: dict[str, list] = {"ihm": [], "los": [], "pheno": []}

    for batch_A, batch_B, batch_C in zip(loaders["A"], loaders["B"], loaders["C"]):
        x_A, mask_A, y_ihm   = batch_A
        x_B, mask_B, y_los   = batch_B
        x_C, mask_C, y_pheno = batch_C

        emb_A = clients["A"].eval_forward(x_A, mask_A)
        emb_B = clients["B"].eval_forward(x_B, mask_B)
        emb_C = clients["C"].eval_forward(x_C, mask_C)

        preds = server.predict({"A": emb_A, "B": emb_B, "C": emb_C})

        all_preds["ihm"].append(preds["ihm"].squeeze(-1).cpu().numpy())
        all_preds["los"].append(preds["los"].cpu().numpy())
        all_preds["pheno"].append(preds["pheno"].cpu().numpy())

        all_labels["ihm"].append(y_ihm.numpy())
        all_labels["los"].append(y_los.numpy())
        all_labels["pheno"].append(y_pheno.numpy())

    return compute_metrics(all_preds, all_labels)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    round_idx: int,
    clients: dict[str, VFLClient],
    server: VFLServer,
) -> None:
    torch.save(
        {
            "round":      round_idx,
            "client_A":   clients["A"].encoder.state_dict(),
            "client_B":   clients["B"].encoder.state_dict(),
            "client_C":   clients["C"].encoder.state_dict(),
            "server":     server.model.state_dict(),
            "server_opt": server.optimizer.state_dict(),
        },
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

def run_training(cfg: TrainConfig) -> list[dict]:
    """
    Run a full VFL-MTL training loop from a TrainConfig and return per-round metrics.

    Returns
    -------
    list of dicts, one per round, each containing:
        round, train_loss, ihm_loss, los_loss, pheno_loss, elapsed_s
        + val metrics (val_ihm_auroc, val_ihm_auprc, val_los_kappa,
          val_pheno_macro_auroc) on eval rounds
    """
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ---- Data ----
    _all_dims = {"A": 7, "B": 4, "C": 3}
    active_sites = list(_all_dims.keys())[:cfg.n_sites]

    if cfg.use_synthetic:
        n_batches = max(1, cfg.n_synthetic // cfg.batch_size)
        train_loaders = make_synthetic_loaders(cfg.batch_size, cfg.max_seq_len, n_batches)
        val_loaders   = make_synthetic_loaders(cfg.batch_size, cfg.max_seq_len, max(1, n_batches // 4))
        # Trim to active sites
        train_loaders = {s: train_loaders[s] for s in active_sites}
        val_loaders   = {s: val_loaders[s]   for s in active_sites}
    else:
        # build_site_loaders expects project root and appends data/vertical_splits internally
        project_root = Path(cfg.splits_dir).parents[1]
        train_loaders = build_site_loaders(
            project_root, "train", cfg.batch_size, cfg.num_workers, cfg.max_seq_len
        )
        val_loaders = build_site_loaders(
            project_root, "val", cfg.batch_size, cfg.num_workers, cfg.max_seq_len
        )
        train_loaders = {s: train_loaders[s] for s in active_sites}
        val_loaders   = {s: val_loaders[s]   for s in active_sites}

    # ---- Models ----
    clients: dict[str, VFLClient] = {
        s: VFLClient(
            input_dim=cfg.site_input_dims.get(s, _all_dims[s]),
            hidden_dim=cfg.hidden_dim,
            embed_dim=cfg.embed_dim,
            lr=cfg.lr,
            device=device,
        )
        for s in active_sites
    }
    server = VFLServer(
        embed_dim=cfg.embed_dim,
        num_experts=cfg.num_experts,
        lr=cfg.lr,
        device=device,
        task_weights=cfg.task_weights,
        n_sites=cfg.n_sites,
    )

    results = []

    for rnd in range(cfg.n_rounds):
        t_round = time.time()

        # Snapshot for FedProx
        global_encoder_params: dict[str, dict] | None = None
        if cfg.fedprox_mu > 0.0:
            global_encoder_params = {s: c.get_encoder_params() for s, c in clients.items()}

        # ---- Train one round ----
        # Build site-limited loaders dict for train_one_round
        round_train = {s: train_loaders[s] for s in active_sites}
        train_losses = _train_one_round_sites(
            clients, server, round_train, active_sites,
            cfg.site_input_dims, _all_dims,
            cfg.fedprox_mu, global_encoder_params,
        )

        # ---- FedAvg ----
        # Only aggregate encoders with matching architectures (same input_dim).
        # In our heterogeneous VFL setup each site has a unique input_dim, so
        # FedAvg across sites is skipped — aggregation would require identical shapes.
        if cfg.use_fedavg and (rnd + 1) % cfg.fedavg_every == 0:
            dims = [cfg.site_input_dims.get(s, _all_dims[s]) for s in active_sites]
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
                                          cfg.site_input_dims, _all_dims)

        elapsed = time.time() - t_round
        row = {
            "round":      rnd + 1,
            "train_loss": train_losses["total_loss"],
            "ihm_loss":   train_losses["ihm_loss"],
            "los_loss":   train_losses["los_loss"],
            "pheno_loss": train_losses["pheno_loss"],
            "elapsed_s":  round(elapsed, 1),
        }
        if val_metrics:
            row["val_ihm_auroc"]        = val_metrics.get("ihm_auroc", float("nan"))
            row["val_ihm_auprc"]        = val_metrics.get("ihm_auprc", float("nan"))
            row["val_los_kappa"]        = val_metrics.get("los_kappa", float("nan"))
            row["val_pheno_macro_auroc"] = val_metrics.get("pheno_macro_auroc", float("nan"))
        results.append(row)

    return results


def _train_one_round_sites(
    clients, server, loaders, active_sites,
    site_input_dims, all_dims,
    fedprox_mu, global_encoder_params,
) -> dict[str, float]:
    """train_one_round generalised to arbitrary active_sites with feature truncation."""
    task_loss_sums = {"ihm": 0.0, "los": 0.0, "pheno": 0.0}
    total_loss_sum = 0.0
    n_batches = 0

    loader_iters = [loaders[s] for s in active_sites]

    for batches in zip(*loader_iters):
        # batches is a tuple of (x, mask, y) per site in active_sites order
        embeddings = {}
        labels = {}
        for site, (x, mask, y) in zip(active_sites, batches):
            dim = site_input_dims.get(site, all_dims[site])
            x = x[..., :dim]
            embeddings[site] = clients[site].forward(x, mask)
            if site == "A":
                labels["ihm"] = y
            elif site == "B":
                labels["los"] = y
            elif site == "C":
                labels["pheno"] = y

        # Pad missing tasks with zero labels if n_sites < 3
        if "los" not in labels:
            labels["los"] = torch.zeros(list(embeddings.values())[0].shape[0], dtype=torch.long)
        if "pheno" not in labels:
            labels["pheno"] = torch.zeros(list(embeddings.values())[0].shape[0], 25)

        server.aggregate_embeddings(embeddings)
        total_loss, task_losses = server.forward_and_loss(labels)
        server.backward_and_step(total_loss)

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
        "total_loss": total_loss_sum / n_batches,
        "ihm_loss":   task_loss_sums["ihm"]   / n_batches,
        "los_loss":   task_loss_sums["los"]   / n_batches,
        "pheno_loss": task_loss_sums["pheno"] / n_batches,
    }


@torch.no_grad()
def _evaluate_sites(clients, server, loaders, active_sites, site_input_dims, all_dims) -> dict[str, float]:
    """evaluate() generalised to arbitrary active_sites with feature truncation."""
    all_preds:  dict[str, list] = {"ihm": [], "los": [], "pheno": []}
    all_labels: dict[str, list] = {"ihm": [], "los": [], "pheno": []}

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
                all_labels["los"].append(y.numpy())
            elif site == "C":
                all_labels["pheno"].append(y.numpy())

        preds = server.predict(embeddings)
        all_preds["ihm"].append(preds["ihm"].squeeze(-1).cpu().numpy())
        all_preds["los"].append(preds["los"].cpu().numpy())
        all_preds["pheno"].append(preds["pheno"].cpu().numpy())

    # Fall back gracefully if a task has no labels (n_sites < 3)
    metrics: dict[str, float] = {}
    if all_labels["ihm"]:
        p = np.concatenate(all_preds["ihm"])
        y = np.concatenate(all_labels["ihm"])
        metrics["ihm_auroc"] = float(roc_auc_score(y, p))
        metrics["ihm_auprc"] = float(average_precision_score(y, p))
    if all_labels["los"]:
        p = np.concatenate(all_preds["los"])
        y = np.concatenate(all_labels["los"])
        metrics["los_kappa"] = float(cohen_kappa_score(y, p.argmax(axis=1), weights="quadratic"))
    if all_labels["pheno"]:
        p = np.concatenate(all_preds["pheno"])
        y = np.concatenate(all_labels["pheno"])
        per_label = [roc_auc_score(y[:, i], p[:, i]) for i in range(y.shape[1]) if y[:, i].sum() > 0]
        metrics["pheno_macro_auroc"] = float(np.mean(per_label)) if per_label else float("nan")

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="VFL-MTL training loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data
    p.add_argument("--root",        default=".",  help="Project root directory")
    p.add_argument("--max-seq-len", type=int, default=48)
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count")

    # Training
    p.add_argument("--rounds",     type=int,   default=50,   help="Total training rounds")
    p.add_argument("--batch-size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=1e-3, help="Adam lr for all components")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="cpu")

    # Architecture
    p.add_argument("--hidden-dim",  type=int, default=128, help="LSTM hidden size")
    p.add_argument("--embed-dim",   type=int, default=64,  help="Cut-layer embedding size")
    p.add_argument("--num-experts", type=int, default=4,   help="MMoE shared expert count")

    # Loss weights
    p.add_argument("--w-ihm",   type=float, default=1.0, help="IHM task loss weight")
    p.add_argument("--w-los",   type=float, default=1.0, help="LOS task loss weight")
    p.add_argument("--w-pheno", type=float, default=1.0, help="Pheno task loss weight")

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
    if args.use_synthetic:
        print("[train] Using synthetic data (smoke-test mode)")
        train_loaders = make_synthetic_loaders(args.batch_size, args.max_seq_len)
        val_loaders   = make_synthetic_loaders(args.batch_size, args.max_seq_len, n_batches=4)
    else:
        print("[train] Building data loaders...")
        train_loaders = build_site_loaders(
            root, "train", args.batch_size, args.num_workers, args.max_seq_len
        )
        val_loaders = build_site_loaders(
            root, "val", args.batch_size, args.num_workers, args.max_seq_len
        )
    print(
        f"[train] Train batches — A: {len(train_loaders['A'])}, "
        f"B: {len(train_loaders['B'])}, C: {len(train_loaders['C'])}"
    )

    # ---- Models ----
    # Input dims fixed by the MIMIC-III vertical split protocol (CLAUDE.md)
    site_input_dims = {"A": 7, "B": 4, "C": 3}
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
        task_weights={"ihm": args.w_ihm, "los": args.w_los, "pheno": args.w_pheno},
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
        "total_loss", "ihm_loss", "los_loss", "pheno_loss",
        "ihm_auroc", "ihm_auprc", "los_kappa", "pheno_macro_auroc",
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
        train_losses = train_one_round(
            clients, server, train_loaders,
            fedprox_mu=args.fedprox_mu,
            global_encoder_params=global_encoder_params,
        )

        # ---- Validate ----
        val_metrics: dict[str, float] = {}
        if (rnd + 1) % args.eval_every == 0:
            val_metrics = evaluate(clients, server, val_loaders)

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
                f"LOS-κ={val_metrics.get('los_kappa', float('nan')):.4f} "
                f"Pheno={val_metrics.get('pheno_macro_auroc', float('nan')):.4f} "
            )
        print(
            f"[round {rnd+1:3d}/{args.rounds}]  "
            f"loss {train_losses['total_loss']:.4f} "
            f"(ihm={train_losses['ihm_loss']:.4f} "
            f"los={train_losses['los_loss']:.4f} "
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
