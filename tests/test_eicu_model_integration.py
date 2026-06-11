"""
tests/test_eicu_model_integration.py

Model-level integration tests for eICU in VFL-MTL.
Covers: tabular DataLoader (__len__ fix), eICU server regression head,
site-B input-dim fix, run_training smoke test, eval metric keys.
No real eICU files needed.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data_prep.dataset import (
    EICU_PHENO_LABEL_COLS,
    EICU_SITE_A_FEATURES,
    EICU_SITE_B_FEATURES,
    EICU_SITE_C_FEATURES,
    VFLSiteDataset,
    build_site_loaders,
)
from fl.client import VFLClient
from fl.server import VFLServer
from train import TrainConfig, _evaluate_sites, run_training

RNG = np.random.default_rng(0)
N       = 120
BATCH   = 16
EMBED   = 64
ALL_DIMS = {"A": 7, "B": 3, "C": 3}
TASK_TYPES = {"decomp": "regression"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_eicu_csvs(root: Path, n: int = N) -> None:
    splits_dir = root / "data" / "eicu_vertical_splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    pids = np.arange(1, n + 1)
    n_train, n_val = int(n * 0.7), int(n * 0.15)
    split_labels = np.array(
        ["train"] * n_train + ["val"] * n_val + ["test"] * (n - n_train - n_val)
    )
    RNG.shuffle(split_labels)

    pd.DataFrame({"patientunitstayid": pids, "split": split_labels}).to_csv(
        splits_dir / "aligned_patient_ids_eicu.csv", index=False
    )

    base = {"patientunitstayid": pids, "split": split_labels}

    site_a = {**base, "y_ihm": RNG.integers(0, 2, n).astype(float),
              **{f: RNG.uniform(0, 1, n).astype(np.float32) for f in EICU_SITE_A_FEATURES}}
    pd.DataFrame(site_a).to_csv(splits_dir / "site_A_eicu.csv", index=False)

    site_b = {**base, "y_rlos": RNG.uniform(0.1, 10.0, n).astype(np.float32),
              **{f: RNG.uniform(0, 1, n).astype(np.float32) for f in EICU_SITE_B_FEATURES}}
    pd.DataFrame(site_b).to_csv(splits_dir / "site_B_eicu.csv", index=False)

    site_c = {**base,
              **{lbl: RNG.integers(0, 2, n).astype(float) for lbl in EICU_PHENO_LABEL_COLS},
              **{f: RNG.uniform(0, 1, n).astype(np.float32) for f in EICU_SITE_C_FEATURES}}
    pd.DataFrame(site_c).to_csv(splits_dir / "site_C_eicu.csv", index=False)


def _clients():
    return {s: VFLClient(input_dim=d) for s, d in ALL_DIMS.items()}


def _server():
    return VFLServer(task_types=TASK_TYPES)


def _loaders(batch: int = BATCH):
    def _l(n_feat, y):
        x    = torch.randn(batch, 1, n_feat)
        mask = torch.ones(batch, 1)
        return DataLoader(TensorDataset(x, mask, y), batch_size=batch)
    return {
        "A": _l(7, torch.randint(0, 2, (batch,)).float()),
        "B": _l(3, torch.rand(batch) * 10.0),
        "C": _l(3, torch.randint(0, 2, (batch, 25)).float()),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def eicu_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("eicu_model_root")
    _make_eicu_csvs(root)
    return root


# ---------------------------------------------------------------------------
# Tabular dataset + DataLoader
# ---------------------------------------------------------------------------

def test_tabular_dataset_len(eicu_root):
    splits_dir = eicu_root / "data" / "eicu_vertical_splits"
    ds = VFLSiteDataset(
        site_csv        = splits_dir / "site_A_eicu.csv",
        feature_cols    = EICU_SITE_A_FEATURES,
        label_col       = "y_ihm",
        split           = "train",
        aligned_ids_csv = splits_dir / "aligned_patient_ids_eicu.csv",
        timeseries_root = None,
        task_type       = "binary",
        id_col          = "patientunitstayid",
    )
    assert len(ds) > 0


def test_tabular_dataset_getitem_shape(eicu_root):
    splits_dir = eicu_root / "data" / "eicu_vertical_splits"
    ds = VFLSiteDataset(
        site_csv        = splits_dir / "site_B_eicu.csv",
        feature_cols    = EICU_SITE_B_FEATURES,
        label_col       = "y_rlos",
        split           = "train",
        aligned_ids_csv = splits_dir / "aligned_patient_ids_eicu.csv",
        timeseries_root = None,
        task_type       = "regression",
        id_col          = "patientunitstayid",
    )
    x, mask, y = ds[0]
    assert x.shape   == (1, len(EICU_SITE_B_FEATURES))
    assert mask.shape == (1,)
    assert y.ndim    == 0  # scalar float


def test_build_site_loaders_eicu_batch_shapes(eicu_root):
    loaders = build_site_loaders(eicu_root, "train", batch_size=BATCH, dataset="eicu")
    assert set(loaders.keys()) == {"A", "B", "C"}
    expected_feats = {"A": 7, "B": 3, "C": 3}
    for site, loader in loaders.items():
        x, mask, y = next(iter(loader))
        assert x.shape[1:] == (1, expected_feats[site]), \
            f"Site {site} x shape {x.shape} — expected (*, 1, {expected_feats[site]})"
        assert mask.shape == (BATCH, 1)


def test_dataloader_iterates_all_splits(eicu_root):
    for split in ("train", "val", "test"):
        loaders = build_site_loaders(eicu_root, split, batch_size=BATCH, dataset="eicu")
        for site, loader in loaders.items():
            assert len(list(loader)) > 0, f"Split {split} site {site} yielded no batches"


def test_site_c_multilabel_shape(eicu_root):
    splits_dir = eicu_root / "data" / "eicu_vertical_splits"
    ds = VFLSiteDataset(
        site_csv        = splits_dir / "site_C_eicu.csv",
        feature_cols    = EICU_SITE_C_FEATURES,
        label_col       = EICU_PHENO_LABEL_COLS,
        split           = "train",
        aligned_ids_csv = splits_dir / "aligned_patient_ids_eicu.csv",
        timeseries_root = None,
        task_type       = "multilabel",
        id_col          = "patientunitstayid",
    )
    _, _, y = ds[0]
    assert y.shape == (25,)


# ---------------------------------------------------------------------------
# LSTM encoder with 1-timestep tabular input
# ---------------------------------------------------------------------------

def test_encoder_accepts_single_timestep():
    client = VFLClient(input_dim=3)
    x    = torch.randn(BATCH, 1, 3)
    mask = torch.ones(BATCH, 1)
    emb  = client.eval_forward(x, mask)
    assert emb.shape == (BATCH, EMBED)


# ---------------------------------------------------------------------------
# VFLServer regression decomp
# ---------------------------------------------------------------------------

def test_server_decomp_output_shape():
    torch.manual_seed(0)
    server = _server()
    concat = torch.randn(BATCH, 3 * EMBED)
    preds  = server.model(concat)
    assert preds["decomp"].shape == (BATCH, 1)


def test_server_decomp_head_has_no_sigmoid():
    """Regression TaskHead must be a plain Linear, not Sequential(Linear, Sigmoid)."""
    import torch.nn as nn
    server = _server()
    head = server.model.heads["decomp"].head
    assert isinstance(head, nn.Linear), \
        f"Expected nn.Linear for regression head, got {type(head).__name__}"


def test_server_decomp_loss_is_mse():
    """With targets ~U(0,10), MSE-scale loss >> BCE-scale (~0.693)."""
    torch.manual_seed(1)
    server = _server()
    embs   = {s: torch.randn(BATCH, EMBED).requires_grad_(True) for s in ("A", "B", "C")}
    labels = {
        "ihm":    torch.randint(0, 2, (BATCH,)).float(),
        "decomp": torch.rand(BATCH) * 10.0,
        "pheno":  torch.randint(0, 2, (BATCH, 25)).float(),
    }
    server.aggregate_embeddings(embs)
    _, task_losses = server.forward_and_loss(labels)
    assert task_losses["decomp"].item() > 1.0, \
        f"Decomp loss {task_losses['decomp'].item():.4f} too small to be MSE"


def test_server_ihm_pheno_unchanged():
    """IHM and Pheno must still use sigmoid/BCE regardless of decomp task_type."""
    torch.manual_seed(2)
    server = _server()
    concat = torch.randn(BATCH, 3 * EMBED)
    preds  = server.model(concat)
    assert (preds["ihm"]   >= 0).all() and (preds["ihm"]   <= 1).all()
    assert (preds["pheno"] >= 0).all() and (preds["pheno"] <= 1).all()


# ---------------------------------------------------------------------------
# site_input_dims fix: TrainConfig default (B=4) must not override eICU B=3
# ---------------------------------------------------------------------------

def test_site_dims_eicu_b_resolves_to_3():
    cfg      = TrainConfig(dataset="eicu")
    all_dims = {"A": 7, "B": 3 if cfg.dataset == "eicu" else 4, "C": 3}
    site_dims = {s: min(cfg.site_input_dims.get(s, all_dims[s]), all_dims[s])
                 for s in all_dims}
    assert site_dims["B"] == 3, \
        f"Expected B=3 for eICU after _site_dims fix, got {site_dims['B']}"


def test_site_dims_mimic_b_unchanged():
    cfg      = TrainConfig(dataset="mimic")
    all_dims = {"A": 7, "B": 4, "C": 3}
    site_dims = {s: min(cfg.site_input_dims.get(s, all_dims[s]), all_dims[s])
                 for s in all_dims}
    assert site_dims["B"] == 4, "MIMIC B dim must remain 4"


# ---------------------------------------------------------------------------
# run_training eICU smoke tests
# ---------------------------------------------------------------------------

def _eicu_cfg(**kwargs) -> TrainConfig:
    defaults = dict(
        dataset="eicu",
        task_types={"decomp": "regression"},
        use_synthetic=True,
        n_synthetic=64,
        batch_size=BATCH,
        patience=0,
        eval_every=1,
        seed=0,
    )
    defaults.update(kwargs)
    return TrainConfig(**defaults)


def test_run_training_eicu_completes():
    results = run_training(_eicu_cfg(n_rounds=3))
    assert len(results) == 3


def test_run_training_eicu_rlos_metrics_present():
    results = run_training(_eicu_cfg(n_rounds=2))
    last = results[-1]
    assert "val_rlos_mae"  in last
    assert "val_rlos_rmse" in last
    assert last["val_rlos_mae"]  >= 0
    assert last["val_rlos_rmse"] >= 0


def test_run_training_eicu_no_decomp_auroc():
    results = run_training(_eicu_cfg(n_rounds=1))
    auroc = results[-1].get("val_decomp_auroc", float("nan"))
    assert math.isnan(auroc), \
        f"Expected NaN val_decomp_auroc for eICU regression, got {auroc}"


def test_run_training_eicu_loss_finite():
    results = run_training(_eicu_cfg(n_rounds=3))
    for row in results:
        assert math.isfinite(row["train_loss"]), f"Infinite train_loss at round {row['round']}"
        assert math.isfinite(row["decomp_loss"])


def test_run_training_eicu_site_b_dim_no_error():
    """Verifies site B input_dim=3 encoder receives 3-feature data without shape error."""
    try:
        run_training(_eicu_cfg(n_rounds=1))
    except RuntimeError as e:
        if "input.size(-1) must be equal to input_size" in str(e):
            pytest.fail(f"site_input_dims fix not applied: {e}")
        raise


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def test_evaluate_sites_rlos_keys():
    torch.manual_seed(3)
    clients = _clients()
    server  = _server()
    loaders = _loaders()
    metrics = _evaluate_sites(
        clients, server, loaders, list(ALL_DIMS.keys()),
        ALL_DIMS, ALL_DIMS, TASK_TYPES,
    )
    assert "rlos_mae"  in metrics
    assert "rlos_rmse" in metrics
    assert "decomp_auroc" not in metrics
    assert metrics["rlos_mae"] >= 0


def test_evaluate_sites_ihm_pheno_still_present():
    torch.manual_seed(4)
    clients = _clients()
    server  = _server()
    loaders = _loaders()
    metrics = _evaluate_sites(
        clients, server, loaders, list(ALL_DIMS.keys()),
        ALL_DIMS, ALL_DIMS, TASK_TYPES,
    )
    assert "ihm_auroc"         in metrics
    assert "pheno_macro_auroc" in metrics


def test_gradient_flows_to_site_b_regression():
    """Site B encoder weights must update under regression loss."""
    torch.manual_seed(5)
    clients = _clients()
    server  = _server()

    before = {k: v.clone() for k, v in clients["B"].encoder.state_dict().items()}

    embs = {s: clients[s].forward(torch.randn(BATCH, 1, d),
                                  torch.ones(BATCH, 1))
            for s, d in ALL_DIMS.items()}
    labels = {
        "ihm":    torch.randint(0, 2, (BATCH,)).float(),
        "decomp": torch.rand(BATCH) * 10.0,
        "pheno":  torch.randint(0, 2, (BATCH, 25)).float(),
    }
    server.aggregate_embeddings(embs)
    loss, _ = server.forward_and_loss(labels)
    server.backward_and_step(loss)
    grads = server.get_embedding_gradients()
    clients["B"].receive_gradient(grads["B"])

    after   = clients["B"].encoder.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "Site B encoder weights did not update after regression backward pass"


if __name__ == "__main__":
    import pytest as _pytest
    _pytest.main([__file__, "-v"])
