"""MLP surrogate : prédit (score_1T, p_qualif, p_victoire) à partir des
paramètres Tier 1. Sert à accélérer 1000× les évaluations pendant CMA-ES.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .parameters import TIER1_PARAMS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class _ResidualBlock(nn.Module):
    """Block résiduel pre-norm : LN -> GELU -> Dropout -> Linear -> + skip."""
    def __init__(self, d: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.lin1 = nn.Linear(d, 4 * d)
        self.lin2 = nn.Linear(4 * d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = nn.functional.gelu(self.lin1(h))
        h = self.dropout(h)
        h = self.lin2(h)
        return x + h


class VillepinPredictor(nn.Module):
    """MLP surrogate v2 : profond et résiduel.

    Default v2 : 4 blocs résiduels de 256 hidden = ~660k params.
    Configurable via `cfg.pipeline.neural_network.hidden` et `n_layers`.
    """
    def __init__(self, n_inputs: int, n_hidden: int = 256, dropout: float = 0.1,
                 n_layers: int = 4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(n_inputs, n_hidden),
            nn.LayerNorm(n_hidden),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            _ResidualBlock(n_hidden, dropout) for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(n_hidden),
            nn.Linear(n_hidden, n_hidden // 2),
            nn.GELU(),
            nn.Linear(n_hidden // 2, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for blk in self.blocks:
            h = blk(h)
        out = self.head(h)
        score = torch.sigmoid(out[:, 0:1]) * 50.0
        p_qualif = torch.sigmoid(out[:, 1:2])
        p_victory = torch.sigmoid(out[:, 2:3])
        return torch.cat([score, p_qualif, p_victory], dim=1)


@dataclass
class TrainResult:
    val_mae_score: float
    val_mae_qualif: float
    val_mae_victory: float
    test_mae_score: float
    test_mae_qualif: float
    test_mae_victory: float
    epochs: int


def _to_tensors(df: pd.DataFrame, x_cols: list[str], y_cols: list[str]):
    X = df[x_cols].values.astype(np.float32)
    Y = df[y_cols].values.astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(Y)


def train(
    data_path: str | Path,
    out_path: str | Path,
    cfg: dict,
    seed: int = 42,
) -> TrainResult:
    torch.manual_seed(seed)
    np.random.seed(seed)
    df = pd.read_parquet(data_path)
    # Inférer x_cols depuis le dataset (Tier 1 + éventuels Tier 2)
    y_set = {"score_1T_mean", "score_1T_median", "score_1T_p5", "score_1T_p95",
             "p_qualif", "p_victory", "n_samples"}
    x_cols = [c for c in df.columns if c not in y_set]
    y_cols = ["score_1T_mean", "p_qualif", "p_victory"]

    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=seed)
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=seed)
    Xtr, Ytr = _to_tensors(train_df, x_cols, y_cols)
    Xv, Yv = _to_tensors(val_df, x_cols, y_cols)
    Xte, Yte = _to_tensors(test_df, x_cols, y_cols)

    nn_cfg = cfg["pipeline"]["neural_network"]
    bs = nn_cfg["batch_size"]
    loader = DataLoader(TensorDataset(Xtr, Ytr), batch_size=bs, shuffle=True)

    model = VillepinPredictor(
        n_inputs=len(x_cols),
        n_hidden=nn_cfg["hidden"],
        dropout=nn_cfg["dropout"],
        n_layers=nn_cfg.get("n_layers", 4),
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  NN: {len(x_cols)} inputs → {n_params:,} params")
    optim = torch.optim.AdamW(model.parameters(), lr=nn_cfg["lr"], weight_decay=nn_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=nn_cfg["max_epochs"])
    weights = torch.tensor(nn_cfg["loss_weights"], dtype=torch.float32, device=DEVICE)

    best_val_loss = float("inf")
    best_state = None
    patience = nn_cfg["patience"]
    bad_epochs = 0

    def weighted_mse(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return ((pred - target) ** 2 * weights).mean()

    Xv_d = Xv.to(DEVICE)
    Yv_d = Yv.to(DEVICE)

    for epoch in range(nn_cfg["max_epochs"]):
        model.train()
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optim.zero_grad()
            loss = weighted_mse(model(xb), yb)
            loss.backward()
            optim.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vloss = weighted_mse(model(Xv_d), Yv_d).item()
        if vloss < best_val_loss - 1e-4:
            best_val_loss = vloss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:3d}  val_loss={vloss:.4f}  best={best_val_loss:.4f}")
        if bad_epochs >= patience:
            print(f"  early stop @ epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        Yp_v = model(Xv_d).cpu().numpy()
        Yp_te = model(Xte.to(DEVICE)).cpu().numpy()
    Yv_np, Yte_np = Yv.numpy(), Yte.numpy()
    result = TrainResult(
        val_mae_score=float(mean_absolute_error(Yv_np[:, 0], Yp_v[:, 0])),
        val_mae_qualif=float(mean_absolute_error(Yv_np[:, 1], Yp_v[:, 1])),
        val_mae_victory=float(mean_absolute_error(Yv_np[:, 2], Yp_v[:, 2])),
        test_mae_score=float(mean_absolute_error(Yte_np[:, 0], Yp_te[:, 0])),
        test_mae_qualif=float(mean_absolute_error(Yte_np[:, 1], Yp_te[:, 1])),
        test_mae_victory=float(mean_absolute_error(Yte_np[:, 2], Yp_te[:, 2])),
        epochs=epoch + 1,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "n_inputs": len(x_cols),
        "hidden": nn_cfg["hidden"],
        "dropout": nn_cfg["dropout"],
        "n_layers": nn_cfg.get("n_layers", 4),
        "x_cols": x_cols,
        "y_cols": y_cols,
    }, out_path)
    return result


def load_model(checkpoint_path: str | Path):
    """Retourne (model, x_cols)."""
    ckpt = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = VillepinPredictor(
        n_inputs=ckpt["n_inputs"],
        n_hidden=ckpt["hidden"],
        dropout=ckpt["dropout"],
        n_layers=ckpt.get("n_layers", 4),
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt["x_cols"]


def predict_batch(model: VillepinPredictor, X: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.from_numpy(X.astype(np.float32)).to(DEVICE)
        return model(x).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.fitted.yaml")
    ap.add_argument("--data", default="outputs/dataset.parquet")
    ap.add_argument("--out", default="outputs/checkpoints/nn_surrogate.pt")
    args = ap.parse_args()
    from .physical_model import load_config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)
    r = train(args.data, args.out, cfg, seed=cfg.get("seed", 42))
    print(f"\n=== Métriques surrogate ===")
    print(f"  val_mae_score    = {r.val_mae_score:.3f}")
    print(f"  val_mae_qualif   = {r.val_mae_qualif:.3f}")
    print(f"  val_mae_victory  = {r.val_mae_victory:.3f}")
    print(f"  test_mae_score   = {r.test_mae_score:.3f}")
    print(f"  test_mae_qualif  = {r.test_mae_qualif:.3f}")
    print(f"  test_mae_victory = {r.test_mae_victory:.3f}")
    print(f"  epochs           = {r.epochs}")


if __name__ == "__main__":
    main()
