"""Génère le dataset synthétique LHS + Monte Carlo pour entraîner le surrogate NN.

Pour chaque configuration de paramètres (LHS dans [0,1]^N), lance M simulations
Monte Carlo (bruit gaussien clippé) et agrège les statistiques.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc
from tqdm import tqdm

from .parameters import TIER1_PARAMS, default_space, space_from_config
from .physical_model import load_config, simulate_monte_carlo


def generate_dataset(
    cfg: dict,
    n_samples: int,
    n_mc: int,
    noise_std: float,
    seed: int = 42,
) -> pd.DataFrame:
    space = space_from_config(cfg)
    sampler = qmc.LatinHypercube(d=space.n, seed=seed)
    raw = sampler.random(n=n_samples)
    X = qmc.scale(raw, space.lower, space.upper)

    rng_master = np.random.default_rng(seed)
    rows = []
    for i in tqdm(range(n_samples), desc="LHS+MC"):
        params = {name: float(X[i, j]) for j, name in enumerate(space.names)}
        rng = np.random.default_rng(rng_master.integers(0, 2**31 - 1))
        stats = simulate_monte_carlo(
            params, cfg, n_samples=n_mc, noise_std=noise_std, rng=rng,
        )
        rows.append({**params, **stats})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.fitted.yaml")
    ap.add_argument("--out", default="outputs/dataset.parquet")
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--n-mc", type=int, default=None)
    ap.add_argument("--noise-std", type=float, default=None)
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
        print(f"[warn] config fitté absent, fallback sur {cfg_path}")
    cfg = load_config(cfg_path)
    mc = cfg["pipeline"]["monte_carlo"]
    n_samples = args.n_samples or (mc["n_samples_full"] if args.mode == "full" else mc["n_samples_quick"])
    n_mc = args.n_mc or mc["n_mc_per_sample"]
    noise = args.noise_std if args.noise_std is not None else mc["noise_std"]

    print(f"Génération : {n_samples} configs × {n_mc} MC, noise σ={noise}")
    df = generate_dataset(cfg, n_samples, n_mc, noise, seed=cfg.get("seed", 42))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✓ {len(df)} lignes -> {out_path}")
    print("\nDistribution score_1T_mean :")
    print(df["score_1T_mean"].describe().round(2))
    print(f"\nP(qualif) > 0 : {(df['p_qualif'] > 0).mean()*100:.1f}% des configs")
    print(f"P(victoire) > 0.05 : {(df['p_victory'] > 0.05).mean()*100:.1f}% des configs")
    print(f"P(victoire) > 0.5 : {(df['p_victory'] > 0.5).mean()*100:.1f}% des configs")


if __name__ == "__main__":
    main()
