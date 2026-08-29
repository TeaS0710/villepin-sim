"""Trouve les conditions MINIMALES qui amènent Villepin à un seuil donné
de P(victoire 2T). Stratégie :

1. Lance d'abord `extreme_search` pour trouver `x_max` (config 13D optimale,
   P(victoire) ≈ 90 %+).
2. Interpole linéairement entre baseline 2027 (`x0`) et `x_max`. À chaque
   pas, mesure P(victoire). Trouve le pas minimal α_T tel que P >= T.
3. Le vecteur x_T = x0 + α_T·(x_max − x0) est le "point d'inflexion" :
   à ce niveau de perturbation, Villepin atteint T % de probabilité.

C'est un chemin sufficient (pas globalement optimal), mais cohérent,
lisible et reproductible : meilleur compromis pour ce type de question.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .extreme_search import ARCH_ORDER, _x_to_config, search
from .parameters import TIER1_PARAMS
from .physical_model import all_candidates_2T_probabilities, load_config
from .winner_analysis import HISTORICAL_BASE_BOUNDS

N_TIER1 = len(TIER1_PARAMS)
N_BASE = len(ARCH_ORDER)


def x_baseline_2027(cfg: dict) -> np.ndarray:
    tier1 = np.full(N_TIER1, 0.5)
    bases = np.array([cfg["competitors"][a]["base"] for a in ARCH_ORDER])
    return np.concatenate([tier1, bases])


def x_to_pv(x: np.ndarray, base_cfg: dict) -> float:
    params, cfg_eff = _x_to_config(x, base_cfg)
    probs = all_candidates_2T_probabilities(params, cfg_eff)
    return probs["villepin"]["p_victory"]


def interpolate_path(x0: np.ndarray, xmax: np.ndarray, cfg: dict,
                     n_steps: int = 51) -> pd.DataFrame:
    rows = []
    for alpha in np.linspace(0, 1, n_steps):
        x = x0 + alpha * (xmax - x0)
        p = x_to_pv(x, cfg)
        rows.append({"alpha": float(alpha), "p_victory": float(p)})
    return pd.DataFrame(rows)


def find_alpha_for_target(path_df: pd.DataFrame, target: float) -> float | None:
    """Plus petit α tel que p_victory(α) >= target."""
    above = path_df[path_df["p_victory"] >= target]
    if above.empty:
        return None
    # Interpolation linéaire pour précision
    idx = above.index[0]
    if idx == 0:
        return 0.0
    prev = path_df.iloc[idx - 1]
    curr = path_df.iloc[idx]
    if curr["p_victory"] - prev["p_victory"] < 1e-9:
        return float(curr["alpha"])
    frac = (target - prev["p_victory"]) / (curr["p_victory"] - prev["p_victory"])
    return float(prev["alpha"] + frac * (curr["alpha"] - prev["alpha"]))


def describe_state(x: np.ndarray) -> dict[str, float]:
    state = {}
    for i, n in enumerate(TIER1_PARAMS):
        state[n] = float(x[i])
    for j, arch in enumerate(ARCH_ORDER):
        state[f"base_{arch}"] = float(x[N_TIER1 + j])
    return state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.fitted.yaml")
    ap.add_argument("--out", default="outputs/path_to_victory.csv")
    ap.add_argument("--targets", nargs="+", type=float,
                    default=[0.05, 0.10, 0.25, 0.50, 0.75])
    ap.add_argument("--extreme-restarts", type=int, default=10)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)

    x0 = x_baseline_2027(cfg)
    p0 = x_to_pv(x0, cfg)
    print(f"📍 Baseline 2027 : P(victoire Villepin) = {p0*100:.3f} %\n")

    print(f"=== Recherche extrême (CMA-ES 13D, {args.extreme_restarts} restarts) ===")
    extreme_df = search(cfg, n_restarts=args.extreme_restarts,
                        maxiter=200, popsize=50, seed=42)
    best_row = extreme_df.iloc[0]
    xmax = np.concatenate([
        np.array([best_row[f"param_{n}"] for n in TIER1_PARAMS]),
        np.array([best_row[f"base_{a}"] for a in ARCH_ORDER]),
    ])
    pmax = float(best_row["p_victory_villepin"])
    print(f"✓ Optimum trouvé : P(victoire) = {pmax*100:.2f} %\n")

    print(f"=== Chemin interpolé baseline → optimum (101 points) ===")
    path_df = interpolate_path(x0, xmax, cfg, n_steps=101)

    summary = []
    for T in args.targets:
        alpha_T = find_alpha_for_target(path_df, T)
        if alpha_T is None or alpha_T > 1.0:
            print(f"  P >= {T*100:.0f} %  : ❌ inatteignable (max {pmax*100:.2f} %)")
            summary.append({"target": T, "achievable": False, "alpha": None,
                            "state": None})
            continue
        x_T = x0 + alpha_T * (xmax - x0)
        state = describe_state(x_T)
        # Calcul des shifts vs baseline
        shifts = {k: round(state[k] - (0.5 if k in TIER1_PARAMS else cfg["competitors"][k.replace("base_", "")]["base"]), 3)
                  for k in state}
        # Vérification réelle de P à ce point
        p_check = x_to_pv(x_T, cfg)
        print(f"\n  P >= {T*100:.0f} % → α = {alpha_T:.3f}, P obtenu = {p_check*100:.2f} %")
        # Top 5 variables qui bougent
        sorted_shifts = sorted(shifts.items(), key=lambda kv: -abs(kv[1]))[:5]
        for k, v in sorted_shifts:
            base_val = (0.5 if k in TIER1_PARAMS
                        else cfg["competitors"][k.replace("base_", "")]["base"])
            print(f"    {k:<28s} {base_val:>6.2f} → {state[k]:>6.2f}  (Δ={v:+.2f})")
        summary.append({"target": T, "achievable": True, "alpha": alpha_T,
                        "p_check": p_check,
                        "state": state, "shifts": shifts})

    # Sauvegarde
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    path_df.to_csv(out, index=False)
    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump({
            "baseline_p_victory": float(p0),
            "extreme_p_victory": float(pmax),
            "baseline_state": describe_state(x0),
            "extreme_state": describe_state(xmax),
            "targets": summary,
        }, f, indent=2, default=str)
    print(f"\n✓ Sauvegardé : {out} (101 points) + {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
