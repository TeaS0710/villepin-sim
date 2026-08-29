"""Analyse comparée : P(victoire 2T) pour TOUS les candidats par scénario.

Repose sur `all_candidates_2T_probabilities` qui symétrise le mécanisme du
front républicain selon l'extrémisme relatif des deux finalistes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import copy

from .genetic_optimizer import EXOGENOUS_SCENARIOS
from .parameters import INTERNAL, TIER1_PARAMS
from .physical_model import all_candidates_2T_probabilities, load_config

# Bornes historiques pour les bases concurrents (plages observées 2002-2022).
# Permet d'explorer des "shocks" sans biais : qu'arrive-t-il si Bardella tombe
# à 18% (Le Pen 2007) ou si Philippe monte à 28% (Macron 2017) ?
HISTORICAL_BASE_BOUNDS = {
    # v6 (config.yaml)
    "bardella":   (15.0, 38.0),
    "philippe":   ( 3.0, 28.0),
    "melenchon":  ( 4.0, 22.0),
    "retailleau": ( 3.0, 22.0),
    "glucksmann": ( 2.0, 18.0),
    # v8 multiclasses (config_multiclass.yaml) : bornes historiques 2002-2022
    "extreme_droite":   (10.0, 38.0),   # Le Pen 2007 (10.4) -> Bardella 2027 ceiling
    "souverainiste":    ( 0.5,  8.0),   # marginal (Villiers 2.2 max)
    "droite_classique": ( 4.0, 35.0),   # Pécresse 2022 (4.8) -> Sarkozy 2007 (31)
    "centre_gouv":      ( 0.5, 30.0),   # quasi vide -> Macron 2017 (24) / 2022 (28)
    "gauche_socdem":    ( 4.0, 35.0),   # Hamon 2017 (6.4) -> Jospin 2002 (16+Mamère)
    "gauche_radicale":  ( 2.0, 25.0),   # PCF Buffet 1.9 -> Mélenchon 2022 (22)
    "extreme_gauche":   ( 0.5, 12.0),   # extrême gauche atomisée (LO+LCR 2002 ~10)
}

# Scénarios "shock" : variantes du baseline où les bases bougent dans bornes
# historiques pour simuler des dynamiques externes (scandale, effondrement…).
SHOCK_SCENARIOS: dict[str, dict[str, float]] = {
    # v6 (clés bardella/philippe) ET v8 multiclasses (clés extreme_droite/centre_gouv)
    # On garde les deux, et `analyse_scenario` filtre celles qui existent dans cfg.
    "baseline":          {},
    "ed_collapse":       {"bardella": 18.0, "extreme_droite": 18.0},
    "centre_collapse":   {"philippe":  8.0, "centre_gouv":     8.0},
    "both_collapse":     {"bardella": 18.0, "extreme_droite": 18.0,
                          "philippe":  8.0, "centre_gouv":     8.0},
    "ed_split":          {"bardella": 24.0, "extreme_droite": 24.0},
    "centre_consolide":  {"philippe": 26.0, "centre_gouv":    26.0},
}


def analyse_scenario(
    cfg: dict, exo: dict[str, float], n_mc: int, noise: float, seed: int,
    internal_strategy: dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Pour un scénario exogène donné, lance n_mc Monte Carlo et agrège les
    P(victoire 2T) par candidat.
    """
    rng = np.random.default_rng(seed)
    if internal_strategy is None:
        # Stratégie interne "moyenne" pour ne pas biaiser
        internal_strategy = {n: 0.5 for n in INTERNAL}
    base_params = {**internal_strategy, **exo}
    accum: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for _ in range(n_mc):
        noisy = {
            k: float(np.clip(v + rng.normal(0, noise), 0.0, 1.0))
            for k, v in base_params.items()
        }
        probs = all_candidates_2T_probabilities(noisy, cfg)
        for name, d in probs.items():
            if name not in accum:
                accum[name] = {"score_1T_sum": 0.0, "qualif_count": 0, "p_victory_sum": 0.0}
                counts[name] = 0
            accum[name]["score_1T_sum"] += d["score_1T"]
            accum[name]["qualif_count"] += int(d["qualified"])
            accum[name]["p_victory_sum"] += d["p_victory"]
            counts[name] += 1
    result = {}
    for name, a in accum.items():
        c = counts[name]
        result[name] = {
            "score_1T_mean": round(a["score_1T_sum"] / c, 2),
            "p_qualif": round(a["qualif_count"] / c, 3),
            "p_victory": round(a["p_victory_sum"] / c, 4),
        }
    return result


def _cfg_with_base_overrides(cfg: dict, overrides: dict[str, float]) -> dict:
    new_cfg = copy.deepcopy(cfg)
    for arch, base in overrides.items():
        if arch in new_cfg["competitors"]:
            new_cfg["competitors"][arch]["base"] = float(base)
    return new_cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.fitted.yaml")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--n-mc", type=int, default=500)
    ap.add_argument("--noise", type=float, default=0.10)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)
    seed = cfg.get("seed", 42)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Combinaison exogénique × shock = produit cartésien
    rows = []
    combos = [(s_name, exo, sh_name, sh_over)
              for s_name, exo in EXOGENOUS_SCENARIOS.items()
              for sh_name, sh_over in SHOCK_SCENARIOS.items()]
    for s_name, exo, sh_name, sh_over in tqdm(combos, desc="scénarios×shocks"):
        cfg_eff = _cfg_with_base_overrides(cfg, sh_over)
        result = analyse_scenario(cfg_eff, exo, n_mc=args.n_mc, noise=args.noise, seed=seed)
        for cand, stats in result.items():
            rows.append({
                "exo_scenario": s_name,
                "shock_scenario": sh_name,
                "candidate": cand,
                **{f"base_{k}": v for k, v in sh_over.items()},
                **stats,
            })
    df = pd.DataFrame(rows)
    out = out_dir / "winner_probabilities.csv"
    df.to_csv(out, index=False)
    print(f"\n✓ {len(df)} lignes -> {out}")

    print("\n=== Vainqueur le plus probable par combinaison (top-1) ===")
    leaders = (df.sort_values(["exo_scenario", "shock_scenario", "p_victory"],
                              ascending=[True, True, False])
                 .groupby(["exo_scenario", "shock_scenario"]).head(1).reset_index(drop=True))
    print(leaders[["exo_scenario", "shock_scenario", "candidate",
                   "score_1T_mean", "p_qualif", "p_victory"]].to_string(index=False))

    print("\n=== Meilleurs scénarios pour Villepin (top-10) ===")
    villepin = df[df["candidate"] == "villepin"].sort_values("p_victory", ascending=False).head(10)
    print(villepin[["exo_scenario", "shock_scenario", "score_1T_mean",
                    "p_qualif", "p_victory"]].to_string(index=False))
    villepin.to_csv(out_dir / "villepin_best_scenarios.csv", index=False)

    print("\n=== Probabilité moyenne par candidat (sur toutes les combinaisons) ===")
    agg = df.groupby("candidate")[["p_victory", "p_qualif", "score_1T_mean"]].mean().round(3)
    agg = agg.sort_values("p_victory", ascending=False)
    print(agg.to_string())
    agg.to_csv(out_dir / "winner_aggregate.csv")


if __name__ == "__main__":
    main()
