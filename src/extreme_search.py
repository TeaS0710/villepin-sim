"""Recherche extrême : optimise CONJOINTEMENT les 8 paramètres Tier 1 ET
les bases concurrents (dans leurs bornes historiques 2002-2022). C'est
l'exploration la plus large possible des conditions sous lesquelles le
modèle prédit une victoire Villepin : sans biais.

Méthode : CMA-ES multi-restart sur le simulateur direct (pas le NN, car
le NN n'inclut pas les bases concurrents comme inputs). Moins rapide mais
plus flexible.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cma
import numpy as np
import pandas as pd

from .parameters import TIER1_PARAMS
from .physical_model import all_candidates_2T_probabilities, load_config
from .winner_analysis import HISTORICAL_BASE_BOUNDS

# Ordre canonique : 8 Tier 1 + 5 bases concurrents
ARCH_ORDER = ("extreme_droite", "souverainiste", "droite_classique",
              "centre_gouv", "gauche_socdem", "gauche_radicale", "extreme_gauche")


def _bounds() -> tuple[list[float], list[float]]:
    lo = [0.0] * len(TIER1_PARAMS) + [HISTORICAL_BASE_BOUNDS[a][0] for a in ARCH_ORDER]
    hi = [1.0] * len(TIER1_PARAMS) + [HISTORICAL_BASE_BOUNDS[a][1] for a in ARCH_ORDER]
    return lo, hi


def _x_to_config(x: np.ndarray, base_cfg: dict) -> tuple[dict[str, float], dict]:
    """Décompose x en (params Tier 1, cfg patché avec nouvelles bases)."""
    params = {n: float(np.clip(x[i], 0, 1)) for i, n in enumerate(TIER1_PARAMS)}
    cfg = copy.deepcopy(base_cfg)
    for j, arch in enumerate(ARCH_ORDER):
        lo, hi = HISTORICAL_BASE_BOUNDS[arch]
        base = float(np.clip(x[len(TIER1_PARAMS) + j], lo, hi))
        cfg["competitors"][arch]["base"] = base
    return params, cfg


def fitness_villepin(x: np.ndarray, base_cfg: dict) -> float:
    """Objectif : maximiser P(victoire Villepin). Retourne -p_victory pour CMA-ES.

    Ajout d'un signal lisse quand Villepin n'est pas qualifié : on récompense
    aussi le score_1T de Villepin et on pénalise les scores des concurrents
    qui le dépassent. Ça évite que la fitness soit plate quand Villepin est
    en 3e ou 4e position (alors p_victory = 0 par défaut).
    """
    params, cfg = _x_to_config(x, base_cfg)
    probs = all_candidates_2T_probabilities(params, cfg)
    p_v = probs["villepin"]["p_victory"]
    # Signal smooth : si pas qualifié, on récompense la *proximité* à
    # la qualification (score relatif au 2ème top).
    villepin_score = probs["villepin"]["score_1T"]
    other_scores = sorted([d["score_1T"] for n, d in probs.items() if n != "villepin"],
                          reverse=True)
    rank2_score = other_scores[1] if len(other_scores) >= 2 else 0.0
    # Gap = villepin_score - rank2_score (positif si villepin top-2)
    gap = villepin_score - rank2_score
    # Fitness composite : p_victory dominant + bonus si gap (smoothed)
    soft_bonus = 0.001 * gap  # petite contribution, ne domine pas p_victory
    return -(p_v + soft_bonus)


def search(base_cfg: dict, n_restarts: int = 10, maxiter: int = 200,
           popsize: int = 50, seed: int = 42) -> pd.DataFrame:
    lo, hi = _bounds()
    n_dim = len(lo)
    all_results = []

    for run_idx in range(n_restarts):
        rng = np.random.default_rng(seed + run_idx)
        # Init aléatoire dans les bornes (avec margin pour éviter les coins)
        x0 = np.array([rng.uniform(lo[i] + 0.05 * (hi[i] - lo[i]),
                                   hi[i] - 0.05 * (hi[i] - lo[i]))
                       for i in range(n_dim)])
        sigma0 = 0.30   # v2 : sigma plus large pour mieux explorer les corners
        es = cma.CMAEvolutionStrategy(
            x0, sigma0,
            {"bounds": [lo, hi], "maxiter": maxiter, "popsize": popsize,
             "seed": seed + run_idx, "verbose": -9},
        )
        while not es.stop():
            sols = es.ask()
            fs = [fitness_villepin(s, base_cfg) for s in sols]
            es.tell(sols, fs)
        xbest = es.result.xbest if es.result.xbest is not None else es.result.xfavorite
        params, cfg_best = _x_to_config(xbest, base_cfg)
        probs = all_candidates_2T_probabilities(params, cfg_best)
        all_results.append({
            "run": run_idx,
            "p_victory_villepin": probs["villepin"]["p_victory"],
            "p_qualif_villepin": float(probs["villepin"]["qualified"]),
            "villepin_score_1T": probs["villepin"]["score_1T"],
            "villepin_opponent": probs["villepin"]["opponent"],
            **{f"param_{k}": v for k, v in params.items()},
            **{f"base_{a}": cfg_best["competitors"][a]["base"] for a in ARCH_ORDER},
            # scores d'autres candidats à l'optimum
            **{f"score_{c}": probs[c]["score_1T"] for c in probs},
        })

    return pd.DataFrame(all_results).sort_values("p_victory_villepin", ascending=False).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.fitted.yaml")
    ap.add_argument("--out", default="outputs/extreme_search.csv")
    ap.add_argument("--n-restarts", type=int, default=10)
    ap.add_argument("--maxiter", type=int, default=200)
    ap.add_argument("--popsize", type=int, default=50)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)

    print(f"Recherche extrême : {args.n_restarts} restarts × {args.maxiter} it × {args.popsize} pop")
    print(f"Dimension : {len(TIER1_PARAMS)} Tier 1 + {len(ARCH_ORDER)} bases = {len(TIER1_PARAMS) + len(ARCH_ORDER)}\n")

    df = search(cfg, n_restarts=args.n_restarts, maxiter=args.maxiter,
                popsize=args.popsize, seed=cfg.get("seed", 42))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print("=== Top 5 optima trouvés ===")
    # Colonnes dynamiques : base_* selon ARCH_ORDER (peut être v6 ou v8 multiclasses)
    base_cols = [f"base_{a}" for a in ARCH_ORDER if f"base_{a}" in df.columns]
    cols = (["run", "p_victory_villepin", "p_qualif_villepin", "villepin_score_1T",
             "villepin_opponent"]
            + base_cols[:2]
            + ["param_crisis", "param_central_collapse", "param_anti_extreme_pressure",
               "param_campaign_machine", "param_media_performance"])
    print(df.head(5)[cols].to_string(index=False))

    best = df.iloc[0]
    print(f"\n=== Meilleur scénario pour Villepin ===")
    print(f"P(victoire) = {best['p_victory_villepin']*100:.2f} %")
    print(f"Score 1T    = {best['villepin_score_1T']:.2f} %  vs {best['villepin_opponent']}")
    bases_str = ", ".join(f"{a}={best[f'base_{a}']:.1f}" for a in ARCH_ORDER)
    print(f"Bases concurrents : {bases_str}")
    print(f"Exogènes : crisis={best['param_crisis']:.2f}, central_collapse={best['param_central_collapse']:.2f},")
    print(f"           volatility={best['param_volatility']:.2f}, anti_extreme={best['param_anti_extreme_pressure']:.2f}")
    print(f"Internes : machine={best['param_campaign_machine']:.2f}, thematic={best['param_thematic_breadth']:.2f},")
    print(f"           media={best['param_media_performance']:.2f}, coalition={best['param_coalition_building']:.2f}")

    summary = {
        "best_p_victory_villepin": float(best["p_victory_villepin"]),
        "best_score_1T_villepin": float(best["villepin_score_1T"]),
        "best_opponent": best["villepin_opponent"],
        "bases_at_optimum": {a: float(best[f"base_{a}"]) for a in ARCH_ORDER},
        "params_at_optimum": {n: float(best[f"param_{n}"]) for n in TIER1_PARAMS},
        "n_restarts": args.n_restarts,
        "distinct_optima_p5_p95": [float(df["p_victory_villepin"].quantile(0.05)),
                                    float(df["p_victory_villepin"].quantile(0.95))],
    }
    (out.parent / "extreme_search_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n✓ {out}\n✓ {out.parent / 'extreme_search_summary.json'}")


if __name__ == "__main__":
    main()
