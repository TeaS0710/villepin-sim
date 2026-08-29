"""CMA-ES multi-restart + clustering KMeans pour découvrir des archétypes
de stratégies.

Trois modes :
- `internal_only`  : optimise UNIQUEMENT les paramètres internes (campagne),
                     les exogènes sont fixés (cf scénario passé en arg).
                     C'est le mode "actionnable" : que peut faire la campagne ?
- `all_params`     : optimise les 8 paramètres. Mode "upper bound" : tells
                     us le plafond théorique.
- `scenario_sweep` : pour chaque scénario exogène prédéfini, optimise
                     uniquement les internes. Donne front Pareto context × proba.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cma
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .neural_predictor import load_model, predict_batch
from .parameters import EXOGENOUS, INTERNAL, TIER1_PARAMS
from .physical_model import load_config

# Scénarios exogènes prédéfinis : (crisis, central_collapse, volatility, anti_extreme_pressure)
EXOGENOUS_SCENARIOS: dict[str, dict[str, float]] = {
    "calme":            {"crisis": 0.2, "central_collapse": 0.1, "volatility": 0.3, "anti_extreme_pressure": 0.3},
    "median":           {"crisis": 0.5, "central_collapse": 0.5, "volatility": 0.5, "anti_extreme_pressure": 0.5},
    "tempete_2017":     {"crisis": 0.6, "central_collapse": 0.85, "volatility": 0.75, "anti_extreme_pressure": 0.65},
    "crise_pure":       {"crisis": 0.95, "central_collapse": 0.4, "volatility": 0.8, "anti_extreme_pressure": 0.7},
    "vide_central":     {"crisis": 0.3, "central_collapse": 0.95, "volatility": 0.6, "anti_extreme_pressure": 0.6},
    "tout_max":         {"crisis": 1.0, "central_collapse": 1.0, "volatility": 1.0, "anti_extreme_pressure": 1.0},
}


@dataclass
class OptimizationResult:
    scenario: str
    best_params: dict[str, float]
    best_p_victory: float
    best_score_1T: float
    best_p_qualif: float
    n_evals: int
    history_best: list[float]


def _apply_constraints(params: dict[str, float], cfg: dict) -> dict[str, float]:
    """Renvoie une copie clippée des params selon les contraintes réalistes."""
    c = cfg["constraints"]
    p = dict(params)
    # coalition <= 0.6 si machine < 0.5
    if p["campaign_machine"] < c["coalition_needs_machine"]["threshold_machine"]:
        p["coalition_building"] = min(p["coalition_building"], c["coalition_needs_machine"]["cap_coalition_if_below"])
    # thematic <= 0.4 si machine < 0.3
    if p["campaign_machine"] < c["thematic_needs_machine"]["threshold_machine"]:
        p["thematic_breadth"] = min(p["thematic_breadth"], c["thematic_needs_machine"]["cap_thematic_if_below"])
    # budget interne <= 3.0
    internals = c["budget_internal"]["params"]
    total = sum(p[k] for k in internals)
    max_sum = c["budget_internal"]["max_sum"]
    if total > max_sum:
        factor = max_sum / total
        for k in internals:
            p[k] = p[k] * factor
    return p


def optimize(
    model,
    cfg: dict,
    scenario_name: str,
    x_cols: list[str],
    fixed_exogenous: dict[str, float] | None = None,
    optimize_exogenous: bool = False,
    sigma0: float | None = None,
    popsize: int | None = None,
    maxiter: int | None = None,
    seed: int = 42,
) -> OptimizationResult:
    """Optimise par CMA-ES. Si `fixed_exogenous` est donné et que
    `optimize_exogenous` est False, on ne cherche que sur les paramètres internes.
    """
    cmaes_cfg = cfg["pipeline"]["cmaes"]
    sigma0 = sigma0 if sigma0 is not None else cmaes_cfg["sigma0"]
    popsize = popsize if popsize is not None else cmaes_cfg["popsize"]
    maxiter = maxiter if maxiter is not None else cmaes_cfg["maxiter"]

    if optimize_exogenous or fixed_exogenous is None:
        free_names = list(TIER1_PARAMS)
        fixed = {}
    else:
        free_names = [n for n in TIER1_PARAMS if n not in fixed_exogenous]
        fixed = dict(fixed_exogenous)

    n_dim = len(free_names)
    history_best = []
    n_evals = 0

    def to_full_params(x: np.ndarray) -> dict[str, float]:
        x = np.clip(x, 0.0, 1.0)
        out = {name: float(x[i]) for i, name in enumerate(free_names)}
        out.update(fixed)
        # Pour les Tier 2 (si NN en a) : valeur neutre 0.5
        for col in x_cols:
            out.setdefault(col, 0.5)
        return _apply_constraints(out, cfg)

    def fitness_batch(xs: list[np.ndarray]) -> list[float]:
        nonlocal n_evals
        params_list = [to_full_params(x) for x in xs]
        X = np.array([[p[n] for n in x_cols] for p in params_list])
        pred = predict_batch(model, X)
        n_evals += len(xs)
        return [-float(p[2]) for p in pred]

    x0 = np.full(n_dim, 0.5)
    es = cma.CMAEvolutionStrategy(
        x0, sigma0,
        {"bounds": [[0.0] * n_dim, [1.0] * n_dim],
         "maxiter": maxiter, "popsize": popsize,
         "seed": seed, "verbose": -9},
    )
    while not es.stop():
        solutions = es.ask()
        fs = fitness_batch(solutions)
        es.tell(solutions, fs)
        history_best.append(-min(fs))

    xbest = es.result.xbest
    if xbest is None:
        xbest = es.result.xfavorite
    best_params = to_full_params(xbest)
    X = np.array([[best_params[n] for n in x_cols]])
    pred = predict_batch(model, X)[0]
    return OptimizationResult(
        scenario=scenario_name,
        best_params=best_params,
        best_p_victory=float(pred[2]),
        best_score_1T=float(pred[0]),
        best_p_qualif=float(pred[1]),
        n_evals=n_evals,
        history_best=history_best,
    )


def multi_restart_collect_top(
    model, cfg: dict, scenario_name: str,
    x_cols: list[str],
    fixed_exogenous: dict[str, float] | None = None,
    n_restarts: int = None, top_k_per_run: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Lance n_restarts CMA-ES avec seeds différents, collecte top-k candidats
    par run pour clustering ultérieur.
    """
    n_restarts = n_restarts if n_restarts is not None else cfg["pipeline"]["cmaes"]["n_restarts"]
    cmaes_cfg = cfg["pipeline"]["cmaes"]
    if fixed_exogenous is None:
        fixed_exogenous = {}
        free_names = list(TIER1_PARAMS)
    else:
        free_names = [n for n in TIER1_PARAMS if n not in fixed_exogenous]

    rows = []
    for run_idx in range(n_restarts):
        rng = np.random.default_rng(seed + run_idx)
        x0 = rng.uniform(0.2, 0.8, size=len(free_names))
        es = cma.CMAEvolutionStrategy(
            x0, cmaes_cfg["sigma0"],
            {"bounds": [[0.0] * len(free_names), [1.0] * len(free_names)],
             "maxiter": cmaes_cfg["maxiter"], "popsize": cmaes_cfg["popsize"],
             "seed": seed + run_idx, "verbose": -9},
        )
        seen_solutions = []
        while not es.stop():
            sols = es.ask()
            params_list = []
            X_eval = []
            for x in sols:
                params = {n: float(np.clip(x[i], 0, 1)) for i, n in enumerate(free_names)}
                params.update(fixed_exogenous)
                for col in x_cols:
                    params.setdefault(col, 0.5)
                params = _apply_constraints(params, cfg)
                params_list.append(params)
                X_eval.append([params[n] for n in x_cols])
            preds = predict_batch(model, np.array(X_eval))
            fs = [-float(p[2]) for p in preds]
            es.tell(sols, fs)
            for params, pred in zip(params_list, preds):
                seen_solutions.append((params, float(pred[2]), float(pred[0]), float(pred[1])))
        seen_solutions.sort(key=lambda x: -x[1])
        for params, pv, s1t, pq in seen_solutions[:top_k_per_run]:
            rows.append({**params, "p_victory": pv, "score_1T": s1t, "p_qualif": pq, "run": run_idx})

    df = pd.DataFrame(rows).sort_values("p_victory", ascending=False).reset_index(drop=True)
    return df


def cluster_archetypes(df_top: pd.DataFrame, k: int, free_names: list[str], seed: int = 42) -> pd.DataFrame:
    """KMeans sur les top candidats. Retourne (centroides, top par cluster)."""
    X = df_top[free_names].values
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
    df_top = df_top.copy()
    df_top["cluster"] = km.labels_
    archetypes = []
    for c in range(k):
        sub = df_top[df_top["cluster"] == c].sort_values("p_victory", ascending=False)
        best = sub.iloc[0]
        centroid = sub[free_names].mean()
        archetypes.append({
            "cluster": c,
            "n_members": len(sub),
            "best_p_victory": float(best["p_victory"]),
            "best_score_1T": float(best["score_1T"]),
            "mean_p_victory": float(sub["p_victory"].mean()),
            **{f"centroid_{n}": float(centroid[n]) for n in free_names},
            **{f"best_{n}": float(best[n]) for n in free_names},
        })
    arche_df = pd.DataFrame(archetypes).sort_values("best_p_victory", ascending=False).reset_index(drop=True)
    return arche_df, df_top


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.fitted.yaml")
    ap.add_argument("--model", default="outputs/checkpoints/nn_surrogate.pt")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--mode", choices=["scenario_sweep", "all_params"], default="scenario_sweep")
    ap.add_argument("--n-restarts", type=int, default=None)
    ap.add_argument("--quick", action="store_true", help="moins de restarts et d'itérations")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)
    model, x_cols = load_model(args.model)
    # Sécurité : on n'optimise que sur les Tier 1 (les Tier 2 viennent de
    # l'extension LLM, conservés à leur valeur médiane si présents)
    if x_cols != list(TIER1_PARAMS):
        print(f"[info] NN a {len(x_cols)} inputs (Tier 1 + {len(x_cols)-len(TIER1_PARAMS)} Tier 2). "
              f"CMA-ES n'optimise que sur Tier 1 ; Tier 2 fixés à 0.5.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.quick:
        cfg["pipeline"]["cmaes"]["maxiter"] = 80
        cfg["pipeline"]["cmaes"]["popsize"] = 30
        cfg["pipeline"]["cmaes"]["n_restarts"] = 5

    results: list[dict] = []
    if args.mode == "scenario_sweep":
        print("=== Scenario sweep (chaque scénario exogène → optimisation interne) ===")
        for name, exo in EXOGENOUS_SCENARIOS.items():
            r = optimize(model, cfg, name, x_cols=x_cols, fixed_exogenous=exo,
                         optimize_exogenous=False, seed=42)
            results.append({
                "scenario": name,
                **exo,
                "best_p_victory": round(r.best_p_victory, 4),
                "best_score_1T": round(r.best_score_1T, 2),
                "best_p_qualif": round(r.best_p_qualif, 4),
                **{f"best_{k}": round(v, 3) for k, v in r.best_params.items() if k not in exo},
            })
            print(f"  [{name:15s}] p_victory={r.best_p_victory:.4f}  score_1T={r.best_score_1T:.2f}")
        df_scenarios = pd.DataFrame(results)
        df_scenarios.to_csv(out_dir / "cmaes_scenarios.csv", index=False)
        print(f"\n✓ outputs/cmaes_scenarios.csv ({len(df_scenarios)} scénarios)")

    print("\n=== Multi-restart pour clustering archétypes (scénario tempête_2017) ===")
    fixed_exo = EXOGENOUS_SCENARIOS["tempete_2017"]
    top_df = multi_restart_collect_top(
        model, cfg, "tempete_2017",
        x_cols=x_cols,
        fixed_exogenous=fixed_exo, n_restarts=args.n_restarts or cfg["pipeline"]["cmaes"]["n_restarts"],
    )
    top_df.to_csv(out_dir / "cmaes_top_candidates.csv", index=False)
    print(f"  {len(top_df)} candidats collectés au total")

    k = cfg["pipeline"]["clustering"]["k"]
    free_names = [n for n in TIER1_PARAMS if n not in fixed_exo]
    archetypes, labeled = cluster_archetypes(top_df, k=k, free_names=free_names)
    archetypes.to_csv(out_dir / "archetypes.csv", index=False)
    labeled.to_csv(out_dir / "cmaes_top_candidates_labeled.csv", index=False)
    print(f"\n=== Top {k} archétypes ===")
    print(archetypes[["cluster", "n_members", "best_p_victory", "mean_p_victory"]].to_string(index=False))


if __name__ == "__main__":
    main()
