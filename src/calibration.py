"""Calibration du simulateur sur élections 2002-2022.

Procédure :
1. Pour chaque élection, configurer un *config snapshot* dérivé du config.yaml
   où les `competitors[arch]['base']` sont remplacés par les bases historiques
   agrégées (`aggregate_competitor_bases`), et où les paramètres Tier 1 sont
   ceux de `CONTEXT[year]`.
2. Lancer le simulateur → score Villepin-équivalent prédit.
3. Comparer au score historique réel.
4. Optimiser un *petit* sous-ensemble de paramètres globaux pour minimiser le
   MAE (avec régularisation ridge contre les priors). Sous-ensemble :
   - `mass_model.bias`
   - `mass_model.m_max`
   - `capture.villepin_scale`
   - `capture.volatility_softening`
5. Validation leave-one-out (LOO) : pour chaque élection, fit sur les 4
   autres, prédire celle-là.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .historical_context import (
    ARCHETYPE_MAPPING,
    CONTEXT,
    VILLEPIN_EQUIVALENT,
    aggregate_competitor_bases,
    get_actual_score,
    year_pools,
)
from .physical_model import first_round_scores, load_config

# Paramètres à optimiser : nom, accès dans le config (chemin), borne min, borne max
# v3 : ajout des paramètres du 2nd tour pour calibrer aussi l'accuracy 2T
# (le précédent 0/5 → 3/5 venait d'un boost_to_score=30 et sigmoid_scale=5
# choisis à dire d'expert sans fit).
PARAMS_TO_FIT = [
    ("mass_bias",            ("mass_model", "bias"),                 -3.0, 2.0),
    ("mass_m_max",           ("mass_model", "m_max"),                 1.5, 5.0),
    ("villepin_scale",       ("capture", "villepin_scale"),           1.0, 8.0),
    ("volatility_softening", ("capture", "volatility_softening"),     0.0, 1.0),
    ("second_round_scale",   ("second_round", "sigmoid_scale"),       2.0, 15.0),
    ("second_round_boost",   ("second_round", "boost_to_score"),      5.0, 40.0),
]


def _set_nested(d: dict, path: tuple[str, ...], value) -> None:
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = value


def _get_nested(d: dict, path: tuple[str, ...]):
    cur = d
    for k in path:
        cur = cur[k]
    return cur


def build_year_config(base_cfg: dict, year: int, hist_df: pd.DataFrame) -> dict:
    """Construit un config 'patché' pour une année historique donnée :
    - bases concurrents = bases agrégées par archétype (mapping rigoureux v2)
    - tailles de pools spécifiques à l'année (v2 : pool sizes calibrés)
    - contexte Tier 1 = CONTEXT[year]
    """
    cfg = copy.deepcopy(base_cfg)
    bases = aggregate_competitor_bases(hist_df, year)
    for arch, base in bases.items():
        if arch in cfg["competitors"]:
            cfg["competitors"][arch]["base"] = base
    pools = year_pools(year)
    if pools is not None:
        for pool_key, vals in pools.items():
            if pool_key in cfg["pools"]:
                cfg["pools"][pool_key]["size"] = vals["size"]
                cfg["pools"][pool_key]["inertia"] = vals["inertia"]
    return cfg


def predict_year(cfg_year: dict, year: int) -> float:
    """Prédit le score du Villepin-équivalent pour une année donnée."""
    params = dict(CONTEXT[year])
    scores = first_round_scores(params, cfg_year)
    return scores["villepin"]


def evaluate_all(base_cfg: dict, hist_df: pd.DataFrame, years: list[int] | None = None) -> pd.DataFrame:
    """Pour chaque année : score réel, prédit, erreur."""
    years = years or list(VILLEPIN_EQUIVALENT.keys())
    rows = []
    for y in years:
        cfg_y = build_year_config(base_cfg, y, hist_df)
        predicted = predict_year(cfg_y, y)
        actual = get_actual_score(hist_df, y)
        rows.append({
            "year": y,
            "candidate": VILLEPIN_EQUIVALENT[y],
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "error": round(predicted - actual, 2),
            "abs_error": round(abs(predicted - actual), 2),
        })
    return pd.DataFrame(rows)


def _params_from_cfg(cfg: dict) -> np.ndarray:
    return np.array([_get_nested(cfg, p[1]) for p in PARAMS_TO_FIT], dtype=np.float64)


def _apply_params_to_cfg(cfg: dict, x: np.ndarray) -> dict:
    new_cfg = copy.deepcopy(cfg)
    for (name, path, lo, hi), v in zip(PARAMS_TO_FIT, x):
        _set_nested(new_cfg, path, float(np.clip(v, lo, hi)))
    return new_cfg


def _2t_loss_for_year(cfg_y: dict, year: int) -> float:
    """Log-loss du résultat 2T pour une année : -log(P(vainqueur réel))
    selon le modèle. Si le vainqueur réel n'est pas qualifié dans la
    prédiction, applique une pénalité fixe (≈ -log(0.05)).
    """
    from .historical_validation import SECOND_ROUND_ACTUAL
    from .physical_model import all_candidates_2T_probabilities
    actual_winner = SECOND_ROUND_ACTUAL[year]["winner"]
    params = dict(CONTEXT[year])
    probs = all_candidates_2T_probabilities(params, cfg_y)
    if actual_winner not in probs or not probs[actual_winner]["qualified"]:
        return 3.0   # pénalité ≈ -log(0.05)
    p_actual = probs[actual_winner]["p_victory"]
    return -float(np.log(max(p_actual, 0.001)))


def fit(base_cfg: dict, hist_df: pd.DataFrame, train_years: list[int],
        ridge_lambda: float = 0.1, lambda_2t: float = 0.5,
        verbose: bool = False) -> tuple[dict, dict]:
    """Fitte les PARAMS_TO_FIT pour minimiser :
      MAE 1T sur train_years (sur Villepin-équivalent)
      + lambda_2t · log-loss du vainqueur 2T sur train_years
      + ridge_lambda · ||x - prior||² (régularisation)
    """
    prior = _params_from_cfg(base_cfg)
    bounds = [(p[2], p[3]) for p in PARAMS_TO_FIT]
    scales = np.array([hi - lo for _, _, lo, hi in PARAMS_TO_FIT])

    def objective(x: np.ndarray) -> float:
        cfg_x = _apply_params_to_cfg(base_cfg, x)
        errs_1t = []
        loss_2t = 0.0
        for y in train_years:
            cfg_y = build_year_config(cfg_x, y, hist_df)
            predicted = predict_year(cfg_y, y)
            actual = get_actual_score(hist_df, y)
            errs_1t.append(abs(predicted - actual))
            loss_2t += _2t_loss_for_year(cfg_y, y)
        mae = float(np.mean(errs_1t))
        loss_2t /= len(train_years)
        reg = ridge_lambda * float(np.sum(((x - prior) / scales) ** 2))
        total = mae + lambda_2t * loss_2t + reg
        if verbose:
            print(f"  x={np.round(x, 3)}  mae={mae:.3f}  2T_loss={loss_2t:.3f}  reg={reg:.3f}")
        return total

    x0 = prior.copy()
    res = minimize(
        objective, x0, method="L-BFGS-B", bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-5},
    )
    fitted = {p[0]: float(np.clip(res.x[i], p[2], p[3])) for i, p in enumerate(PARAMS_TO_FIT)}
    return fitted, _apply_params_to_cfg(base_cfg, res.x)


def evaluate_all_archetypes(base_cfg: dict, hist_df: pd.DataFrame) -> pd.DataFrame:
    """Test rigueur : prédire le score de TOUS les archétypes (pas seulement
    le Villepin-équivalent). Si la prédiction d'un archétype dévie fortement
    de la base agrégée qu'on lui a donnée en entrée, c'est que les dynamiques
    (mobile/stuck, flux concurrentiel) distordent excessivement le signal.

    On compare :
    - `base_assigned` : la base agrégée historique qu'on a injectée dans le simu
    - `predicted` : ce que `first_round_scores` retourne pour cet archétype
    - delta = predicted - base_assigned

    Un bon modèle a |delta| petit. Un grand delta = dynamiques pas neutres.
    """
    from .historical_context import aggregate_competitor_bases
    rows = []
    for y in VILLEPIN_EQUIVALENT.keys():
        cfg_y = build_year_config(base_cfg, y, hist_df)
        params = dict(CONTEXT[y])
        scores = first_round_scores(params, cfg_y)
        bases = aggregate_competitor_bases(hist_df, y)
        for arch, base in bases.items():
            if arch in scores:
                rows.append({
                    "year": y,
                    "archetype": arch,
                    "base_assigned": round(base, 2),
                    "predicted": round(scores[arch], 2),
                    "delta": round(scores[arch] - base, 2),
                    "abs_delta": round(abs(scores[arch] - base), 2),
                })
    return pd.DataFrame(rows)


def sensitivity_analysis(
    base_cfg: dict, hist_df: pd.DataFrame, n_samples: int = 100,
    perturbation: float = 0.20, seed: int = 42,
) -> pd.DataFrame:
    """Pour chaque élection historique, perturber le `CONTEXT[year]` de ±perturbation
    sur chaque paramètre, et calculer la distribution des scores prédits du
    Villepin-équivalent.

    Si l'écart-type des prédictions est grand (> 3 points), notre estimation
    de contexte est très subjective et le résultat instable. Si petit, le
    modèle est robuste au choix de contexte.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for y in VILLEPIN_EQUIVALENT.keys():
        cfg_y = build_year_config(base_cfg, y, hist_df)
        ctx_base = CONTEXT[y]
        preds = []
        for _ in range(n_samples):
            ctx_perturbed = {
                k: float(np.clip(v + rng.uniform(-perturbation, perturbation), 0.0, 1.0))
                for k, v in ctx_base.items()
            }
            scores = first_round_scores(ctx_perturbed, cfg_y)
            preds.append(scores["villepin"])
        preds = np.array(preds)
        actual = get_actual_score(hist_df, y)
        rows.append({
            "year": y,
            "candidate": VILLEPIN_EQUIVALENT[y],
            "actual": round(actual, 2),
            "predicted_mean": round(float(preds.mean()), 2),
            "predicted_std": round(float(preds.std()), 2),
            "predicted_p5": round(float(np.percentile(preds, 5)), 2),
            "predicted_p95": round(float(np.percentile(preds, 95)), 2),
            "in_range": bool((np.percentile(preds, 5) <= actual <= np.percentile(preds, 95))),
        })
    return pd.DataFrame(rows)


def leave_one_out(base_cfg: dict, hist_df: pd.DataFrame, ridge_lambda: float = 0.1) -> pd.DataFrame:
    years = list(VILLEPIN_EQUIVALENT.keys())
    rows = []
    for y_test in years:
        train_years = [y for y in years if y != y_test]
        fitted, cfg_fit = fit(base_cfg, hist_df, train_years, ridge_lambda=ridge_lambda)
        cfg_y = build_year_config(cfg_fit, y_test, hist_df)
        predicted = predict_year(cfg_y, y_test)
        actual = get_actual_score(hist_df, y_test)
        rows.append({
            "year_held_out": y_test,
            "candidate": VILLEPIN_EQUIVALENT[y_test],
            "actual": round(actual, 2),
            "predicted": round(predicted, 2),
            "abs_error": round(abs(predicted - actual), 2),
            **{k: round(v, 3) for k, v in fitted.items()},
        })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default="data/historical_elections.csv")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--ridge", type=float, default=0.1)
    ap.add_argument("--write-fitted-config", action="store_true",
                    help="Si présent, sauvegarde un config.fitted.yaml avec les valeurs ajustées sur TOUTES les années.")
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    hist_df = pd.read_csv(args.data)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== Audit asymétrie : prédiction de tous les archétypes ===")
    arch_df = evaluate_all_archetypes(base_cfg, hist_df)
    print(arch_df.to_string(index=False))
    arch_mae = arch_df["abs_delta"].mean()
    arch_max = arch_df["abs_delta"].max()
    print(f"MAE archétypes : {arch_mae:.2f}  |  max |delta| : {arch_max:.2f}")
    arch_df.to_csv(out / "audit_archetypes.csv", index=False)

    print("\n=== Sensibilité au contexte (perturbation ±0.20) ===")
    sens_df = sensitivity_analysis(base_cfg, hist_df, n_samples=100, perturbation=0.20)
    print(sens_df.to_string(index=False))
    coverage = sens_df["in_range"].mean()
    print(f"Couverture (réel dans p5-p95) : {coverage*100:.0f}%")
    sens_df.to_csv(out / "audit_sensitivity.csv", index=False)

    print("\n=== Baseline (priors, pas de fit) ===")
    base_df = evaluate_all(base_cfg, hist_df)
    print(base_df.to_string(index=False))
    print(f"MAE baseline : {base_df['abs_error'].mean():.3f}")

    print("\n=== Fit sur TOUTES les années (in-sample) ===")
    fitted_params, fitted_cfg = fit(base_cfg, hist_df, list(VILLEPIN_EQUIVALENT.keys()), ridge_lambda=args.ridge)
    print("Paramètres fittés :")
    for k, v in fitted_params.items():
        prior = _get_nested(base_cfg, dict(zip([p[0] for p in PARAMS_TO_FIT], [p[1] for p in PARAMS_TO_FIT]))[k])
        print(f"  {k:25s}  prior={prior:.3f}  fitted={v:.3f}")
    in_sample_df = evaluate_all(fitted_cfg, hist_df)
    print(in_sample_df.to_string(index=False))
    print(f"MAE in-sample : {in_sample_df['abs_error'].mean():.3f}")

    print("\n=== Leave-one-out validation ===")
    loo_df = leave_one_out(base_cfg, hist_df, ridge_lambda=args.ridge)
    print(loo_df.to_string(index=False))
    print(f"MAE LOO : {loo_df['abs_error'].mean():.3f}")

    base_df.to_csv(out / "calibration_baseline.csv", index=False)
    in_sample_df.to_csv(out / "calibration_in_sample.csv", index=False)
    loo_df.to_csv(out / "calibration_loo.csv", index=False)
    with open(out / "calibration_summary.json", "w") as f:
        json.dump({
            "mae_baseline": float(base_df["abs_error"].mean()),
            "mae_in_sample": float(in_sample_df["abs_error"].mean()),
            "mae_loo": float(loo_df["abs_error"].mean()),
            "archetype_mae": float(arch_df["abs_delta"].mean()),
            "archetype_max_delta": float(arch_df["abs_delta"].max()),
            "sensitivity_coverage_pct": float(sens_df["in_range"].mean() * 100),
            "fitted_params": fitted_params,
            "ridge_lambda": args.ridge,
        }, f, indent=2)
    print(f"\n✓ Résultats sauvegardés dans {out}/")

    if args.write_fitted_config:
        import yaml
        fitted_path = Path(args.config).with_name("config.fitted.yaml")
        with open(fitted_path, "w") as f:
            yaml.safe_dump(fitted_cfg, f, sort_keys=False, allow_unicode=True)
        print(f"✓ Config fitté écrit : {fitted_path}")


if __name__ == "__main__":
    main()
