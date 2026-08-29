"""Simulateur électoral physique-statistique.

Différences vs brief original :
- La masse Villepin est *additive + sigmoïde*, pas multiplicative.
  La forme multiplicative crée un optimum trivial ("tout à 1") et une zone morte
  ("un seul facteur à 0 -> masse nulle"), ce qui rend l'optimisation inintéressante.
- Les poids de masse sont lus depuis config.yaml : ils seront recalibrés par
  ridge regression sur élections 2002-2022 (cf calibration.py).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .parameters import TIER1_PARAMS


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_config(path: str | Path = "config.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class SimResult:
    scores_1T: dict[str, float]    # score 1er tour par candidat (Villepin + competitors)
    qualified: bool                 # Villepin dans le top 2 ?
    opponent: str | None            # nom de l'adversaire au 2T, ou None
    p_victory_2T: float             # probabilité victoire 2T (0 si pas qualifié)


# ----------------------------------------------------------------------
# Masse Villepin (additive + sigmoïde, sans optimum trivial)
# ----------------------------------------------------------------------
def compute_villepin_mass(params: dict[str, float], cfg: dict) -> float:
    """Calcule la 'masse gravitationnelle' de Villepin.

    Forme : m = m_min + (m_max - m_min) * sigmoid( Σ wi (xi - 0.5) + bias )

    v2 : itère sur TOUTES les clés présentes dans `mass_model.weights`,
    pas seulement les Tier 1. Les Tier 2 (ajoutés par le LLM) contribuent
    additivement à l'argument du sigmoïde. Les params absents de `params`
    (ex : Tier 2 d'une version antérieure) sont silencieusement ignorés.
    """
    mm = cfg["mass_model"]
    w = mm["weights"]
    z = mm["bias"]
    for name, weight in w.items():
        if name in params:
            z += weight * (params[name] - 0.5)
    s = _sigmoid(z)
    return mm["m_min"] + (mm["m_max"] - mm["m_min"]) * s


# ----------------------------------------------------------------------
# Capture concurrentielle (Bradley-Terry adapté)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Effondrement central : transfère une fraction du pool central -> indécis
# ----------------------------------------------------------------------
def _apply_central_collapse(pools: dict, params: dict, cfg: dict) -> dict:
    cc = params["central_collapse"]
    if cc <= 0:
        return pools
    cc_cfg = cfg["central_collapse"]
    ratio = cc_cfg["transfer_ratio"]
    # source_pool : nom du pool à effondrer (defaut "central" pour rétrocompat,
    # "centre_gouv" pour les configs multiclasses).
    src = cc_cfg.get("source_pool", "central")
    if src not in pools:
        return pools
    new = {k: dict(v) for k, v in pools.items()}
    transfer = new[src]["size"] * cc * ratio
    new[src]["size"] = max(0.0, new[src]["size"] - transfer)
    new["indecis"]["size"] += transfer
    return new


def _pool_pulls(
    pool_key: str,
    villepin_mass: float,
    villepin_aff: dict,
    competitors: dict,
    villepin_scale: float,
) -> tuple[dict[str, float], dict[str, float]]:
    """Pour un pool donné, retourne (pulls, affinités positives) des candidats."""
    pulls: dict[str, float] = {}
    affinities: dict[str, float] = {}

    v_aff = villepin_aff.get(pool_key, 0.0)
    if v_aff > 0:
        pulls["villepin"] = villepin_mass * v_aff * villepin_scale
        affinities["villepin"] = v_aff

    for cname, comp in competitors.items():
        aff = comp["affinity"].get(pool_key, 0.0)
        if aff > 0:
            pulls[cname] = comp["base"] * aff
            affinities[cname] = aff

    return pulls, affinities


# ----------------------------------------------------------------------
# Score 1er tour : modèle mobile/stuck par pool
# ----------------------------------------------------------------------
def first_round_scores(params: dict[str, float], cfg: dict) -> dict[str, float]:
    """Distribue chaque pool entre les candidats à affinité positive.

    Pour chaque pool :
    - `effective_mobility` = (1-inertie) + softening * volatility * inertie
      (la volatility globale ramollit l'inertie de tous les pools)
    - fraction "stuck" = pool * (1 - effective_mobility) -> attribuée au
      "propriétaire naturel" du pool (plus haute affinité positive)
    - fraction "mobile" = pool * effective_mobility -> répartie au prorata
      des pulls Bradley-Terry (mass × affinité × scale)
    """
    pools = _apply_central_collapse(cfg["pools"], params, cfg)
    competitors = cfg["competitors"]
    villepin_aff = cfg["villepin_affinity"]
    villepin_scale = cfg["capture"]["villepin_scale"]
    softening = cfg["capture"]["volatility_softening"]

    mass = compute_villepin_mass(params, cfg)
    volatility = params["volatility"]

    scores: dict[str, float] = {"villepin": 0.0}
    for cname in competitors:
        scores[cname] = 0.0

    for pool_key, pool in pools.items():
        pulls, affinities = _pool_pulls(
            pool_key, mass, villepin_aff, competitors, villepin_scale,
        )
        if not pulls:
            continue

        # Mobility (volatility ramollit l'inertie)
        eff_mob = (1.0 - pool["inertia"]) + softening * volatility * pool["inertia"]
        eff_mob = min(1.0, eff_mob)
        stuck = pool["size"] * (1.0 - eff_mob)
        mobile = pool["size"] * eff_mob

        # Stuck -> natural owner (plus haute affinité positive)
        natural_owner = max(affinities, key=affinities.get)
        scores[natural_owner] += stuck

        # Mobile -> Bradley-Terry sur pulls
        total_pull = sum(pulls.values())
        if total_pull > 0:
            for name, pull in pulls.items():
                scores[name] += mobile * pull / total_pull

    # Cap Villepin pour réalisme
    caps = cfg["score_caps"]
    scores["villepin"] = float(np.clip(scores["villepin"], caps["min"], caps["max"]))
    return scores


def first_round_scores_breakdown(
    params: dict[str, float], cfg: dict,
) -> dict[str, dict[str, float]]:
    """Variante de `first_round_scores` qui retourne {candidat: {pool: contribution}}.
    Utile pour des plots de décomposition (d'où vient chaque score).
    """
    pools = _apply_central_collapse(cfg["pools"], params, cfg)
    competitors = cfg["competitors"]
    villepin_aff = cfg["villepin_affinity"]
    villepin_scale = cfg["capture"]["villepin_scale"]
    softening = cfg["capture"]["volatility_softening"]
    mass = compute_villepin_mass(params, cfg)
    volatility = params["volatility"]

    breakdown: dict[str, dict[str, float]] = {"villepin": {}}
    for cname in competitors:
        breakdown[cname] = {}
    for pool_key in pools:
        for name in breakdown:
            breakdown[name][pool_key] = 0.0

    for pool_key, pool in pools.items():
        pulls, affinities = _pool_pulls(
            pool_key, mass, villepin_aff, competitors, villepin_scale,
        )
        if not pulls:
            continue
        eff_mob = min(1.0, (1.0 - pool["inertia"]) + softening * volatility * pool["inertia"])
        stuck = pool["size"] * (1.0 - eff_mob)
        mobile = pool["size"] * eff_mob
        natural_owner = max(affinities, key=affinities.get)
        breakdown[natural_owner][pool_key] += stuck
        total_pull = sum(pulls.values())
        if total_pull > 0:
            for name, pull in pulls.items():
                breakdown[name][pool_key] += mobile * pull / total_pull
    return breakdown


# ----------------------------------------------------------------------
# Second tour
# ----------------------------------------------------------------------
def second_round_probability(
    scores_1T: dict[str, float], params: dict[str, float], cfg: dict
) -> tuple[bool, str | None, float]:
    """Retourne (qualifié, adversaire, P(victoire 2T))."""
    ranked = sorted(scores_1T.items(), key=lambda kv: kv[1], reverse=True)
    top2 = [name for name, _ in ranked[:2]]
    if "villepin" not in top2:
        return False, None, 0.0
    opponent = top2[0] if top2[1] == "villepin" else top2[1]
    villepin_score = scores_1T["villepin"]
    opp_score = scores_1T[opponent]

    boost_cfg = cfg["second_round"]["report_boost"]
    bc = boost_cfg.get(opponent, boost_cfg["default"])
    boost = float(np.clip(bc["base"] + bc["slope"] * params["anti_extreme_pressure"], 0.0, 0.95))

    bts = cfg["second_round"]["boost_to_score"]
    scale = cfg["second_round"]["sigmoid_scale"]
    advantage = (villepin_score + boost * bts) - opp_score
    p = float(_sigmoid(advantage / scale))
    return True, opponent, p


# ----------------------------------------------------------------------
# Probabilité de victoire 2T pour TOUS les candidats
# ----------------------------------------------------------------------
# Classification approximative des candidats selon leur affinité dominante
# au pool extrême-droite (rn). Sert à orienter le boost report.
_CAMP_EXTREME = {"bardella": 1.0, "melenchon": 0.6, "philippe": 0.0,
                 "retailleau": 0.2, "glucksmann": 0.1, "villepin": 0.1}


def _pair_winner_prob(
    a_name: str, a_score: float, b_name: str, b_score: float,
    params: dict[str, float], cfg: dict,
) -> float:
    """P(a bat b en 2T), symétrique : P(a) + P(b) = 1.

    Boost signé pour `a` = `+0.45 · aep · (extremisme(b) - extremisme(a))`.
    Le terme constant 0.10 du brief est volontairement omis pour préserver
    la symétrie (sinon P(a) + P(b) > 1 quand extremismes sont opposés).
    """
    extreme_a = _CAMP_EXTREME.get(a_name, 0.1)
    extreme_b = _CAMP_EXTREME.get(b_name, 0.1)
    aep = params["anti_extreme_pressure"]
    bts = cfg["second_round"]["boost_to_score"]
    scale = cfg["second_round"]["sigmoid_scale"]
    delta_ext = extreme_b - extreme_a
    boost = 0.45 * aep * delta_ext
    advantage = (a_score + boost * bts) - b_score
    return float(_sigmoid(advantage / scale))


def all_candidates_2T_probabilities(
    params: dict[str, float], cfg: dict,
) -> dict[str, dict[str, float | str | None]]:
    """Pour une configuration donnée, calcule pour chaque candidat :
    - son score 1T
    - s'il est dans le top-2 (qualifié)
    - son adversaire au 2T si qualifié
    - sa P(victoire 2T) si qualifié, 0 sinon
    """
    scores = first_round_scores(params, cfg)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top2 = {ranked[0][0], ranked[1][0]}
    out: dict[str, dict] = {}
    for name, sc in scores.items():
        if name in top2:
            opp = [n for n in top2 if n != name][0]
            opp_score = scores[opp]
            p_win = _pair_winner_prob(name, sc, opp, opp_score, params, cfg)
        else:
            opp, p_win = None, 0.0
        out[name] = {
            "score_1T": float(sc),
            "qualified": name in top2,
            "opponent": opp,
            "p_victory": float(p_win),
        }
    return out


# ----------------------------------------------------------------------
# Simulation déterministe (un point)
# ----------------------------------------------------------------------
def simulate_once(params: dict[str, float], cfg: dict) -> SimResult:
    scores = first_round_scores(params, cfg)
    qualified, opponent, p2T = second_round_probability(scores, params, cfg)
    return SimResult(
        scores_1T=scores,
        qualified=qualified,
        opponent=opponent,
        p_victory_2T=p2T,
    )


# ----------------------------------------------------------------------
# Simulation Monte Carlo (bruit sur les paramètres)
# ----------------------------------------------------------------------
def simulate_monte_carlo(
    params: dict[str, float],
    cfg: dict,
    n_samples: int | None = None,
    noise_std: float | None = None,
    rng: np.random.Generator | None = None,
) -> dict[str, Any]:
    if n_samples is None:
        n_samples = cfg["pipeline"]["monte_carlo"]["n_mc_per_sample"]
    if noise_std is None:
        noise_std = cfg["pipeline"]["monte_carlo"]["noise_std"]
    if rng is None:
        rng = np.random.default_rng(cfg.get("seed", 42))

    villepin_scores = np.empty(n_samples, dtype=np.float64)
    qualified_flags = np.empty(n_samples, dtype=bool)
    p_victories = np.empty(n_samples, dtype=np.float64)
    opponents: list[str | None] = []

    for i in range(n_samples):
        noisy = {
            name: float(np.clip(v + rng.normal(0, noise_std), 0.0, 1.0))
            for name, v in params.items()
        }
        res = simulate_once(noisy, cfg)
        villepin_scores[i] = res.scores_1T["villepin"]
        qualified_flags[i] = res.qualified
        p_victories[i] = res.p_victory_2T
        opponents.append(res.opponent)

    return {
        "score_1T_mean": float(villepin_scores.mean()),
        "score_1T_median": float(np.median(villepin_scores)),
        "score_1T_p5": float(np.percentile(villepin_scores, 5)),
        "score_1T_p95": float(np.percentile(villepin_scores, 95)),
        "p_qualif": float(qualified_flags.mean()),
        "p_victory": float(p_victories.mean()),
        "n_samples": n_samples,
    }
