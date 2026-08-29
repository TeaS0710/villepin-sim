"""Tests de cohérence du simulateur. Pas de couverture exhaustive — on cible
les invariants qui, s'ils sont cassés, invalident toute analyse en aval.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.physical_model import (  # noqa: E402
    compute_villepin_mass,
    first_round_scores,
    load_config,
    second_round_probability,
    simulate_monte_carlo,
    simulate_once,
)
from src.parameters import TIER1_PARAMS  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


@pytest.fixture
def neutral_params():
    return {name: 0.5 for name in TIER1_PARAMS}


@pytest.fixture
def all_ones():
    return {name: 1.0 for name in TIER1_PARAMS}


@pytest.fixture
def all_zeros():
    return {name: 0.0 for name in TIER1_PARAMS}


# ---------------------------------------------------------------- masse
class TestMass:
    def test_bounded(self, cfg, neutral_params, all_ones, all_zeros):
        mm = cfg["mass_model"]
        for p in (neutral_params, all_ones, all_zeros):
            m = compute_villepin_mass(p, cfg)
            assert mm["m_min"] <= m <= mm["m_max"], f"masse hors bornes pour {p}"

    def test_monotone_in_crisis(self, cfg, neutral_params):
        params_low = {**neutral_params, "crisis": 0.0}
        params_high = {**neutral_params, "crisis": 1.0}
        assert compute_villepin_mass(params_high, cfg) > compute_villepin_mass(params_low, cfg)

    def test_no_runaway(self, cfg, all_ones):
        # Pour invalider la formule multiplicative pathologique : la masse
        # à 1 partout ne doit pas être >> m_max ni 100x plus que la baseline.
        m_max = cfg["mass_model"]["m_max"]
        m_ones = compute_villepin_mass(all_ones, cfg)
        m_neutral = compute_villepin_mass({n: 0.5 for n in TIER1_PARAMS}, cfg)
        assert m_ones <= m_max + 1e-6
        assert m_ones < 50 * m_neutral, "explosion non bornée détectée"


# ---------------------------------------------------------------- scores 1T
class TestFirstRound:
    def test_scores_present(self, cfg, neutral_params):
        scores = first_round_scores(neutral_params, cfg)
        expected = {"villepin", "bardella", "philippe", "melenchon", "retailleau", "glucksmann"}
        assert set(scores) == expected

    def test_scores_sum_reasonable(self, cfg, neutral_params):
        scores = first_round_scores(neutral_params, cfg)
        s = sum(scores.values())
        # Σ scores doit être proche de la somme des pools (~100% des votants),
        # avec une marge pour les bornes basses/hautes.
        assert 80.0 < s < 102.0, f"somme scores irréaliste : {s:.2f}"

    def test_villepin_score_in_range(self, cfg, neutral_params):
        scores = first_round_scores(neutral_params, cfg)
        assert 0.5 <= scores["villepin"] <= 38.0

    def test_villepin_increases_with_crisis(self, cfg, neutral_params):
        low = first_round_scores({**neutral_params, "crisis": 0.0}, cfg)
        high = first_round_scores({**neutral_params, "crisis": 1.0}, cfg)
        assert high["villepin"] > low["villepin"]

    def test_villepin_increases_with_machine(self, cfg, neutral_params):
        low = first_round_scores({**neutral_params, "campaign_machine": 0.0}, cfg)
        high = first_round_scores({**neutral_params, "campaign_machine": 1.0}, cfg)
        assert high["villepin"] > low["villepin"]

    def test_central_collapse_transfers(self, cfg, neutral_params):
        # Effondrement central -> Philippe perd ; les autres peuvent gagner.
        no_collapse = first_round_scores({**neutral_params, "central_collapse": 0.0}, cfg)
        full_collapse = first_round_scores({**neutral_params, "central_collapse": 1.0}, cfg)
        assert full_collapse["philippe"] < no_collapse["philippe"]


# ---------------------------------------------------------------- second tour
class TestSecondRound:
    def test_not_qualified_returns_zero(self, cfg):
        scores = {"villepin": 3.0, "bardella": 34.0, "philippe": 20.0,
                  "melenchon": 11.0, "retailleau": 9.0, "glucksmann": 9.0}
        params = {n: 0.5 for n in TIER1_PARAMS}
        qualified, opp, p = second_round_probability(scores, params, cfg)
        assert not qualified
        assert opp is None
        assert p == 0.0

    def test_qualified_vs_bardella(self, cfg):
        scores = {"villepin": 22.0, "bardella": 25.0, "philippe": 18.0,
                  "melenchon": 11.0, "retailleau": 9.0, "glucksmann": 9.0}
        params = {n: 0.5 for n in TIER1_PARAMS}
        params["anti_extreme_pressure"] = 1.0
        qualified, opp, p = second_round_probability(scores, params, cfg)
        assert qualified
        assert opp == "bardella"
        # Front républicain max : la probabilité devrait être substantielle
        assert p > 0.5, f"P(victoire 2T) faible vs RN avec anti_extreme=1 : {p:.3f}"

    def test_anti_extreme_pressure_helps_vs_rn(self, cfg):
        scores = {"villepin": 22.0, "bardella": 28.0, "philippe": 18.0,
                  "melenchon": 11.0, "retailleau": 9.0, "glucksmann": 9.0}
        low = second_round_probability(scores, {**{n: 0.5 for n in TIER1_PARAMS}, "anti_extreme_pressure": 0.0}, cfg)[2]
        high = second_round_probability(scores, {**{n: 0.5 for n in TIER1_PARAMS}, "anti_extreme_pressure": 1.0}, cfg)[2]
        assert high > low


# ---------------------------------------------------------------- full sim
class TestSimulate:
    def test_simulate_once_reproducible(self, cfg, neutral_params):
        r1 = simulate_once(neutral_params, cfg)
        r2 = simulate_once(neutral_params, cfg)
        assert r1.scores_1T == r2.scores_1T
        assert r1.p_victory_2T == r2.p_victory_2T

    def test_monte_carlo_reproducible(self, cfg, neutral_params):
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        m1 = simulate_monte_carlo(neutral_params, cfg, n_samples=20, rng=rng1)
        m2 = simulate_monte_carlo(neutral_params, cfg, n_samples=20, rng=rng2)
        assert m1["score_1T_mean"] == pytest.approx(m2["score_1T_mean"])
        assert m1["p_victory"] == pytest.approx(m2["p_victory"])

    def test_no_trivial_optimum(self, cfg, all_ones):
        # Test clé : tout à 1 ne doit PAS donner P(victoire) = 1.
        # Si oui, le modèle a un optimum trivial -> bug de design.
        res = simulate_once(all_ones, cfg)
        assert res.p_victory_2T < 0.99, "optimum trivial détecté : tout à 1 -> P~1"

    def test_baseline_villepin_modeste(self, cfg, neutral_params):
        # Avec paramètres neutres, Villepin doit être un outsider, pas en tête.
        scores = first_round_scores(neutral_params, cfg)
        assert scores["villepin"] < scores["bardella"]
        assert scores["villepin"] < 15.0, f"Villepin trop fort en baseline : {scores['villepin']:.2f}"
