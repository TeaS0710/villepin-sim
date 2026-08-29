"""Espace de paramètres : noms, ordre, bornes, encodage/décodage vecteur.

v2 : support dynamique des Tier 2 (sous-paramètres LLM). Le ParamSpace est
construit depuis `config.mass_model.weights.keys()` + métadata
`tier2_params` (parent, classification interne/exogène).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TIER1_PARAMS: tuple[str, ...] = (
    "crisis",
    "central_collapse",
    "volatility",
    "anti_extreme_pressure",
    "campaign_machine",
    "thematic_breadth",
    "media_performance",
    "coalition_building",
)

TIER1_EXOGENOUS: frozenset[str] = frozenset({
    "crisis", "central_collapse", "volatility", "anti_extreme_pressure",
})
TIER1_INTERNAL: frozenset[str] = frozenset(TIER1_PARAMS) - TIER1_EXOGENOUS

# Alias rétrocompat
EXOGENOUS = TIER1_EXOGENOUS
INTERNAL = TIER1_INTERNAL


@dataclass(frozen=True)
class ParamSpace:
    names: tuple[str, ...]
    lower: np.ndarray
    upper: np.ndarray
    parents: tuple[str, ...]  # pour chaque param : nom du parent Tier 1
                              # (pour Tier 1 : parent = nom lui-même)

    @property
    def n(self) -> int:
        return len(self.names)

    def to_dict(self, x: np.ndarray) -> dict[str, float]:
        return {name: float(v) for name, v in zip(self.names, x)}

    def to_vector(self, d: dict[str, float]) -> np.ndarray:
        return np.array([d[n] for n in self.names], dtype=np.float64)

    def clip(self, x: np.ndarray) -> np.ndarray:
        return np.clip(x, self.lower, self.upper)

    def exogenous_mask(self) -> np.ndarray:
        """True pour paramètres exogènes (incl. enfants d'un parent exogène)."""
        return np.array([self.parents[i] in TIER1_EXOGENOUS for i in range(self.n)])

    def internal_mask(self) -> np.ndarray:
        return ~self.exogenous_mask()


def default_space() -> ParamSpace:
    n = len(TIER1_PARAMS)
    return ParamSpace(
        names=TIER1_PARAMS,
        lower=np.zeros(n, dtype=np.float64),
        upper=np.ones(n, dtype=np.float64),
        parents=TIER1_PARAMS,
    )


def space_from_config(cfg: dict) -> ParamSpace:
    """Construit le ParamSpace depuis `config.mass_model.weights` et le mapping
    `tier2_params` (si présent). Les Tier 1 sont toujours présents.
    """
    weights = cfg["mass_model"]["weights"]
    tier2_meta: dict = cfg.get("tier2_params", {})  # name -> {parent: str, ...}

    names = list(TIER1_PARAMS)
    parents = list(TIER1_PARAMS)
    for name in weights:
        if name in TIER1_PARAMS:
            continue
        if name not in tier2_meta:
            # Tolérant : sans métadata, on l'inclut avec parent inconnu
            parent = tier2_meta.get(name, {}).get("parent", "campaign_machine")
        else:
            parent = tier2_meta[name].get("parent", "campaign_machine")
        names.append(name)
        parents.append(parent)
    n = len(names)
    return ParamSpace(
        names=tuple(names),
        lower=np.zeros(n, dtype=np.float64),
        upper=np.ones(n, dtype=np.float64),
        parents=tuple(parents),
    )
