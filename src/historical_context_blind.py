"""Version "ex-ante" du contexte historique : dérive YEAR_POOLS depuis les
sondages T-2 mois (scrapés ailleurs), donc SANS jamais voir le résultat final.

C'est l'antithèse de `historical_context.py` où `CONTEXT[year]` et
`YEAR_POOLS[year]` sont hard-codés en connaissant le score de chaque candidat.

Utilisé par `historical_validation_walk_forward.py` pour produire une mesure
honnête de la capacité prédictive du modèle.

Hypothèses :
- `CONTEXT_ex_ante[year]` = 0.5 partout (agnostique : on ne sait rien des 8
  dimensions Tier 1 avant l'élection). Ces dimensions sont des artefacts
  rétro-construits et n'ont pas d'équivalent mesurable T-2 mois.
- `YEAR_POOLS_ex_ante[year][pool]["size"]` = somme des sondages T-2 mois des
  candidats appartenant à ce pool (selon ARCHETYPE_MAPPING). Ré-échelonné pour
  que la somme = 95 (les 5 % restants vont à "indecis").
- Les `inertia` restent constantes (propriétés structurelles, pas calibrées
  sur le résultat).
"""
from __future__ import annotations
import unicodedata
from pathlib import Path
import pandas as pd

from .historical_context import ARCHETYPE_MAPPING, VILLEPIN_EQUIVALENT


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("'", "").replace("-", " ").lower().strip()


def _strip_party_suffix(s: str) -> str:
    """'Marine Le Pen (FN)' -> 'Marine Le Pen' (pour matching)."""
    if "(" in s:
        s = s.split("(", 1)[0]
    return s.strip()


# Inerties stables (extraites de YEAR_POOLS d'origine, qui sont structurelles
# et non calibrées sur le résultat ; voir docstring historical_context.py).
DEFAULT_INERTIA = {
    "rn":      0.80,
    "central": 0.42,
    "gauche":  0.50,
    "lfi":     0.65,
    "lr":      0.60,
    "indecis": 0.15,
}

# Map archétype  ->  pool sociologique (1:1 dans ce projet)
ARCH_TO_POOL = {
    "bardella":   "rn",
    "philippe":   "central",
    "villepin":   "central",   # outsider centriste agrégé avec le centre
    "retailleau": "lr",
    "glucksmann": "gauche",
    "melenchon":  "lfi",
}


def load_polls_T2(path: Path | str = "data/historical_polls_T2.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["candidate_clean"] = df["candidate"].apply(
        lambda s: _strip_party_suffix(str(s))
    )
    df["candidate_norm_clean"] = df["candidate_clean"].apply(_norm)
    return df


def derive_year_pools_ex_ante(year: int, polls_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Dérive les tailles de pool pour une année à partir des sondages T-2 mois.

    Chaque candidat sondé en T-2 mois est rattaché à un pool sociologique :
    - via ARCHETYPE_MAPPING[year] si listé
    - via VILLEPIN_EQUIVALENT[year] (centriste outsider) -> pool 'central'
    - sinon non comptabilisé (les autres candidats marginaux n'ont pas de pool)
    Le reste des suffrages va à 'indecis' avec un plancher de 3 %.
    """
    sub = polls_df[polls_df["year"] == year]
    cand_to_pct = dict(zip(sub["candidate_norm_clean"], sub["mean_pct"]))
    matched_cands: set[str] = set()

    pool_sizes = {p: 0.0 for p in DEFAULT_INERTIA}

    def _match_and_add(name: str, pool: str) -> None:
        n = _norm(_strip_party_suffix(name))
        v = cand_to_pct.get(n)
        matched_key = n
        if v is None:
            for k, pct in cand_to_pct.items():
                if n in k or k in n:
                    v = pct
                    matched_key = k
                    break
        if v is not None:
            pool_sizes[pool] += float(v)
            matched_cands.add(matched_key)

    for arch, candidates in ARCHETYPE_MAPPING[year].items():
        pool = ARCH_TO_POOL.get(arch)
        if pool is None:
            continue
        for c in candidates:
            _match_and_add(c, pool)

    # Le proxy Villepin (centriste outsider de l'époque) va dans 'central'
    villepin_eq = VILLEPIN_EQUIVALENT.get(year)
    if villepin_eq:
        _match_and_add(villepin_eq, "central")

    # 'indecis' = ce qui n'a pas été alloué (plancher 3 % pour le bruit de pool)
    known = sum(pool_sizes[p] for p in DEFAULT_INERTIA if p != "indecis")
    pool_sizes["indecis"] = max(3.0, 100.0 - known)

    # Re-normaliser à 100 (sondages se sont approchés de 95-100% selon l'année)
    total = sum(pool_sizes.values())
    if total > 0:
        pool_sizes = {k: v * 100.0 / total for k, v in pool_sizes.items()}

    return {p: {"size": round(pool_sizes[p], 2),
                "inertia": DEFAULT_INERTIA[p]}
            for p in DEFAULT_INERTIA}


def derive_context_ex_ante(year: int, polls_df: pd.DataFrame) -> dict[str, float]:
    """Contexte Tier 1 ex-ante : impossible à mesurer directement T-2 mois,
    on retourne 0.5 partout (= prior agnostique).

    Limite documentée : la valeur 0.5 dégrade artificiellement la performance
    par rapport à la version post-hoc, mais c'est la seule honnête possible
    sans externalités (Google Trends d'époque, INSEE T-2, etc.).
    """
    return {
        "crisis": 0.50,
        "central_collapse": 0.50,
        "volatility": 0.50,
        "anti_extreme_pressure": 0.50,
        "campaign_machine": 0.50,
        "thematic_breadth": 0.50,
        "media_performance": 0.50,
        "coalition_building": 0.50,
    }


def all_year_pools_ex_ante(polls_df: pd.DataFrame | None = None
                            ) -> dict[int, dict[str, dict[str, float]]]:
    if polls_df is None:
        polls_df = load_polls_T2()
    return {y: derive_year_pools_ex_ante(y, polls_df)
            for y in [2002, 2007, 2012, 2017, 2022]}


def all_context_ex_ante(polls_df: pd.DataFrame | None = None
                         ) -> dict[int, dict[str, float]]:
    if polls_df is None:
        polls_df = load_polls_T2()
    return {y: derive_context_ex_ante(y, polls_df)
            for y in [2002, 2007, 2012, 2017, 2022]}


if __name__ == "__main__":
    df = load_polls_T2()
    print("=== YEAR_POOLS_ex_ante ===")
    for year, pools in all_year_pools_ex_ante(df).items():
        sizes = {p: v["size"] for p, v in pools.items()}
        total = sum(sizes.values())
        print(f"\n{year} (somme={total:.1f}):")
        for p, v in pools.items():
            print(f"  {p:8s}  size={v['size']:5.1f}  inertia={v['inertia']:.2f}")
    print("\n=== CONTEXT_ex_ante (constant = 0.5 partout, par design) ===")
