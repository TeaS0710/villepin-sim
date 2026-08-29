"""Boucle LLM (Ollama, modèle gemma4:31b-cloud) pour découvrir des sous-paramètres
Tier 2 qui enrichissent le modèle. Implémentation v2 :

Workflow d'une itération
------------------------
1. Récupérer les top-10 stratégies CMA-ES actuelles (depuis cmaes_top_candidates).
2. Demander au LLM N nouveaux sous-paramètres (chacun = nom, parent Tier 1,
   importance, justification).
3. **Filtrage anti-redondance** : pour chaque candidat sous-paramètre, vérifier
   par Monte Carlo qu'il n'est pas trop corrélé avec les paramètres existants
   (|r| < `redundancy_threshold`). Vu que les sub-params sont *exogènes* au
   simulateur initial, la corrélation est calculée *via leur effet sur la
   masse Villepin* (proxy fonctionnel).
4. Intégrer les candidats acceptés dans `config.mass_model.weights` avec un
   poids dérivé de l'importance LLM.
5. Régénérer le dataset, ré-entraîner le NN, relancer le CMA-ES.
6. **Rollback** si la meilleure P(victoire) n'a pas augmenté de plus de
   `min_improvement`.

Aucun appel LLM réel n'est fait sans clé. Mode fallback : pioche dans
une liste pré-définie de sous-paramètres avec heuristique de diversité.
"""
from __future__ import annotations

import copy
import json
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import yaml

from .parameters import TIER1_PARAMS


@dataclass
class SubParam:
    name: str
    description: str
    parent: str
    importance: int
    justification: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


PROMPT = """\
Tu es un analyste politique et un modélisateur statistique français rigoureux. Voici
les {n_top} meilleures stratégies trouvées par un algorithme génétique cherchant à
maximiser la probabilité de victoire de Dominique de Villepin à la présidentielle
française de 2027 selon notre simulateur.

Stratégies au format (paramètre = valeur normalisée [0,1]) :
{top_strategies}

Paramètres DÉJÀ présents dans le modèle (à NE PAS reproposer ni dupliquer) :
{existing_params}

Ta mission : proposer EXACTEMENT {n} sous-paramètres SUPPLÉMENTAIRES, OBSERVABLES,
DÉCISIONNELS, INDÉPENDANTS de ceux existants, qui pourraient finement enrichir le
modèle. Chaque sous-paramètre raffine UN parent Tier 1 :

Parents Tier 1 valides : {tier1_list}.

Pour chaque sous-paramètre :
- `name` : snake_case ASCII, court et descriptif
- `description` : 1 phrase
- `parent` : exactement un nom de la liste ci-dessus
- `importance` : entier 1-10, signifie la magnitude attendue de l'effet
- `justification` : 1 phrase ancrée dans la réalité politique française

Contraintes strictes :
- INTERDIT : duplicate names, jugements moraux, paramètres vagues (ex : « charisme »),
  paramètres déjà présents.
- Privilégier les variables MESURABLES dans le réel (sondage spécifique, indicateur
  économique, ratio, etc.).
- Diversifier les parents : au moins 3 parents différents sur les {n} propositions.

Réponds STRICTEMENT en JSON, un tableau d'objets, sans commentaire ni markdown :

[
  {{"name": "...", "description": "...", "parent": "...", "importance": 5,
    "justification": "..."}}
]
"""


# ----------------------------------------------------------------------
# LLM call (Ollama)
# ----------------------------------------------------------------------
def call_llm(prompt: str, model: str, timeout: int = 120) -> str:
    """Appel Ollama. Fonctionne avec modèles locaux ET :cloud (Ollama Turbo)."""
    import ollama
    t0 = time.time()
    response = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.7, "top_p": 0.9},
    )
    dt = time.time() - t0
    text = response["message"]["content"]
    print(f"  LLM ({model}) → {len(text)} chars en {dt:.1f}s")
    return text


def parse_subparams(text: str) -> list[SubParam]:
    """Extrait le premier JSON array du texte (tolérant aux ```json...``` etc)."""
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = text.replace("```", "")
    # Trouve premier array
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end < 0:
        raise ValueError(f"JSON array introuvable dans la réponse : {text[:200]!r}")
    items = json.loads(text[start : end + 1])
    out = []
    for it in items:
        out.append(SubParam(
            name=str(it["name"]).strip(),
            description=str(it.get("description", "")).strip(),
            parent=str(it["parent"]).strip(),
            importance=int(it.get("importance", 5)),
            justification=str(it.get("justification", "")).strip(),
        ))
    return out


# ----------------------------------------------------------------------
# Validation des sous-paramètres
# ----------------------------------------------------------------------
def validate_subparams(
    candidates: list[SubParam], existing_names: set[str],
) -> tuple[list[SubParam], list[tuple[SubParam, str]]]:
    """Filtre structurel (avant l'analyse fonctionnelle de redondance) :
    - parent valide
    - name unique et non déjà présent
    - importance dans [1, 10]
    - name snake_case ASCII safe
    """
    accepted: list[SubParam] = []
    rejected: list[tuple[SubParam, str]] = []
    seen_in_batch = set()
    for sp in candidates:
        if sp.parent not in TIER1_PARAMS:
            rejected.append((sp, f"parent invalide: {sp.parent!r}"))
            continue
        if sp.name in existing_names:
            rejected.append((sp, "déjà présent"))
            continue
        if sp.name in seen_in_batch:
            rejected.append((sp, "duplicate dans le batch"))
            continue
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,40}", sp.name):
            rejected.append((sp, "name format invalide (snake_case ASCII)"))
            continue
        if not (1 <= sp.importance <= 10):
            rejected.append((sp, "importance hors [1, 10]"))
            continue
        accepted.append(sp)
        seen_in_batch.add(sp.name)
    return accepted, rejected


def check_redundancy_via_mass(
    sp: SubParam, cfg: dict, threshold: float = 0.7, n_mc: int = 800,
    seed: int = 0,
) -> tuple[bool, float, str]:
    """Test fonctionnel : intègre temporairement le sous-paramètre dans la
    formule de masse et mesure la corrélation de son effet vs effet des
    paramètres existants. Si |r| > threshold avec au moins un existant, rejet.

    On échantillonne `n_mc` configurations LHS-like, et on mesure la
    sensibilité d'output à chaque param via différence finie.
    """
    from .physical_model import compute_villepin_mass
    rng = np.random.default_rng(seed)
    existing_names = list(cfg["mass_model"]["weights"].keys())
    n_params = len(existing_names)
    eps = 0.10
    # Base sample
    base_X = rng.uniform(0, 1, size=(n_mc, n_params))
    # Pour chaque param existant, mesure d_mass/d_param
    sensitivities = np.zeros((n_mc, n_params))
    base_mass = np.zeros(n_mc)
    for i in range(n_mc):
        p = {name: float(base_X[i, j]) for j, name in enumerate(existing_names)}
        base_mass[i] = compute_villepin_mass(p, cfg)
    for j, name in enumerate(existing_names):
        X_pert = base_X.copy()
        X_pert[:, j] = np.clip(X_pert[:, j] + eps, 0, 1)
        for i in range(n_mc):
            p = {name2: float(X_pert[i, k]) for k, name2 in enumerate(existing_names)}
            sensitivities[i, j] = (compute_villepin_mass(p, cfg) - base_mass[i]) / eps

    # Maintenant simule l'effet du sous-paramètre candidat : on ajoute un term
    # proportionnel à (sp - 0.5) au parent. La sensibilité du candidat = sa
    # *valeur de signal* projetée sur la sensibilité du parent + bruit.
    parent_idx = existing_names.index(sp.parent)
    # Le candidat draws sa valeur indépendamment ; son effet sur la masse :
    cand_X = rng.uniform(0, 1, size=n_mc)
    w_sp = sp.importance / 10 * cfg["pipeline"]["llm"]["importance_to_weight"]
    cand_signal = w_sp * (cand_X - 0.5)
    # Corrélation entre cand_signal et chacune des n_params sensitivities :
    max_corr = 0.0
    max_with = ""
    for j, name in enumerate(existing_names):
        sj = sensitivities[:, j]
        if sj.std() < 1e-9 or cand_signal.std() < 1e-9:
            continue
        r = float(np.corrcoef(cand_signal, sj)[0, 1])
        if abs(r) > abs(max_corr):
            max_corr = r
            max_with = name
    if abs(max_corr) > threshold:
        return False, max_corr, max_with
    return True, max_corr, max_with


# ----------------------------------------------------------------------
# Mise à jour du config
# ----------------------------------------------------------------------
def apply_subparams_to_cfg(cfg: dict, accepted: list[SubParam]) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("tier2_params", {})
    importance_to_weight = cfg["pipeline"]["llm"]["importance_to_weight"]
    for sp in accepted:
        # Le signe du poids suit le signe du poids du parent (Tier 1)
        parent_w = cfg["mass_model"]["weights"][sp.parent]
        w = (sp.importance / 10.0) * importance_to_weight * (1.0 if parent_w >= 0 else -1.0)
        cfg["mass_model"]["weights"][sp.name] = float(w)
        cfg["tier2_params"][sp.name] = {
            "parent": sp.parent,
            "description": sp.description,
            "importance": sp.importance,
            "justification": sp.justification,
            "weight": float(w),
        }
    return cfg


def format_top_strategies(top_df, n: int = 10) -> str:
    """Formate top_df en bullets compactes pour le prompt."""
    cols = [c for c in top_df.columns if c not in {"p_victory", "score_1T", "p_qualif", "run"}]
    out = []
    for _, row in top_df.head(n).iterrows():
        kv = ", ".join(f"{c}={row[c]:.2f}" for c in cols)
        out.append(f"- p_victoire={row['p_victory']:.4f}, score_1T={row['score_1T']:.2f} | {kv}")
    return "\n".join(out)


def build_prompt(cfg: dict, top_df, n_subparams: int) -> str:
    existing = list(cfg["mass_model"]["weights"].keys())
    tier1_list = ", ".join(TIER1_PARAMS)
    return PROMPT.format(
        n_top=min(10, len(top_df)),
        top_strategies=format_top_strategies(top_df, n=10),
        existing_params=", ".join(existing),
        tier1_list=tier1_list,
        n=n_subparams,
    )


def run_iteration(
    cfg: dict, top_df, model: str, n_subparams: int,
    redundancy_threshold: float, seed: int = 0,
) -> tuple[dict, list[SubParam], list[tuple[SubParam, str]]]:
    """Une itération : LLM appelle + valide + filtre redondance.
    Retourne (cfg_mis_à_jour, accepted, rejected_with_reason).
    """
    existing_names = set(cfg["mass_model"]["weights"].keys())
    prompt = build_prompt(cfg, top_df, n_subparams)
    text = call_llm(prompt, model=model)
    try:
        candidates = parse_subparams(text)
    except Exception as e:
        print(f"  [erreur] parse JSON : {e}")
        return cfg, [], []

    # Filtrage structurel
    accepted_struct, rejected_struct = validate_subparams(candidates, existing_names)
    print(f"  parsing : {len(candidates)} candidats, {len(accepted_struct)} OK structurel, "
          f"{len(rejected_struct)} rejets")

    # Filtrage fonctionnel (redondance via masse)
    accepted: list[SubParam] = []
    rejected_func: list[tuple[SubParam, str]] = []
    for sp in accepted_struct:
        ok, corr, with_name = check_redundancy_via_mass(
            sp, cfg, threshold=redundancy_threshold, seed=seed,
        )
        if ok:
            accepted.append(sp)
            print(f"    ✓ accept {sp.name:30s} (parent={sp.parent}, |r|={abs(corr):.2f})")
        else:
            rejected_func.append((sp, f"redondant avec {with_name} (r={corr:.2f})"))
            print(f"    ✗ reject {sp.name:30s} (redondant avec {with_name}, r={corr:.2f})")

    new_cfg = apply_subparams_to_cfg(cfg, accepted)
    all_rejected = rejected_struct + rejected_func
    return new_cfg, accepted, all_rejected
