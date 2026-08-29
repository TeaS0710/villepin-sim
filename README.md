# DDV_ML v2 — Simulation contrefactuelle d'une victoire Villepin 2027

> ⚠️ **Exercice exploratoire**. Pas une prédiction. Pas un outil de campagne.
> Les sorties sont des hypothèses contrefactuelles à interroger.

---

## TL;DR du verdict v1

Après calibration historique sur les 5 dernières présidentielles (2002-2022)
et **audit explicite des biais pro-Villepin de la version 0**, le modèle prédit :

| métrique | valeur |
|---|---|
| MAE leave-one-out (calibration) | **4.43 points** |
| Couverture sensibilité contexte ±0.20 | **0%** |
| % de configs LHS atteignant qualification 2T | **0%** |
| Plafond P(victoire 2T) tous scénarios | **0.82%** |
| Optimum CMA-ES (scénario tout_max) | machine=1.0, thematic=0.0, media=0.45, coalition=1.0 |

**Lecture honnête** : sous des affinités électorales sobres (Villepin n'est ni
magnétique pour la gauche, ni hyper-favori des indécis), **le modèle prédit
l'impossibilité structurelle quasi-totale d'une victoire**. La version v0 du
même modèle annonçait un plafond ~6% — c'était l'effet d'affinités initiales
généreuses (`gauche: +0.4`, `lfi: +0.1`, `indecis: +0.7`) qui ont été corrigées.

Pour la critique méthodologique complète : voir [docs/BIASES.md](docs/BIASES.md).

---

## Structure du projet

```
villepin_sim/
├── config.yaml                  # priors (à éditer = changer les hypothèses)
├── config.fitted.yaml           # priors après calibration (généré)
├── requirements.txt
├── main.py                      # orchestration (CLI)
├── src/
│   ├── parameters.py            # espace des 8 paramètres Tier 1
│   ├── physical_model.py        # simulateur : pools, masse, capture
│   ├── historical_data.py       # scraping reproductible Wikipédia
│   ├── historical_context.py    # contexte Tier 1 estimé par élection
│   ├── calibration.py           # ridge fit + LOO + audits
│   ├── dataset_generator.py     # LHS + Monte Carlo -> parquet
│   ├── neural_predictor.py      # MLP surrogate
│   ├── genetic_optimizer.py     # CMA-ES multi-restart + KMeans
│   ├── llm_param_discovery.py   # stub Ollama Turbo (v2)
│   ├── reporter.py              # rapport markdown + plots
│   └── dashboard.py             # Streamlit
├── tests/
│   └── test_physical_model.py   # 16 tests (invariants)
├── data/
│   ├── raw/                     # snapshots HTML Wikipédia + SHA256
│   └── historical_elections.csv # 61 candidats × 5 élections
├── outputs/
│   ├── checkpoints/             # nn_surrogate.pt
│   ├── plots/                   # 4 PNG (calibration, scenarios, archetypes, dataset)
│   ├── final_report.md          # rapport généré
│   ├── calibration_*.csv        # baseline / in-sample / LOO
│   ├── audit_*.csv              # archetypes / sensitivity
│   ├── cmaes_*.csv              # scenarios / top candidats / labels
│   ├── archetypes.csv           # centroïdes KMeans
│   └── calibration_summary.json # métriques consolidées
└── docs/
    ├── METHODOLOGY.md
    ├── RESULTS.md
    └── BIASES.md
```

---

## Installation

```bash
cd villepin_sim/
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Pour la boucle LLM v2 (non branchée par défaut) :

```bash
export OLLAMA_API_KEY="..."   # Ollama Turbo
```

---

## Lancement

```bash
# Pipeline complète v2 (scrape + calibration + dataset + NN + CMA-ES + winner + extreme + rapport)
python main.py --mode full     # 80k configs, NN profond, CMA-ES large — ~25 min CPU
python main.py --mode quick    # 4k configs — ~30 s (debug/dev)

# Avec boucle LLM Tier 2 (4 itérations, modèle gemma4:31b-cloud)
python main.py --mode full --llm-iterations 4

# Boucle LLM seule (après une pipeline initiale)
python -m src.llm_loop --iterations 4 --mode quick

# Régénérer seulement le rapport
python main.py --only-report

# Skip une étape (les outputs précédents sont réutilisés)
python main.py --skip-scrape --skip-calibration

# Dashboard interactif
streamlit run src/dashboard.py
```

Les tests :

```bash
python -m pytest tests/ -v
```

---

## Documentation détaillée

- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** — Modèle physique-statistique
  pas à pas : compartiments électoraux, masse sigmoïde additive, capture
  Bradley-Terry mobile/stuck, calibration ridge, surrogate NN, CMA-ES.
- **[docs/RESULTS.md](docs/RESULTS.md)** — Tableaux complets calibration,
  scenarios CMA-ES, archétypes, distributions du dataset.
- **[docs/BIASES.md](docs/BIASES.md)** — Audit explicite des biais : ceux qu'on
  a corrigés (affinités v0), ceux qu'on n'a pas corrigés (contexte post-hoc,
  agrégation 5 archétypes), et leur impact mesuré.

---

## Nouveautés v2

- **Espace de paramètres dynamique** : Tier 1 (8 fixes) + Tier 2 (illimité,
  proposé par le LLM, intégré à la formule de masse). Cf `src/parameters.py § space_from_config`.
- **NN surrogate plus profond** : 4 blocs résiduels × 256 hidden (~660k params,
  vs 33k en v1). Pré-norm GELU + dropout. Cf `src/neural_predictor.py § _ResidualBlock`.
- **Boucle d'inférence Tier 2** branchée à **Ollama** (modèle par défaut
  `gemma4:31b-cloud`) : LLM propose, filtre structurel + filtre redondance
  par projection fonctionnelle sur la masse, rollback automatique si la
  validation ne progresse pas. Cf `src/llm_loop.py` et `src/llm_param_discovery.py`.
- **Compute revu à la hausse** : 80k configs LHS (vs 30k), CMA-ES popsize 200
  × 30 restarts × 800 itérations (vs 100×20×500).
- **Plots additionnels** : évolution P(victoire) sur itérations LLM,
  poids Tier 2 groupés par parent (cf `outputs/plots/`).

## Principes de design

1. **Calibration empirique d'abord** — fit ridge sur 5 élections avec validation
   leave-one-out, *avant* toute optimisation.
2. **Formule additive + sigmoïde** — pas d'optimum trivial. Bornée. Calibrable.
3. **Tier 2 = modulateurs de masse** : chaque sous-paramètre LLM contribue
   additivement à l'argument du sigmoïde, pondéré par son importance estimée
   par le LLM (capped via `pipeline.llm.importance_to_weight`).
4. **Reproductibilité** — seeds fixées (`config.seed: 42`), snapshots HTML versionnés
   avec SHA256, parquet pour le dataset, config snapshots par itération LLM
   dans `outputs/llm_history/iter_NN/`.
5. **Audit anti-biais intégré** — chaque calibration produit
   `audit_archetypes.csv` (asymétrie structurelle) et `audit_sensitivity.csv`
   (robustesse au contexte).
6. **Contraintes réalistes** — budget interne ≤ 3.0, coalition ≤ 0.6 si machine
   < 0.5, etc. (cf `config.yaml § constraints`).
7. **Rollback systématique** : une itération LLM qui ne fait pas progresser la
   best P(victoire) de plus de 0.001 est rejetée. Évite les régressions par
   ajout de dimensions inutiles. Cf `src/llm_loop.py § main`.
8. **Filtre de redondance fonctionnel** : un sous-paramètre candidat est
   rejeté si |corrélation| > 0.7 avec les paramètres existants (mesurée sur
   sensitivities de masse via différences finies). Cf `src/llm_param_discovery.py § check_redundancy_via_mass`.
9. **Honnêteté épistémique** — si le modèle dit "impossible", on l'écrit dans
   le rapport sans cherry-picking.

---

## Limites connues (cf `docs/BIASES.md`)

- Affinités électorales `villepin_affinity` restent à dire d'expert, même
  recalibrées.
- Contexte historique `CONTEXT[year]` connaît rétrospectivement les résultats.
- Bayrou 2007 (18.57% réel) reste inexpliqué — manque un facteur "momentum".
- 5 archétypes pour ~12 candidats par élection = compression majeure.
- Adversaires non adaptatifs (statiques).
- Calibration sous-déterminée : 5 points pour 4 paramètres globaux.

---

## License

Code MIT. Données : Wikipédia (CC-BY-SA).

## Licence et données

Code sous licence MIT. Les données électorales agrégées (`data/`) proviennent de sources
publiques : résultats officiels des présidentielles 2002-2022 et sondages de second tour
archivés depuis Wikipédia (CC BY-SA). Le scraper `src/historical_polls_scraper.py` permet
de reconstruire les données brutes; elles ne sont pas redistribuées ici.
