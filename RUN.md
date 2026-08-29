# Commandes pour réitérer les expériences DDV_ML

> Toutes les commandes sont à exécuter depuis `villepin_sim/`.
> Le Python global (`python3`) a déjà toutes les dépendances installées (pas
> de venv requis dans cet environnement). Pour repartir d'un env propre :
> `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

---

## 0. Tests rapides (vérifier l'environnement)

```bash
# 16 tests d'invariants sur le simulateur physique (~5 s) — tous PASS au 2026-05-21
python3 -m pytest tests/ -v

# Sanity-check : tous les modules sont importables et exposent --help
python3 main.py --help

# Régénérer le rapport depuis les CSV existants (~10 s) — vérification rapide
python3 main.py --only-report
```

**État de vérification (2026-05-21)** :
- ✅ `pytest tests/` — 16/16 PASS
- ✅ `python3 main.py --only-report` — exit 0, rapport 28 655 chars
- ✅ `python3 main.py --mode quick --skip-scrape --skip-extreme` — pipeline complète OK
- ❌ `python3 main.py --mode quick` (sans `--skip-extreme`) — plante à `extreme_search` avec `config.yaml` (cf §1)

---

## 1. Pipeline complète

> ⚠️ **Limitation connue (vérifiée 2026-05-21)** : `src/extreme_search.py`
> et `src/path_to_victory.py` sont hard-codés sur des noms d'archétypes
> (`extreme_droite`, `souverainiste`, `centre_gouv`, …) qui n'existent
> que dans `config_multiclass.yaml`. Avec la `config.yaml` par défaut
> (clés `bardella`, `philippe`, …) ces étapes plantent sur
> `KeyError: 'extreme_droite'`. Solution : passer `--skip-extreme` sur
> la pipeline par défaut, ou basculer sur `config_multiclass.yaml`
> (cf §5) avant de les lancer.

```bash
# Pipeline rapide vérifiée OK (~60 s) — utile pour dev/debug
python3 main.py --mode quick --skip-extreme

# Pipeline complète (~25 min CPU) — 80k configs LHS, NN profond, CMA-ES large
python3 main.py --mode full --skip-extreme

# Pipeline complète + boucle LLM Tier 2 (4 itérations, Ollama)
# Nécessite : export OLLAMA_API_KEY="..."  (Ollama Turbo)
python3 main.py --mode full --skip-extreme --llm-iterations 4
```

Étapes orchestrées par `main.py` (dans l'ordre) :
1. Scraping Wikipédia (idempotent via cache HTML + SHA256)
2. Calibration ridge + LOO → écrit `config.fitted.yaml`
3. Génération dataset LHS + Monte Carlo → `outputs/dataset.parquet`
4. Entraînement NN surrogate → `outputs/checkpoints/nn_surrogate.pt`
5. CMA-ES (scénarios + multi-restart + KMeans)
6. Validation honnête walk-forward (sondages T-2 mois)
7. (Optionnel) Boucle LLM Tier 2 + refit
8. Winner analysis (P(victoire) tous candidats)
9. Extreme search + chemin minimal vers la victoire
10. Rapport final → `outputs/final_report.md`

---

## 2. Étapes individuelles (réutilise les outputs précédents)

### Skip ciblé (les `--skip-*` réutilisent les fichiers déjà présents)

```bash
# Refaire seulement NN + CMA-ES + rapport (garde scrape/calibration/dataset)
python3 main.py --mode full --skip-scrape --skip-calibration --skip-dataset

# Juste régénérer le rapport markdown + plots à partir des CSV existants
python3 main.py --only-report
# équivalent à :
python3 -m src.reporter
```

### Lancer un module isolé

```bash
# Scraping (force re-download : --force)
python3 -m src.historical_data
python3 -m src.historical_polls_scraper

# Calibration ridge + audits + écriture config.fitted.yaml
python3 -m src.calibration --write-fitted-config

# Dataset (config par défaut : config.fitted.yaml)
python3 -m src.dataset_generator --mode quick     # 4k configs
python3 -m src.dataset_generator --mode full      # 80k configs

# NN surrogate (lit outputs/dataset.parquet)
python3 -m src.neural_predictor

# CMA-ES (rapide vs large)
python3 -m src.genetic_optimizer --quick
python3 -m src.genetic_optimizer                  # mode large par défaut

# Validation honnête (in-sample puis walk-forward T-2 mois)
python3 -m src.historical_validation
python3 -m src.historical_validation_walk_forward

# Qui gagne ? (P(victoire) tous candidats × scénarios × shocks)
python3 -m src.winner_analysis

# Recherche extrême : peut-on faire gagner Villepin sans biais ?
# ⚠️ Nécessite config_multiclass.yaml (cf §5). Plante sur config.yaml.
python3 -m src.extreme_search --n-restarts 10

# Chemin minimal vers la victoire (depuis baseline)
# ⚠️ Même limitation que extreme_search (mêmes archétypes attendus).
python3 -m src.path_to_victory \
    --extreme-restarts 10 \
    --targets 0.05 0.10 0.25 0.50 0.75

# Rapport final
python3 -m src.reporter
```

---

## 3. Boucle LLM Tier 2 (Ollama)

```bash
# Boucle LLM seule (après une pipeline initiale), mode quick par défaut
export OLLAMA_API_KEY="..."          # Ollama Turbo (cloud)
python3 -m src.llm_loop --iterations 4 --mode quick

# Intégrée dans la pipeline (refait dataset + NN + CMA-ES après accept)
python3 main.py --mode full --llm-iterations 4
```

L'historique des itérations est sauvegardé dans `outputs/llm_history/iter_NN/`.
Rollback automatique si la P(victoire) ne progresse pas de >0.001.

---

## 4. Dashboard interactif

```bash
streamlit run src/dashboard.py
# → http://localhost:8501
```

---

## 5. Variantes de configuration (8 pools / multiclasses)

Les variantes `config_8pools.yaml` et `config_multiclass.yaml` ne sont pas
branchées dans `main.py`. Pour les rejouer, basculer la config par défaut :

```bash
# Sauvegarde la config courante
cp config.yaml config.yaml.bak

# Variante 8 pools (voir outputs_8pools/)
cp config_8pools.yaml config.yaml
python3 main.py --mode full
mv outputs outputs_8pools_$(date +%Y%m%d)

# Variante multiclasses (voir outputs_multiclass/)
cp config_multiclass.yaml config.yaml
python3 main.py --mode full
mv outputs outputs_multiclass_$(date +%Y%m%d)

# Restauration
cp config.yaml.bak config.yaml
```

---

## 6. Reset / nettoyage

```bash
# Supprime les artefacts générés (garde data/ et config.yaml)
rm -rf outputs/checkpoints outputs/plots outputs/*.csv outputs/*.json outputs/*.parquet outputs/final_report.md
rm -f config.fitted.yaml

# Force re-scrape (ignore le cache HTML)
python3 -m src.historical_data --force
python3 -m src.historical_polls_scraper --force
```

---

## 7. Reproductibilité

- `seed: 42` dans `config.yaml` → résultats identiques d'un run à l'autre.
- Snapshots HTML versionnés dans `data/raw/` avec SHA256.
- `config.fitted.yaml` est régénéré déterministiquement par la calibration.
- À versionner pour reproduire un run : `config.yaml`, `data/raw/`, `data/historical_elections.csv`.
