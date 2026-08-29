# Méthodologie

## 1. Modèle physique-statistique

### 1.1 Compartiments électoraux

L'électorat est partitionné en 6 pools, calibrés sur agrégats de sondages
(mai 2026). Chaque pool a une *taille* (% des votants exprimés) et une
*inertie* ∈ [0,1] (résistance au changement) :

| pool | taille | inertie | sémantique |
|---|---|---|---|
| `rn`      | 33% | 0.85 | extrême-droite (RN/Reconquête) — très fidèle |
| `central` | 26% | 0.45 | centre macroniste + LR-centriste |
| `gauche`  | 14% | 0.55 | PS + Verts + sociaux-démocrates |
| `lfi`     | 11% | 0.70 | LFI + PC + extrême-gauche — fidèle |
| `lr`      |  9% | 0.60 | LR conservateur (Wauquiez/Retailleau) |
| `indecis` |  7% | 0.15 | indécis / abstention récupérable |

**Note de calibration** : la somme = 100% par construction. Les valeurs sont
ajustables dans `config.yaml § pools` sans changer le code.

### 1.2 Affinités candidat × pool

Chaque candidat (Villepin + 5 archétypes) a un vecteur d'affinités ∈ [-1, 1]
par pool. Une affinité positive signifie "attraction" ; négative signifie
"hostilité" (pas de captation).

L'archétype `bardella` représente le candidat dominant du camp RN (Bardella,
Le Pen). `philippe` = candidat centriste/macroniste sortant. `melenchon` =
candidat LFI. `retailleau` = candidat LR. `glucksmann` = candidat PS/sociodem.

**Affinités Villepin (post-audit v1)** :

| pool | affinité | justification |
|---|---|---|
| `rn`      | -0.70 | très hostile (gaulliste anti-FN) |
| `central` | +0.40 | centre-droit chiraquien, plausible |
| `gauche`  | -0.05 | mince capital antiguerre Irak |
| `lfi`     | -0.30 | libéral économiquement, hostile insoumis |
| `lr`      | +0.45 | héritage gaulliste-chiraquien (son terreau) |
| `indecis` | +0.30 | capital de marque modéré |

### 1.3 Masse gravitationnelle de Villepin

La masse module la "force d'attraction" sur les pools où Villepin a une
affinité positive. Forme **additive + sigmoïde** (pas multiplicative — cf
`docs/BIASES.md § Formule multiplicative pathologique évitée`) :

```
m = m_min + (m_max - m_min) * σ( Σᵢ wᵢ · (xᵢ - 0.5) + b )

où   σ(z) = 1 / (1 + exp(-z))
     xᵢ ∈ [0, 1]  paramètres Tier 1
     wᵢ           poids par paramètre (calibrés)
     b            biais
```

**Propriétés** :
- bornée dans `[m_min, m_max]` → pas d'explosion.
- centrée sur `0.5` → un paramètre "neutre" ne contribue pas.
- saturante → pas d'optimum trivial "tout à 1".
- monotone en chaque paramètre.

### 1.4 Capture concurrentielle (mobile / stuck)

Pour chaque pool, on distingue :

- **stuck** = `pool_size × (1 - effective_mobility)` → attribué au *propriétaire
  naturel* du pool (candidat avec l'affinité positive maximale).
- **mobile** = `pool_size × effective_mobility` → réparti au prorata des "pulls"
  Bradley-Terry parmi les candidats à affinité positive.

```
pull_villepin = m × aff_v × scale_v
pull_concurrent = base × aff_c

share_mobile = pull / Σ pulls
```

`effective_mobility = (1 - inertia) + softening · volatility · inertia` : la
volatility globale "ramollit" l'inertie de tous les pools.

### 1.5 Effondrement central

Si `central_collapse > 0`, transfert `central → indecis` :

```
transfer = pool[central].size × central_collapse × transfer_ratio
```

`transfer_ratio = 0.55` (paramètre). Modélise l'érosion du centre vers les
indécis quand PS/LR s'effondrent.

### 1.6 Score final 1er tour

```
score_villepin = Σpool (stuck_si_natural + mobile_share × mobile)
                clippé à [score_min, score_max] = [0.5, 38]
```

### 1.7 Second tour

Si Villepin est dans le top-2 du 1er tour :

```
boost = clip(base + slope · anti_extreme_pressure, 0, 0.95)
P(victoire) = σ((score_v + boost · 30 - score_opp) / 5)
```

`base` et `slope` dépendent de l'identité de l'adversaire (cf
`config.yaml § second_round.report_boost`). Camp RN → fort front républicain.
Centre/LR → faible.

---

## 2. Calibration historique

### 2.1 Source de données

Scraping reproductible Wikipédia (5 pages) avec snapshots HTML versionnés et
hash SHA256. 61 candidats × 5 élections (2002, 2007, 2012, 2017, 2022).

### 2.2 Mapping candidat → archétype

`src/historical_context.py § ARCHETYPE_MAPPING` agrège les candidats historiques
sur les 5 archétypes. Exemple 2017 :

- `bardella` ← Le Pen (21.30%) + Dupont-Aignan (4.70%) = 26.00%
- `philippe` ← (vide — pas de centriste sortant en 2017)
- `glucksmann` ← Hamon (6.36%)
- `melenchon` ← Mélenchon + Poutou + Arthaud = 20.45%
- `retailleau` ← Fillon + Asselineau + Cheminade + Lassalle = 21.55%

### 2.3 Cible de calibration

Pour chaque année, on cherche à prédire le score du *Villepin-équivalent* :

| année | candidat-cible | score réel |
|---|---|---|
| 2002 | François Bayrou  |  6.84% |
| 2007 | François Bayrou  | 18.57% |
| 2012 | François Bayrou  |  9.13% |
| 2017 | Emmanuel Macron  | 24.01% |
| 2022 | Valérie Pécresse |  4.78% |

### 2.4 Contexte Tier 1 estimé

Pour chaque année, 8 valeurs ∈ [0,1] (cf `src/historical_context.py § CONTEXT`).
Estimations à dire d'expert ancrées dans :
- inflation/chômage INSEE de l'époque (proxy `anti_extreme_pressure`)
- événements géopolitiques majeurs (proxy `crisis`)
- état des partis sortants (proxy `central_collapse`)
- ressources de campagne documentées (proxy `campaign_machine`, etc.)

**Limite** : ces valeurs connaissent rétrospectivement le résultat. Cf
`docs/BIASES.md § Biais post-hoc`.

### 2.5 Fit ridge

On optimise par L-BFGS-B (avec bornes) un sous-ensemble de paramètres
globaux pour minimiser le MAE sur la cible, avec régularisation ridge contre
les priors :

```
objective(x) = MAE(prédictions, réels) + λ · Σᵢ ((xᵢ - prior_i) / scale_i)²
```

Paramètres fittés :
- `mass_model.bias`
- `mass_model.m_max`
- `capture.villepin_scale`
- `capture.volatility_softening`

λ = 0.1 (configurable). Choix volontairement parcimonieux : 4 paramètres
pour 5 points = sous-déterminé mais identifiable avec régularisation.

### 2.6 Validation leave-one-out

Pour chaque année `y` :
1. Fit sur les 4 autres années
2. Prédiction sur `y`
3. Calcul MAE

Le **MAE LOO de la v1 = 4.43 points**.

---

## 3. Surrogate neural network

Justification : le simulateur prend ~10 ms par évaluation (Monte Carlo
30 échantillons). Le NN prend ~50 μs sur CPU. Pour 1M évaluations CMA-ES, on
gagne ~2 ordres de grandeur.

**Architecture** :

```
Linear(8, 128) → LayerNorm → GELU → Dropout(0.1)
Linear(128, 128) → LayerNorm → GELU → Dropout(0.1)
Linear(128, 64) → GELU
Linear(64, 3) → [σ × 50, σ, σ]   (score 1T, p_qualif, p_victoire)
```

**Training** :
- 30 000 configs LHS, 70/15/15 train/val/test
- MSE pondérée [1, 5, 20] (la victoire est rare → poids élevé)
- AdamW lr=1e-3 wd=1e-4, CosineAnnealing
- Early stop patience=10
- ~50-120 epochs

**Métriques v1 (mode full)** :

| sortie | MAE val | MAE test |
|---|---|---|
| score 1T   | 0.178 | 0.179 |
| p_qualif   | 0.003 | 0.003 |
| p_victoire | 0.003 | 0.003 |

→ surrogate très précis (largement sous la cible <1.5 du brief).

---

## 4. Optimisation CMA-ES

### 4.1 Mode `scenario_sweep`

Pour chaque scénario exogène (`crisis`, `central_collapse`, `volatility`,
`anti_extreme_pressure` fixés), on optimise par CMA-ES les 4 paramètres
internes (`campaign_machine`, `thematic_breadth`, `media_performance`,
`coalition_building`) sous contraintes :

- coalition_building ≤ 0.6 si campaign_machine < 0.5
- thematic_breadth ≤ 0.4 si campaign_machine < 0.3
- Σ paramètres internes ≤ 3.0 (budget)

CMA-ES : sigma0=0.2, popsize=100, maxiter=500.

### 4.2 Mode multi-restart + clustering

20 restarts CMA-ES avec initialisations aléatoires différentes. Collecte des
top-50 candidats par run = 1000 candidats. Clustering KMeans (k=6) sur les
8 paramètres → archétypes de stratégie.

En v1, **les 6 clusters convergent vers la même P(victoire)** = pas d'optimum
multimodal détecté pour ce modèle.

---

## 5. Audits anti-biais (intégrés au pipeline)

### 5.1 `evaluate_all_archetypes`

Pour chaque année, prédit le score de TOUS les archétypes (pas seulement le
Villepin-équivalent) et compare à la base agrégée historique. Si le delta
est grand, la dynamique mobile/stuck distord excessivement les bases.

**v1** : MAE archétypes 6.62, max |delta| 22.86 (Bardella 2007).

### 5.2 `sensitivity_analysis`

Pour chaque année, perturbe le `CONTEXT[year]` de ±0.20 sur 100 échantillons.
Calcule la distribution des prédictions Villepin-équivalent. La *couverture*
= % de cas où le score réel tombe dans [p5, p95] des prédictions.

**v1** : couverture 0%. → l'estimation de contexte est trop influente, le
modèle ne devrait pas être lu en valeur exacte.

---

## 6. Reporting

Markdown auto-généré (`outputs/final_report.md`) + 4 plots PNG :

- `calibration_loo.png` : actual vs predicted leave-one-out
- `scenarios_p_victory.png` : P(victoire) par scénario
- `archetypes_radar.png` : centroïdes paramètres par cluster
- `dataset_distrib.png` : distributions score_1T / p_qualif / p_victory

Plus dashboard Streamlit interactif (`streamlit run src/dashboard.py`).
