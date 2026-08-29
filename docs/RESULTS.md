# Résultats v1 (mode full, audité anti-biais)

> Tous les chiffres ci-dessous sont issus de `outputs/`. Pour la critique
> méthodologique : voir [BIASES.md](BIASES.md).

## 1. Calibration historique

### 1.1 Métriques agrégées

| métrique | valeur |
|---|---|
| MAE baseline (priors, sans fit) | **8.98 points** |
| MAE in-sample (fit sur 5 années) | **3.09 points** |
| MAE leave-one-out (LOO) | **4.43 points** |
| MAE archétypes (audit asymétrie) | **6.62 points** |
| Couverture sensibilité ±0.20 | **0%** |

### 1.2 Paramètres ajustés (ridge λ=0.1)

| paramètre | prior | fitted |
|---|---|---|
| `mass_model.bias`        | -1.000 |  0.962 |
| `mass_model.m_max`       |  3.000 |  4.679 |
| `capture.villepin_scale` |  4.000 |  6.628 |
| `capture.volatility_softening` |  0.600 |  1.000 |

**Observation** : `villepin_scale` et `volatility_softening` se poussent contre
les bornes. Signe de **calibration sous-spécifiée** (4 paramètres pour 5 points).

### 1.3 Erreurs par année (LOO)

| année | candidat | réel | prédit | |erreur| |
|---|---|---|---|---|
| 2002 | François Bayrou  |  6.84 |  9.82 | 2.98 |
| 2007 | François Bayrou  | 18.57 |  8.42 | **10.15** |
| 2012 | François Bayrou  |  9.13 |  6.70 | 2.43 |
| 2017 | Emmanuel Macron  | 24.01 | 23.20 | 0.81 |
| 2022 | Valérie Pécresse |  4.78 | 10.54 | 5.76 |

**Outliers** :
- **Bayrou 2007** (10.15 pts d'erreur) : sous-prédit. Le modèle ne peut pas
  générer un score "outsider" élevé sans effondrement central simultané. C'est
  une limite structurelle révélant qu'un facteur (charisme/momentum/débat TV)
  manque.
- **Pécresse 2022** (5.76 pts d'erreur, sur-estimation) : à l'inverse, le
  modèle prédit qu'elle aurait dû mieux faire dans ce contexte (guerre Ukraine).
  Son discours catastrophique du Zenith n'est pas modélisable.

---

## 2. Distribution du dataset (30 000 configs LHS)

| statistique | score_1T_mean |
|---|---|
| min   |  3.49 |
| Q1    |  9.02 |
| médiane | 10.57 |
| Q3    | 12.21 |
| max   | 15.70 |

| seuil | % configs |
|---|---|
| P(qualif) > 0 | **0.0%** |
| P(victoire) > 0.05 | 0.0% |
| P(victoire) > 0.5  | 0.0% |

→ Sur les 30 000 configurations aléatoires (LHS dans l'hypercube [0,1]⁸),
**aucune** n'atteint la qualification au 2T. Le score 1T plafonne à 15.70%,
insuffisant pour passer Philippe (~20-25%).

---

## 3. Scenario sweep CMA-ES

Optimisation des 4 paramètres internes (`campaign_machine`, `thematic_breadth`,
`media_performance`, `coalition_building`) pour chaque scénario exogène fixé.

| scénario | crisis | central_collapse | volat. | anti_ext. | **P(vict.)** | score_1T |
|---|---|---|---|---|---|---|
| `calme`        | 0.20 | 0.10 | 0.30 | 0.30 | **0.35%** |  4.44 |
| `median`       | 0.50 | 0.50 | 0.50 | 0.50 | **0.43%** | 12.12 |
| `tempete_2017` | 0.60 | 0.85 | 0.75 | 0.65 | **0.62%** | 14.22 |
| `crise_pure`   | 0.95 | 0.40 | 0.80 | 0.70 | **0.59%** | 14.17 |
| `vide_central` | 0.30 | 0.95 | 0.60 | 0.60 | **0.55%** | 13.34 |
| `tout_max`     | 1.00 | 1.00 | 1.00 | 1.00 | **0.82%** | 15.50 |

**Plafond global identifié** : P(victoire 2T) = **0.82%** dans le scénario
*irréaliste* tout-au-max. Pour des scénarios plus réalistes (calme à
tempete_2017), le plafond est entre 0.35% et 0.62%.

---

## 4. Stratégies optimales par scénario

Les paramètres internes optimaux montrent une convergence frappante :

| scénario | machine | thematic | media | coalition | Σ interne |
|---|---|---|---|---|---|
| `calme`        | 0.285 | 0.035 | 0.368 | 0.095 | 0.78 |
| `median`       | 1.000 | 0.000 | 1.000 | 1.000 | 3.00 |
| `tempete_2017` | 1.000 | 0.000 | 1.000 | 1.000 | 3.00 |
| `crise_pure`   | 1.000 | 0.000 | 1.000 | 1.000 | 3.00 |
| `vide_central` | 1.000 | 0.000 | 1.000 | 1.000 | 3.00 |
| `tout_max`     | 1.000 | 0.000 | 0.448 | 1.000 | 2.45 |

**Lecture politique** :
- Dans tous les scénarios non-calmes, le modèle recommande de **saturer le
  budget interne** (Σ = 3.00) en mettant **machine, media et coalition au max**
  et en **abandonnant `thematic_breadth`**. Conclusion contre-intuitive
  signalant une faiblesse possible du modèle.
- Dans le scénario `tout_max`, même la média baisse à 0.45 — quand les
  conditions externes sont parfaites, l'effort interne devient moins critique
  (mais le résultat reste sous 1%).
- Dans le scénario `calme`, optimum très bas (toutes valeurs ≤ 0.4) : le
  modèle dit "ne pas dépenser l'effort si le contexte est défavorable".

---

## 5. Clustering archétypes (scénario tempete_2017)

Multi-restart 20× CMA-ES, top-50 par run = 1000 candidats. KMeans k=6.

| cluster | n_membres | best P(victoire) |
|---|---|---|
| 0 | 831 | 0.616% |
| 1 |  85 | 0.616% |
| 2 |  54 | 0.616% |
| 3 |   1 | 0.616% |
| 4 |  28 | 0.616% |
| 5 |   1 | 0.616% |

**Tous les clusters convergent** vers la même P(victoire). Pas d'optimum
multimodal. L'algorithme trouve un seul "rocher" optimal vers lequel les 20
restarts convergent. Pas d'archétypes stratégiques distincts.

---

## 6. Sensibilité au contexte

100 perturbations ±0.20 sur les 8 paramètres `CONTEXT[year]`, par élection.

| année | candidat | réel | p5 | p50 | p95 | dans p5-p95 ? |
|---|---|---|---|---|---|---|
| 2002 | Bayrou   |  6.84 | 0.86 |  1.14 |  1.45 | ✘ |
| 2007 | Bayrou   | 18.57 | 1.24 |  1.80 |  2.38 | ✘ |
| 2012 | Bayrou   |  9.13 | 0.86 |  1.16 |  1.55 | ✘ |
| 2017 | Macron   | 24.01 | 11.91 | 12.94 | 14.04 | ✘ |
| 2022 | Pécresse |  4.78 | 1.04 |  1.42 |  1.96 | ✘ |

**0% de couverture**. Le modèle, partant des `CONTEXT[year]` perturbés, produit
des prédictions systématiquement *décalées* par rapport au réel. Cela signifie
que le fit historique **dépend très fortement** du choix exact de `CONTEXT[year]`.

→ La précision affichée dans les autres résultats (3 chiffres après la virgule)
**n'est pas réelle**. Le modèle doit être lu en ordres de grandeur, pas en
valeurs exactes.

---

## 7. Qui peut gagner ? (winner_analysis)

Pour chaque combinaison exogène × shock concurrents (36 cellules × 6 candidats
= 216 cas), Monte Carlo 500 itérations + calcul symétrisé P(victoire 2T).

| candidat | P(victoire) moyenne | P(qualif) moyenne | score 1T moyen |
|---|---|---|---|
| **bardella**   | **72.3%** | 100.0% | 35.10 |
| philippe       | 23.2%     |  75.4% | 20.06 |
| villepin       |  4.6%     |  24.6% | 13.56 |
| melenchon      |  0.0%     |   0.0% | 13.77 |
| glucksmann     |  0.0%     |   0.0% | 11.69 |
| retailleau     |  0.0%     |   0.0% |  5.82 |

**Verdict** : Bardella gagne dans 100% des combinaisons étudiées (P entre 51% et 95%).

**Top 10 meilleurs scénarios pour Villepin** :

| exo_scenario | shock | score_1T | p_qualif | p_victory |
|---|---|---|---|---|
| tout_max     | both_collapse     | 21.6 | 1.000 | **47.2%** |
| crise_pure   | both_collapse     | 19.4 | 1.000 | 26.0% |
| tout_max     | philippe_collapse | 20.1 | 1.000 | 23.2% |
| tempete_2017 | both_collapse     | 19.1 | 1.000 | 20.0% |
| crise_pure   | philippe_collapse | 18.4 | 1.000 | 13.5% |
| vide_central | both_collapse     | 17.4 | 0.998 | 12.8% |
| tempete_2017 | philippe_collapse | 17.8 | 1.000 |  8.9% |
| vide_central | philippe_collapse | 16.1 | 0.976 |  5.5% |

→ Pour P(Villepin) > 20%, il faut SIMULTANÉMENT :
- conditions exogènes au max (`tout_max` ou `crise_pure`)
- ET effondrement de Bardella + Philippe (`both_collapse`)

---

## 8. Recherche extrême (extreme_search)

CMA-ES sur 13 dimensions (8 Tier 1 + 5 bases concurrents bornées historiquement
2002-2022). 10 restarts. Objectif : maximiser P(victoire Villepin).

**Résultat** : tous les restarts convergent vers **P(victoire) = 93.64%**.

| variable | valeur à l'optimum | sémantique |
|---|---|---|
| `base_bardella`        | **15.0** | plancher RN/FN historique (Le Pen 2007 = 10.4) |
| `base_philippe`        | **3.0**  | plancher sortants centristes |
| `base_melenchon`       |  4.0     | plancher LFI |
| `base_retailleau`      |  3.0     | plancher LR |
| `base_glucksmann`      |  2.0     | plancher PS |
| `crisis`               | 1.00     | crise géopolitique max |
| `central_collapse`     | 0.00     | (non nécessaire — Philippe déjà au plancher) |
| `volatility`           | 1.00     | volatilité électorale max |
| `anti_extreme_pressure`| 1.00     | front républicain max |
| `campaign_machine`     | 1.00     | machine de campagne max |
| `thematic_breadth`     | 1.00     | largeur thématique max |
| `media_performance`    | 1.00     | performance médiatique max |
| `coalition_building`   | 1.00     | ralliements max |

**Score Villepin à l'optimum : 28.3%** au 1er tour, vs Bardella 15%.

**Interprétation rigoureuse** : ce n'est PAS *« Villepin a 93% de chances »*.
C'est *« sous une conjonction extrêmement improbable de 4 chocs simultanés
(effondrement Bardella, effondrement Philippe, crise géopolitique majeure,
campagne saturée), le modèle prédit 93% »*. Chaque choc seul est rare
(probabilité ~5%) ; leur conjonction est dans le bruit (~0.001%).

---

## 9. Conclusion v1

**Verdict sobre** :

- **Bardella est le favori massif** : P(victoire) moyenne 72%, jamais < 50% sur
  les combinaisons étudiées.
- **Sous baseline May 2026** (bases concurrents fixées, Tier 1 optimisé) :
  Villepin plafonne à **0.82% de P(victoire)**.
- **Sous shocks plausibles** (bases concurrents libres dans bornes historiques) :
  Villepin atteint **47%** (`tout_max + both_collapse`) — mais cela requiert la
  conjonction de chocs très improbables.
- **À l'optimum théorique 13D sans biais** : Villepin atteint **93.64%**, mais
  uniquement avec Bardella à 15% ET Philippe à 3% ET crisis/anti_extreme/
  volatility = 1.0 ET campagne saturée. Probabilité conjointe réelle ≪ 1%.

**Trois interprétations légitimes** :

1. **Le modèle a raison** : Villepin n'a structurellement pas les attributs
   pour gagner 2027 (pas de socle naturel suffisamment large dans le profil
   actuel de l'électorat).
2. **Le modèle sous-estime un facteur** : charisme, momentum, ralliement
   surprise, événement noir. Bayrou 2007 indique qu'il existe des "moments"
   non capturables structurellement.
3. **Les hypothèses sont trop pessimistes** : les affinités sobres v1 sont
   peut-être trop restrictives. Une calibration sur données réelles de
   transferts électoraux (pas de proxy historique agrégé) donnerait des
   chiffres différents.

Voir [BIASES.md](BIASES.md) pour le détail.
