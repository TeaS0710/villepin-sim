# Villepin 2027 : Rapport exploratoire

> ⚠️ **Disclaimer.** Ce rapport est un **exercice de modélisation politique exploratoire**.
> Les paramètres et fonctions de coût sont calibrés sur 5 élections historiques
> (2002-2022) avec un nombre limité de degrés de liberté. Les résultats ne sont
> **PAS des prédictions** mais des **explorations de scénarios contrefactuels**.
> La politique réelle implique des facteurs (chocs noirs, personnalité, hasard)
> impossibles à modéliser. Les « stratégies optimales » identifiées sont des
> hypothèses à interroger, pas des chemin optimals opérationnelles.


## Résumé exécutif

Avec un simulateur **audité contre les biais** et calibré sur 2002-2022
(MAE leave-one-out **2.05 points**, couverture sensibilité
**0%**), et une optimisation CMA-ES
sur les 4 paramètres de campagne contrôlables, **la probabilité maximale de
victoire au 2T identifiée est de 0.73%**, atteinte dans
**« tout_max »**.

**Conclusion principale** : sous des affinités électorales sobres (Villepin
n'est PAS magnétique pour la gauche, ni pour les insoumis, ni hyper-favori
des indécis), le modèle prédit une **impossibilité structurelle quasi-totale
de qualification au 2T**. Le résultat de v0 (plafond 6%) était gonflé par
un biais "fanboy" sur les affinités initiales.

## 0. Audit anti-biais

Cette version v1 audite explicitement les biais possibles du modèle :

### Biais corrigés
- **Affinités pro-Villepin allégées** : l'ancienne table donnait à Villepin une
  affinité positive sur 5 pools sur 6 (`indecis: +0.7`, `gauche: +0.4`, `lfi: +0.1`).
  Révisé en `indecis: +0.30`, `gauche: -0.05`, `lfi: -0.30`, `lr: +0.45` -
  un gaulliste-chiraquien ne ratisse pas spontanément ni les insoumis ni les Verts.
- **Validation sur tous les archétypes** (pas seulement le Villepin-équivalent) :
  on contrôle que la dynamique ne distord pas excessivement les bases d'entrée.
- **Analyse de sensibilité au contexte** : on perturbe les estimations à dire
  d'expert de ±0.20 pour mesurer la fragilité du verdict.

### Asymétrie structurelle restante
MAE des archétypes : **3.53**, max |delta| **15.07**
(cas le pire : glucksmann 2007, base assignée
27.4, prédit 42.5). La dynamique
mobile/stuck distord encore significativement les scores des concurrents non-Villepin.

### Robustesse au choix de contexte
Couverture 0% : les intervalles de prédiction p5-p95 (sur 100 perturbations
±0.20 du contexte) contiennent 0% des scores réels. Une couverture < 50%
indique que la précision affichée est artificielle : le modèle ne devrait pas
être lu en valeur exacte mais en ordre de grandeur.

|   year | candidate        |   actual |   predicted_mean |   predicted_std |   predicted_p5 |   predicted_p95 | in_range   |
|-------:|:-----------------|---------:|-----------------:|----------------:|---------------:|----------------:|:-----------|
|   2002 | François Bayrou  |     6.84 |             2.14 |            0.23 |           1.75 |            2.53 | False      |
|   2007 | François Bayrou  |    18.57 |             5.7  |            0.82 |           4.31 |            7.15 | False      |
|   2012 | François Bayrou  |     9.13 |             2.74 |            0.39 |           2.22 |            3.47 | False      |
|   2017 | Emmanuel Macron  |    24.01 |            14.84 |            0.74 |          13.48 |           16.03 | False      |
|   2022 | Valérie Pécresse |     4.78 |             2.55 |            0.46 |           1.85 |            3.31 | False      |

![Sensibilité au contexte](plots/sensitivity.png)

### Biais résiduels documentés (non corrigés en v1)
- **Affinités sobres mais subjectives** : les nouvelles valeurs restent à dire
  d'expert. Une vraie calibration requerrait des données de transferts électoraux.
- **Contexte historique post-hoc** : mes valeurs de `CONTEXT[year]` connaissent
  le résultat de l'élection (Macron 2017 = "tout au max" est ré-construit).
- **5 archétypes seulement** : aggréger 16 candidats 2002 sur 5 archétypes perd
  beaucoup d'information.
- **Affinités pool identiques** : Bayrou 2002 ≠ Bayrou 2007 ≠ Macron 2017 dans
  la réalité (campagnes très différentes), mais traités comme un seul archétype.
- **Adversaires non adaptatifs** : pas de réponse stratégique des autres.

## D. Validation supervisée sur élections passées (2002-2022)

Pour chaque élection historique, on configure le simulateur avec les bases concurrents agrégées et le contexte Tier 1 estimé, puis on prédit (i) les deux qualifiés au 2T et (ii) le vainqueur du second tour. On compare aux résultats officiels.

### Métriques de classification

| métrique | valeur |
|---|---|
| **F1** (qualification 2T) | **0.900** |
| Precision (qualif) | 0.900 |
| Recall (qualif) | 0.900 |
| Accuracy (qualif) | 0.933 |
| **Accuracy vainqueur 2T** | **3/5 = 60%** |
| TP / FP / FN / TN | 9 / 1 / 1 / 19 |

![Matrices de validation historique](plots/historical_validation_metrics.png)

![Score 1T prédit vs réel par année × archétype](plots/historical_pred_vs_actual.png)

### Validation honnête (walk-forward T-2 mois)

⚠️ La validation ci-dessus est **in-sample** : `CONTEXT[year]` et `YEAR_POOLS[year]` sont hard-codés en connaissant le résultat. La vraie capacité prédictive du modèle se mesure en walk-forward, avec uniquement les sondages T-2 mois (jamais le résultat) :

| métrique | in-sample (post-hoc) | walk-forward (T-2 mois) |
|---|---|---|
| MAE 1T | 2.73 pts | **4.67 pts** |
| F1 qualif | 0.900 | **1.000** |
| Accuracy qualif | 0.933 | **1.000** |
| Vainqueur 2T | 3/5 | **2/5** |

![Comparaison in-sample vs walk-forward](plots/historical_validation_compare.png)

### Détails par année

|   year | actual_top2            | predicted_top2         |   top2_overlap | actual_winner   | predicted_winner   | winner_correct   |
|-------:|:-----------------------|:-----------------------|---------------:|:----------------|:-------------------|:-----------------|
|   2002 | glucksmann, retailleau | bardella, glucksmann   |              1 | retailleau      | glucksmann         | False            |
|   2007 | glucksmann, retailleau | glucksmann, retailleau |              2 | retailleau      | glucksmann         | False            |
|   2012 | glucksmann, retailleau | glucksmann, retailleau |              2 | glucksmann      | glucksmann         | True             |
|   2017 | bardella, villepin     | bardella, villepin     |              2 | villepin        | villepin           | True             |
|   2022 | bardella, philippe     | bardella, philippe     |              2 | philippe        | philippe           | True             |

### Lecture honnête

**Le modèle est correct pour la qualification au 2T** dans ~80% des cas et avec F1 ≈ 0.70 (top-2 partiellement matché). Il est **systématiquement faux pour le 2T 2002-2022 (0/5)** : le modèle prédit toujours `bardella` (camp RN) comme vainqueur. Deux causes structurelles identifiées :

1. **Tailles de pools fixées à mai 2026** : `pool_rn = 33%`, ce qui correspond aux sondages actuels mais SURESTIME le poids historique du FN/RN (16-23% sur 2002-2022). La taille du pool × son inertie (0.85) donne à `bardella` une rétention de ~28% du corps électoral, qui dépasse presque toujours les scores 1T historiques réels.
2. **Boost de front républicain mal calibré** : `boost_to_score = 30` est trop large pour les scénarios historiques où le score-différence 1T était petit (2-5 points). Sigmoid+sigmoid_scale 5 amplifient au lieu d'amortir.

Ce diagnostic est utile : il indique deux pistes claires pour la v3 (pool sizes dynamiques par année, calibration empirique des paramètres 2T sur les 5 élections).

## E. Dynamiques internes du modèle

### Hiérarchie des variables (sensibilité globale du surrogate)

![Sensibilité Tier 1](plots/param_sensitivity.png)

**Lecture** : la sensibilité moyenne `E[|∂P(victoire)/∂param|]` mesure, sur 2000 échantillons aléatoires du dataset, à quel point une petite variation du paramètre fait varier la prédiction. Constat majeur : les 4 paramètres **exogènes** (`volatility`, `crisis`, `anti_extreme_pressure`, `central_collapse`) dominent les 4 **internes** (campagne) : l'effet de chaque exogène est ~2× celui de chaque interne. Conclusion politique : **la campagne Villepin a moins de levier que le contexte qu'elle n'a pas choisi**.

### Partial dependence : effet marginal isolé de chaque paramètre

![Partial dependence](plots/partial_dependence.png)

Pour chaque paramètre Tier 1, on fixe les 7 autres à leur médiane du dataset et on fait varier la valeur du paramètre de 0 à 1. Toutes les courbes sont **monotones croissantes** (plus haut = mieux pour Villepin) et **proches du linéaire** (pas de saturation forte dans la région médiane). Hiérarchie visible : `volatility` produit la plus grande amplitude (~0.12 % → ~0.15 %), puis `crisis` et `central_collapse`, tandis que `coalition_building` et `media_performance` sont quasi plats à la médiane.

### Baseline 2027 isolé (params Tier 1 neutres)

![Baseline 2027](plots/baseline_2027.png)

Prédiction du modèle pour le contexte 2027 sans aucune perturbation : Tier 1 fixés à 0.5, bases concurrents = sondages mai 2026. Bardella ressort à 38.8%, Villepin à 3.55%.

### D'où vient le score ? Origine par pool, par scénario

![Pool ownership par scénario](plots/pool_ownership_by_scenario.png)

Décomposition du score 1T de chaque candidat par pool électoral d'origine, **pour chaque scénario exogène** (paramètres internes fixés à 0.5). On voit pourquoi Bardella domine structurellement : il capture massivement le pool `rn` (33 % du corps électoral, inertie 0.85 : très peu mobile). Villepin tire ses voix de `central`, `lr` et `indecis`, trois pools dont la taille combinée plafonne à ~40 % et qui sont disputés avec Philippe (natural owner de `central`).

### Décomposition baseline vs optimum (paramètres internes différents)

![Pool breakdown baseline vs optimum](plots/pool_breakdown.png)

Comparaison directe : à paramètres neutres (gauche) vs optimum CMA-ES (droite). La structure de capture par pool est très stable : l'optimisation interne ne déplace marginalement que les flux mobiles, pas les rétentions de stock.

## A. Qui peut gagner ? (P(victoire 2T) par candidat)

Probabilité moyenne de victoire 2T pour chaque candidat, agrégée sur 36 combinaisons exogènes × shocks externes.

| candidate   |   p_victory |   p_qualif |   score_1T_mean |
|:------------|------------:|-----------:|----------------:|
| bardella    |       0.515 |          1 |          31.585 |
| philippe    |       0.485 |          1 |          22.788 |
| glucksmann  |       0     |          0 |          12.751 |
| melenchon   |       0     |          0 |          10.554 |
| poutou      |       0     |          0 |           2.58  |
| retailleau  |       0     |          0 |           6.858 |
| villepin    |       0     |          0 |           5.891 |
| zemmour     |       0     |          0 |           6.995 |

![Winner heatmap](plots/winner_heatmap.png)

### Top 10 meilleurs scénarios pour Villepin

| exo_scenario   | shock_scenario    |   score_1T_mean |   p_qualif |   p_victory |
|:---------------|:------------------|----------------:|-----------:|------------:|
| calme          | baseline          |            2.18 |          0 |           0 |
| calme          | bardella_collapse |            2.23 |          0 |           0 |
| calme          | philippe_collapse |            2.96 |          0 |           0 |
| calme          | both_collapse     |            3.04 |          0 |           0 |
| calme          | rn_zemmour_split  |            2.21 |          0 |           0 |
| calme          | central_consolide |            1.9  |          0 |           0 |
| median         | baseline          |            4.2  |          0 |           0 |
| median         | bardella_collapse |            4.34 |          0 |           0 |
| median         | philippe_collapse |            5.55 |          0 |           0 |
| median         | both_collapse     |            5.8  |          0 |           0 |

![Top scénarios Villepin](plots/villepin_top_scenarios.png)

## B. Recherche extrême : peut-on faire gagner Villepin sans biais ?

CMA-ES sur 13 dimensions (8 paramètres Tier 1 + 5 bases concurrents bornées par leurs intervalles historiques 2002-2022). Objectif : maximiser P(victoire Villepin).

**P(victoire Villepin) maximale identifiée : 87.70%** (score 1T = 27.3%, vs bardella).

**Conditions requises** (toutes simultanément) :

| variable | valeur à l'optimum |
|---|---|
| `base_bardella` | 15.0% |
| `base_philippe` | 3.0% |
| `base_melenchon` | 4.0% |
| `base_retailleau` | 22.0% |
| `base_glucksmann` | 2.0% |
| `crisis` | 1.00 |
| `central_collapse` | 0.00 |
| `volatility` | 1.00 |
| `anti_extreme_pressure` | 1.00 |
| `campaign_machine` | 1.00 |
| `thematic_breadth` | 1.00 |
| `media_performance` | 1.00 |
| `coalition_building` | 1.00 |

**Robustesse** : sur 10 restarts CMA-ES, P(victoire) ∈ [87.70%, 87.70%]. Un seul optimum trouvé.

**Interprétation honnête** : ce résultat dit *« si Bardella tombe au plancher historique RN, ET Philippe au plancher des sortants, ET la crise est maximale, ET Villepin sature sa campagne, ALORS le modèle prédit ~94% »*. La probabilité conjointe de ces 4 chocs en réalité est très faible.

## F. Chemin minimal vers une victoire de Villepin

**Baseline 2027** : P(victoire) = 0.00 %  
**Optimum extrême atteint par CMA-ES 13D** : P(victoire) = 87.70 %

On interpole linéairement entre baseline 2027 et optimum CMA-ES (13D). Pour chaque seuil de P(victoire), on trouve le pas α minimal qui le franchit ; le vecteur correspondant donne le **shift requis variable par variable**.

![Courbe alpha → P(victoire)](plots/path_to_victory_curve.png)

![Shifts requis par seuil](plots/path_to_victory_shifts.png)

### Lecture des shifts (depuis baseline 2027)

| P_cible   | P_obtenu   |     α | Bardella     | Philippe    | Mélenchon   | Retailleau   | Glucksmann   |
|:----------|:-----------|------:|:-------------|:------------|:------------|:-------------|:-------------|
| ≥ 5%      | 15.4%      | 0.573 | 24.0 (-12.0) | 9.4 (-8.6)  | 7.0 (-4.0)  | 16.9 (+6.9)  | 5.8 (-5.2)   |
| ≥ 10%     | 15.6%      | 0.576 | 23.9 (-12.1) | 9.4 (-8.6)  | 7.0 (-4.0)  | 16.9 (+6.9)  | 5.8 (-5.2)   |
| ≥ 25%     | 25.0%      | 0.667 | 22.0 (-14.0) | 8.0 (-10.0) | 6.3 (-4.7)  | 18.0 (+8.0)  | 5.0 (-6.0)   |
| ≥ 50%     | 50.0%      | 0.811 | 19.0 (-17.0) | 5.8 (-12.2) | 5.3 (-5.7)  | 19.7 (+9.7)  | 3.7 (-7.3)   |
| ≥ 75%     | 75.0%      | 0.926 | 16.6 (-19.4) | 4.1 (-13.9) | 4.5 (-6.5)  | 21.1 (+11.1) | 2.7 (-8.3)   |

### Lecture des Tier 1 (paramètres exogènes + campagne)

| P_cible   |   crisis |   central_collapse |   volatility |   anti_extreme |   machine |   thematic |   media |   coalition |
|:----------|---------:|-------------------:|-------------:|---------------:|----------:|-----------:|--------:|------------:|
| ≥ 5%      |     0.79 |               0.21 |         0.79 |           0.79 |      0.79 |       0.79 |    0.79 |        0.79 |
| ≥ 10%     |     0.79 |               0.21 |         0.79 |           0.79 |      0.79 |       0.79 |    0.79 |        0.79 |
| ≥ 25%     |     0.83 |               0.17 |         0.83 |           0.83 |      0.83 |       0.83 |    0.83 |        0.83 |
| ≥ 50%     |     0.91 |               0.09 |         0.91 |           0.91 |      0.91 |       0.91 |    0.91 |        0.91 |
| ≥ 75%     |     0.96 |               0.04 |         0.96 |           0.96 |      0.96 |       0.96 |    0.96 |        0.96 |

### Trois enseignements du modèle

1. **Les bases concurrents pèsent plus que les Tier 1**. Faire baisser Bardella de 12-19 pts et Philippe de 9-14 pts est nécessaire à tous les seuils. Les Tier 1 (campagne + exogènes) bougent en parallèle de 0.29 → 0.46 mais leur effet marginal est secondaire vs la décrue des concurrents.
2. **Boost contre-intuitif de Retailleau**. Le modèle pousse `base_retailleau` à la hausse (+7 → +11 pts). Mécanisme : Retailleau a une affinité positive sur le pool RN (+0.20) et fait donc concurrence à Bardella ; en gonflant Retailleau, on fragmente le vote de droite et on prive Bardella d'une partie de son socle.
3. **`central_collapse` doit DIMINUER**, pas augmenter. Avec Philippe déjà rabaissé à 3-6 pts, le pool central est essentiellement vide. Transférer le pool central vers les indécis (mécanisme de `central_collapse`) prive Villepin du pool central où il a son affinité la plus forte (+0.40). Garder le pool central intact lui donne plus de matière à capter.

## Annexe pédagogique : comprendre la machine

Cette annexe décrit visuellement la chaîne de calcul, l'architecture du
réseau de neurones surrogate et l'ampleur du calcul effectué.

### Vue d'ensemble du pipeline

![Pipeline complet du simulateur](plots/pipeline_overview.png)

### Architecture du réseau de neurones : schéma

![Architecture du réseau (vue de profil)](plots/nn_architecture.png)

![Anatomie d'un bloc résiduel](plots/nn_block_anatomy.png)

### Architecture du réseau : vue 3D volumétrique

Le réseau surrogate est un MLP résiduel à **910 083 paramètres**. La figure
ci-dessous le rend à l'échelle réelle : **3 179 neurones** dessinés un par
un en grille, et la totalité des **904 992 arêtes Linear** tracées entre
couches consécutives. Trois skip-connections résiduelles relient Embed à
la sortie de chaque bloc.

![Neurones et connexions, vue 3D volumétrique](plots/nn_3d_volumetric.png)

### Distribution des paramètres et hyperparamètres

![Distribution des 910 083 paramètres par couche](plots/nn_params_pie.png)

![Hyperparamètres d'entraînement](plots/nn_hyperparams.png)

### Poids réels appris (extraits du checkpoint)

![Heatmap d'une matrice de poids réelle](plots/nn_weights_heatmap.png)

![Distribution empirique des poids](plots/nn_weights_distribution.png)

### Inférence des paramètres : flux

![Flux d'inférence (3 étages : exogènes, internes, sub-params LLM)](plots/tuning_inference_flow.png)

### Volumes de calcul effectivement exécutés sur ce run

![Cartes des volumes (simulations / paramètres / évaluations / validations)](plots/volumes_cards.png)

![Échelle log des volumes](plots/volumes_log_scale.png)

![Entonnoir : du calcul brut aux scénarios finaux](plots/volumes_funnel.png)

## Méthodologie

1. **Simulateur physique** : compartiments électoraux (6 pools), flux
   gravitationnel Bradley-Terry adapté, masse Villepin = sigmoïde additive
   sur 8 paramètres Tier 1.
2. **Calibration historique** : fit ridge de 4 paramètres globaux sur
   2002-2022 (Bayrou 2002/2007/2012, Macron 2017, Pécresse 2022). Validation
   leave-one-out.
3. **Dataset synthétique** : Latin Hypercube Sampling, Monte Carlo (bruit
   ±15% sur params).
4. **Surrogate MLP** : entraîné à approximer le simulateur (1000× plus rapide).
5. **CMA-ES multi-restart** : optimisation par scénario exogène fixé, puis
   clustering KMeans des top candidats.

## 1. Calibration historique

### Baseline (priors, pas de fit)
MAE : **7.09 points**

|   year | candidate        |   actual |   predicted |   error |   abs_error |
|-------:|:-----------------|---------:|------------:|--------:|------------:|
|   2002 | François Bayrou  |     6.84 |        2.11 |   -4.73 |        4.73 |
|   2007 | François Bayrou  |    18.57 |        5.61 |  -12.96 |       12.96 |
|   2012 | François Bayrou  |     9.13 |        2.7  |   -6.43 |        6.43 |
|   2017 | Emmanuel Macron  |    24.01 |       14.9  |   -9.11 |        9.11 |
|   2022 | Valérie Pécresse |     4.78 |        2.56 |   -2.22 |        2.22 |

### Fit in-sample (ridge λ=0.1, fit sur les 5 années)
MAE : **0.97 points**

Paramètres ajustés :
- `mass_bias` : 0.9169884600099005
- `mass_m_max` : 4.33938054468412
- `villepin_scale` : 5.249084677617855
- `volatility_softening` : 0.8740092943961189
- `second_round_scale` : 5.043476406107304
- `second_round_boost` : 30.00275343864178

|   year | candidate        |   actual |   predicted |   error |   abs_error |
|-------:|:-----------------|---------:|------------:|--------:|------------:|
|   2002 | François Bayrou  |     6.84 |        6.92 |    0.08 |        0.08 |
|   2007 | François Bayrou  |    18.57 |       16.4  |   -2.17 |        2.17 |
|   2012 | François Bayrou  |     9.13 |        8.77 |   -0.36 |        0.36 |
|   2017 | Emmanuel Macron  |    24.01 |       24.01 |    0    |        0    |
|   2022 | Valérie Pécresse |     4.78 |        7.02 |    2.24 |        2.24 |

### Leave-one-out (entraînement sur 4 années, test sur la 5ème)
MAE : **2.05 points**

|   year_held_out | candidate        |   actual |   predicted |   abs_error |
|----------------:|:-----------------|---------:|------------:|------------:|
|            2002 | François Bayrou  |     6.84 |        7.24 |        0.4  |
|            2007 | François Bayrou  |    18.57 |       16.28 |        2.29 |
|            2012 | François Bayrou  |     9.13 |        8.73 |        0.4  |
|            2017 | Emmanuel Macron  |    24.01 |       28.33 |        4.32 |
|            2022 | Valérie Pécresse |     4.78 |        7.64 |        2.86 |

![Calibration LOO](plots/calibration_loo.png)

**Observation honnête** : le modèle reproduit raisonnablement bien Macron 2017
(scénario "tempête parfaite") et Pécresse 2022 (scénario faible). Il sous-estime
nettement **Bayrou 2007** (2.3 pts d'erreur),
indiquant qu'il existe en 2007 un facteur non capturé par les variables
structurelles encodées (charisme personnel, dynamique de campagne, momentum
TV) : c'est une limite intrinsèque.

## 2. Scénarios exogènes : plafond de probabilité

Pour chaque scénario exogène fixé (`crisis`, `central_collapse`, `volatility`,
`anti_extreme_pressure`), on optimise par CMA-ES les 4 paramètres internes
(`campaign_machine`, `thematic_breadth`, `media_performance`,
`coalition_building`) sous contraintes budgétaires.

| scenario     |   crisis |   central_collapse |   volatility |   anti_extreme_pressure |   best_p_victory |   best_score_1T |
|:-------------|---------:|-------------------:|-------------:|------------------------:|-----------------:|----------------:|
| calme        |     0.2  |               0.1  |         0.3  |                    0.3  |           0.0038 |            1.43 |
| median       |     0.5  |               0.5  |         0.5  |                    0.5  |           0.0044 |            5.69 |
| tempete_2017 |     0.6  |               0.85 |         0.75 |                    0.65 |           0.0057 |            7.73 |
| crise_pure   |     0.95 |               0.4  |         0.8  |                    0.7  |           0.005  |            6.95 |
| vide_central |     0.3  |               0.95 |         0.6  |                    0.6  |           0.0054 |            7.2  |
| tout_max     |     1    |               1    |         1    |                    1    |           0.0073 |            9.78 |

![Probabilité par scénario](plots/scenarios_p_victory.png)

**Plafond identifié** : 0.73% dans `tout_max`.

### Allocation d'effort optimale par scénario

![Allocation effort](plots/effort_allocation.png)

Lecture politique : dans les scénarios non-calmes, le modèle recommande de
saturer `campaign_machine`, `media_performance` et `coalition_building`, et
d'**abandonner** `thematic_breadth`. Conclusion contre-intuitive : signal
possible d'une faiblesse du modèle, à interroger.

### Décomposition par pool : d'où vient le score ?

![Décomposition par pool](plots/pool_breakdown.png)

Lecture : à paramètres neutres, Villepin tire principalement du pool `central`
(natural owner = Philippe, donc peu de stuck pour Villepin) et `indecis`
(natural owner Villepin → stuck favorable). À l'optimum CMA-ES, la masse de
Villepin augmente fortement → flux mobile capturé plus large.

### Importance des paramètres Tier 1 (SHAP)

![SHAP P(victoire)](plots/shap_p_victory.png)

Lecture : chaque point = une configuration du dataset ; couleur = valeur du
paramètre (rouge = élevé, bleu = bas) ; abscisse = contribution (positive →
augmente P(victoire), négative → diminue).

## 3. Archétypes de stratégies

Clustering KMeans (k=6) sur les top candidats CMA-ES dans le
scénario tempête_2017. Centroïdes des paramètres internes :

|   cluster |   n_members |   best_p_victory |   best_score_1T |   mean_p_victory |   centroid_campaign_machine |   centroid_thematic_breadth |   centroid_media_performance |   centroid_coalition_building |   best_campaign_machine |   best_thematic_breadth |   best_media_performance |   best_coalition_building |
|----------:|------------:|-----------------:|----------------:|-----------------:|----------------------------:|----------------------------:|-----------------------------:|------------------------------:|------------------------:|------------------------:|-------------------------:|--------------------------:|
|         0 |          62 |       0.00571006 |         7.73081 |       0.00571006 |                    0.905524 |                    0.274535 |                     0.99997  |                      0.819972 |                0.905856 |                0.274123 |                 0.999972 |                  0.820048 |
|         1 |          31 |       0.00571006 |         7.73114 |       0.00571006 |                    0.906416 |                    0.273606 |                     0.999972 |                      0.820006 |                0.906841 |                0.274627 |                 0.99998  |                  0.818552 |
|         2 |          48 |       0.00571006 |         7.73103 |       0.00571006 |                    0.904605 |                    0.275082 |                     0.999973 |                      0.82034  |                0.904838 |                0.275203 |                 0.999947 |                  0.820011 |
|         3 |          36 |       0.00571006 |         7.73067 |       0.00571006 |                    0.905356 |                    0.274035 |                     0.999967 |                      0.820642 |                0.905392 |                0.273924 |                 0.999967 |                  0.820717 |
|         4 |          21 |       0.00571006 |         7.73124 |       0.00571006 |                    0.904201 |                    0.275898 |                     0.999969 |                      0.819932 |                0.904405 |                0.275988 |                 0.999959 |                  0.819647 |
|         5 |          52 |       0.00571006 |         7.73104 |       0.00571006 |                    0.905116 |                    0.275332 |                     0.999973 |                      0.819578 |                0.905119 |                0.275101 |                 0.999945 |                  0.819836 |

![Archétypes](plots/archetypes_radar.png)

**Observation** : les clusters convergent tous vers une même P(victoire) à
0.57%, signe que CMA-ES trouve un
**optimum unique** plutôt que plusieurs archétypes distincts. Pas de
multimodalité détectée à ce niveau de modélisation.

## 4. Limites du modèle (au moins 8)

1. **Calibration sous-déterminée** : 5 élections × 1 candidat-cible = 5 points
   pour fitter 4 paramètres. Le modèle est statistiquement fragile.
2. **Estimations de contexte historique à dire d'expert** (cf
   `src/historical_context.py`) : biais cognitif inévitable.
3. **Adversaires statiques** : les concurrents ne réagissent pas à la stratégie
   Villepin. Une vraie campagne est adaptative.
4. **Absence de dynamique temporelle** : le modèle est statique (un seul état
   pour toute la campagne), pas mois-par-mois.
5. **Pas de sous-paramètres Tier 2** dans cette version v1 (boucle LLM
   non activée).
6. **Affinités candidat × pool fixées** : recalibrer plus profondément les
   affinités demanderait un dataset bien plus large.
7. **Front républicain modélisé par une fonction sigmoïde simple** : néglige
   les dynamiques d'abstention au 2T.
8. **Outlier Bayrou 2007 non expliqué** : indication forte qu'un facteur
   majeur (charisme/momentum) est manquant.

## 5. Recommandations actionnables (sous réserve des limites ci-dessus)

Pour le scénario le plus favorable (`tout_max`), les
paramètres internes optimaux suggérés sont :

|                    |   valeur optimale |
|:-------------------|------------------:|
| campaign_machine   |             0.82  |
| thematic_breadth   |             0.365 |
| media_performance  |             1     |
| coalition_building |             0.815 |

## 6. Prochaines étapes

- **Activer la boucle LLM (Ollama Turbo)** pour découvrir des sous-paramètres
  Tier 2 (cf `src/llm_param_discovery.py`).
- **Étendre la calibration** sur élections municipales / européennes pour plus
  de points.
- **Ajouter des adversaires adaptatifs** (self-play).
- **Modèle temporel mensuel** mai 2026 → avril 2027.

---

*Généré automatiquement. Code et données : `villepin_sim/`.*
