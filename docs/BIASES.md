# Audit des biais

## 1. Biais corrigés en v1

### 1.1 Affinités pro-Villepin "fanboy" (corrigé)

**v0** (héritée du brief original) :
```yaml
villepin_affinity:
  rn: -0.6 ; central: 0.5 ; gauche: 0.4 ; lfi: 0.1 ; lr: 0.2 ; indecis: 0.7
```

**Problème** : 5 affinités positives sur 6 pools. Villepin attire à la fois
les indécis (+0.7), les Verts (+0.4), les insoumis (+0.1) — irréaliste pour un
ex-PM chiraquien-gaulliste.

**v1** (révisé) :
```yaml
villepin_affinity:
  rn: -0.70 ; central: 0.40 ; gauche: -0.05 ; lfi: -0.30 ; lr: 0.45 ; indecis: 0.30
```

**Impact mesuré** : plafond P(victoire) v0 ≈ 6.1% → v1 ≈ 0.82%. Le biais v0
gonflait artificiellement le score de Villepin par un facteur 7×.

### 1.2 Formule multiplicative pathologique (corrigé dès la conception)

**Brief original** :
```python
mass = international × refuge × machine × thematic × media × coalition
```

**Problème** : un seul facteur à 0 → masse nulle. À l'inverse, tous à 1 →
masse explosive. Crée un optimum trivial "tout à 1" et une zone morte
inexploitable.

**v1** : masse additive + sigmoïde, bornée, sans optimum trivial. Cf
`docs/METHODOLOGY.md § 1.3`.

### 1.3 Validation sur un seul candidat-cible (corrigé)

**v0** : la calibration ne mesurait que l'erreur sur le score du
Villepin-équivalent (Bayrou/Macron/Pécresse).

**v1** : ajout de `evaluate_all_archetypes` qui prédit aussi le score de
tous les concurrents et compare à leurs bases historiques agrégées. Révèle
une asymétrie structurelle (max |delta| = 22.86 pts).

### 1.4 Absence d'analyse de sensibilité (corrigé)

**v0** : prédictions présentées en valeurs exactes.

**v1** : `sensitivity_analysis` perturbe `CONTEXT[year]` de ±0.20 sur 100
échantillons, mesure la couverture (réel dans [p5, p95]). Résultat : 0% de
couverture → la précision affichée est artificielle.

---

## 2. Biais documentés mais NON corrigés en v1

### 2.1 Affinités sobres mais subjectives

Les nouvelles `villepin_affinity` restent posées à dire d'expert. Une vraie
calibration empirique requerrait des **données de transferts électoraux**
(qui a voté Villepin parmi ceux qui ont voté Bayrou ?) qu'on n'a pas.

**Impact estimé** : ±20% sur les chiffres v1, voire plus si Villepin a un
profil idiosyncratique non capturable par les 5 archétypes.

### 2.2 Contexte historique post-hoc

Mes valeurs `CONTEXT[year]` dans `src/historical_context.py` connaissent
rétrospectivement le résultat de chaque élection. Exemple : Macron 2017 est
encodé `central_collapse=0.85` (très haut) parce qu'on SAIT que PS/UMP ont
explosé. Ce n'est pas un signal *prédictif* mais un signal *post-hoc*.

**Solution v2** : utiliser des sondages d'époque (6 mois avant l'élection)
pour estimer le contexte. Demande un dataset additionnel.

### 2.3 Agrégation 5 archétypes / N candidats

En 2002, 16 candidats sont compressés sur 5 archétypes. La diversité fine
(Mamère ≠ Taubira, par exemple) est perdue. Acceptable en première
approximation mais perd du signal.

### 2.4 Affinités archétypes identiques sur 20 ans

Le modèle suppose que Bayrou 2002 = Bayrou 2007 = Macron 2017 = Pécresse 2022
en termes de profil pool×affinité. C'est faux : Macron 2017 captait
explicitement de la gauche, Pécresse 2022 quasi-pas. Les affinités historiques
devraient être ajustées par candidat.

### 2.5 Adversaires non adaptatifs

Bardella, Philippe, etc. ont des affinités et bases fixées. Dans la réalité,
ils réagissent à la stratégie Villepin. Self-play ou jeu à somme nulle =
extension v2 nécessaire.

### 2.6 Statique, pas temporel

Le modèle prédit un état terminal (élection). Une vraie campagne est dynamique
(débats, gaffes, ralliements progressifs). Modèle temporel mensuel = v2.

### 2.7 Asymétrie de paramétrisation

Villepin a 8 paramètres flexibles (mass model). Les concurrents ont une `base`
fixe. Cette asymétrie suppose que seule la qualité de la campagne Villepin
varie, alors que la qualité des campagnes concurrentes varie aussi en
réalité.

### 2.8 Calibration sous-déterminée

5 points (Bayrou×3, Macron 2017, Pécresse 2022) pour 4 paramètres globaux fittés.
Le fit pousse contre les bornes (`m_max=4.68`, `volatility_softening=1.0`),
signalant une sous-spécification. Plus de points (municipales, européennes)
amélioreraient l'identifiabilité.

### 2.9 Pas de modélisation explicite des Français de l'étranger / outre-mer

Le simulateur considère un électorat global. Pas de décomposition régionale
ni de spécificités (vote des Français de l'étranger, DOM-TOM) qui peuvent
basculer un résultat serré.

### 2.10 Pas d'effet "candidat surprise"

Le modèle ne prévoit pas qu'un candidat inattendu (Zemmour 2022, Ravier,
Cazeneuve...) entre en course et redistribue les pools. Cette possibilité
augmente la variance réelle.

---

## 3. Biais inhérents à la démarche (irréductibles)

### 3.1 Validation par calibration tautologique

Le NN apprend parfaitement le simulateur (MAE 0.18). Le CMA-ES optimise le
NN. Donc CMA-ES trouve l'optimum **du simulateur**, pas du monde réel. Sans
validation hors-distribution (sur 2027 réel), la fiabilité prédictive reste
indémontrable.

### 3.2 Encodage des "choix de scénarios"

Les 6 `EXOGENOUS_SCENARIOS` (calme, médian, tempete_2017, crise_pure,
vide_central, tout_max) sont mes choix. Un autre analyste choisirait
d'autres coupes de l'hyperespace.

### 3.3 Métriques choisies = lecture du modèle

J'optimise P(victoire 2T). Mais la "victoire" est un objet binarisé qui
absorbe énormément d'incertitude. Optimiser le score 1er tour ou la
probabilité de qualification donnerait des stratégies différentes.

### 3.4 Pas de calibration adversariale

Le modèle pourrait être attaqué : "et si on inversait Villepin et Bardella ?".
Le modèle devrait prédire Bardella ~22% (réel ~34% en sondages). Il ne le fait
pas naturellement parce que la mass-formule est attachée à Villepin.

---

## 4. Conclusion : comment lire les chiffres

| seuil de confiance | lecture |
|---|---|
| **0.82%** est-il un chiffre fiable ? | Non. C'est un ordre de grandeur. |
| Le **classement** des scénarios est-il fiable ? | Oui, modulo les biais §2. |
| Le **signe** des recommandations (max machine + media + coalition) est-il fiable ? | Modérément. Le modèle dit ça plutôt clairement. |
| Le modèle prédit-il **2027** ? | Non. Il dit "sous ces hypothèses, ça donne ça". |

**Recommandation de lecture** : utiliser les résultats comme **diagnostic**
("Villepin a un profil structurellement difficile, voici lesquelles des 8
variables semblent compter") et NON comme **prédiction** ("Villepin a 0.82%
de chances").

---

## 5. Plan de réduction des biais (v2 et au-delà)

Par priorité décroissante :

1. **Calibration des affinités électorales** sur sondages de transferts
   (jour de vote, OpinionWay/Ifop/Ipsos). 50-100 points de calibration.
2. **Contexte pré-électoral** : remplacer `CONTEXT[year]` rétrospectif par
   indicateurs mesurés 6 mois avant chaque élection (INSEE, OFCE, sondages
   d'opinion).
3. **Affinités par candidat** : décomposer chaque archétype historique en
   profil propre. Multiplie les degrés de liberté mais améliore le fit.
4. **Adversaires adaptatifs** : self-play CMA-ES où chaque archétype optimise
   sa propre P(victoire). Equilibre de Nash.
5. **Modèle temporel** : 12 pas de temps mai 2026 → avril 2027, avec
   dynamique de campagne (débats datés, ralliements, événements).
6. **Boucle LLM (Ollama Turbo)** : déjà scaffold dans `src/llm_param_discovery.py`,
   à activer avec garde-fous anti-redondance et rollback.
