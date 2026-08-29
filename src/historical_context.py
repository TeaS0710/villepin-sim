"""Contexte historique 2002-2022 : mapping candidats → archétypes, contexte Tier 1,
tailles de pools spécifiques à chaque année.

⚠️ Version v2 corrigée. La v1 était bâclée (Chirac→`philippe` p.ex., alors que
le RPR est l'ancêtre direct du LR ; Sarkozy 2007 idem).

Sources principales :
- Wikipédia (résultats officiels + classement politique des candidats)
- vie-publique.fr (notes des candidats)
- France Politique (positionnement)
- Cairn (article "Le vote Bayrou", 2007)

Hypothèses :
- Les ARCHÉTYPES correspondent à des FAMILLES POLITIQUES stables :
    bardella   = extrême-droite (FN/RN/MNR/MPF/Reconquête/DLF)
    retailleau = droite classique (RPR/UMP/LR/DL/CPNT/conservateurs)
    philippe   = centre/centre-droit gouvernemental (LREM, Cap21, Lassalle)
    glucksmann = gauche modérée et écologie (PS/PRG/EELV/MdC)
    melenchon  = gauche radicale (PCF/LO/LCR/NPA/FG/LFI/PT/SP)
    VILLEPIN_EQ = centriste outsider non gouvernemental (UDF/MoDem/EM-2017)
- `philippe` est vacant en 2002, 2007, 2012 : pas de candidat centriste sortant
  à l'époque. C'est documenté comme `[]` dans le mapping ; la base 0.1 (placeholder
  config) reste alors active mais sans poids historique.
"""
from __future__ import annotations


# --------------------------------------------------------------------
# Candidat "Villepin-équivalent" par élection (centriste outsider)
# --------------------------------------------------------------------
VILLEPIN_EQUIVALENT: dict[int, str] = {
    2002: "François Bayrou",       # UDF, 6.84%
    2007: "François Bayrou",       # UDF/MoDem, 18.57% (succès)
    2012: "François Bayrou",       # MoDem, 9.13%
    2017: "Emmanuel Macron",       # EM, 24.01% (succès, devenu président)
    2022: "Valérie Pécresse",      # LR-centre, 4.78% (échec patent)
}


# --------------------------------------------------------------------
# Mapping candidat -> archétype (familles politiques cohérentes 2002-2026)
# Justifications pour les choix non triviaux dans les commentaires.
# --------------------------------------------------------------------
ARCHETYPE_MAPPING: dict[int, dict[str, list[str]]] = {
    2002: {
        # Le Pen = FN, Mégret = MNR (dissidence FN). Pure extrême-droite.
        "bardella":   ["Jean-Marie Le Pen", "Bruno Mégret"],
        # Chirac (RPR = ancêtre direct UMP→LR), Madelin (DL, libéral droite),
        # Saint-Josse (CPNT, conservateurs ruraux droite), Boutin (FRS, droite
        # chrétienne). Pas "philippe" : aucun centriste sortant en 2002.
        "retailleau": ["Jacques Chirac", "Alain Madelin",
                       "Jean Saint-Josse", "Christine Boutin"],
        # Lepage (Cap21, écolo-centriste libérale). Seule candidate
        # vraiment "centre" hors Bayrou.
        "philippe":   ["Corinne Lepage"],
        # PS Jospin (gauche gouv.) + Mamère (Verts) + Taubira (PRG centre-gauche)
        # + Chevènement (MdC républicain souverainiste de gauche, ex-PS).
        "glucksmann": ["Lionel Jospin", "Noël Mamère",
                       "Christiane Taubira", "Jean-Pierre Chevènement"],
        # Hue PCF (gauche communiste), Laguiller LO, Besancenot LCR,
        # Gluckstein PT : toutes nuances trotskystes/communistes radicales.
        "melenchon":  ["Robert Hue", "Arlette Laguiller",
                       "Olivier Besancenot", "Daniel Gluckstein"],
    },
    2007: {
        "bardella":   ["Jean-Marie Le Pen", "Philippe de Villiers"],
                       # Villiers MPF : souverainisme droite radicale, anti-immigration,
                       # plus proche FN qu'UMP : classement bardella plus juste que retailleau.
        # Sarkozy candidat unique de l'UMP (futur LR). Nihous (CPNT)
        # marginal, droite rurale, identique à Saint-Josse 2002.
        "retailleau": ["Nicolas Sarkozy", "Frédéric Nihous"],
        # Pas de centriste sortant en 2007. Bayrou est l'UNIQUE outsider centriste.
        "philippe":   [],
        # PS Royal + Verts Voynet.
        "glucksmann": ["Ségolène Royal", "Dominique Voynet"],
        # PCF Buffet, LCR Besancenot, LO Laguiller, altermondialistes Bové,
        # PT Schivardi.
        "melenchon":  ["Marie-George Buffet", "Olivier Besancenot",
                       "Arlette Laguiller", "José Bové", "Gérard Schivardi"],
    },
    2012: {
        "bardella":   ["Marine Le Pen"],
        # Sarkozy sortant UMP. Dupont-Aignan (DLR) souverainiste droite mais
        # pas extrême-droite (encore allié RN seulement à partir 2017).
        # Cheminade (SP), souverainiste folklorique, droite par défaut.
        "retailleau": ["Nicolas Sarkozy", "Nicolas Dupont-Aignan",
                       "Jacques Cheminade"],
        "philippe":   [],  # vide
        # Hollande PS + Joly EELV.
        "glucksmann": ["François Hollande", "Eva Joly"],
        # Mélenchon FG (PCF+Parti de Gauche), Poutou NPA, Arthaud LO.
        "melenchon":  ["Jean-Luc Mélenchon", "Philippe Poutou",
                       "Nathalie Arthaud"],
    },
    2017: {
        # Le Pen FN + Dupont-Aignan (DLF allié RN au 2T).
        "bardella":   ["Marine Le Pen", "Nicolas Dupont-Aignan"],
        # Fillon LR + Asselineau (UPR souverainiste anti-UE) + Cheminade.
        "retailleau": ["François Fillon", "François Asselineau",
                       "Jacques Cheminade"],
        # Lassalle (RES) ruraliste centriste : pas extrême ni de gauche,
        # plutôt "centre populiste rural". Catégorisé `philippe` comme
        # centriste "non-villepin".
        "philippe":   ["Jean Lassalle"],
        # PS Hamon (centre-gauche écolo).
        "glucksmann": ["Benoît Hamon"],
        # LFI Mélenchon, NPA Poutou, LO Arthaud.
        "melenchon":  ["Jean-Luc Mélenchon", "Philippe Poutou",
                       "Nathalie Arthaud"],
    },
    2022: {
        # RN Le Pen + Reconquête Zemmour + DLF Dupont-Aignan.
        "bardella":   ["Marine Le Pen", "Éric Zemmour", "Nicolas Dupont-Aignan"],
        # LR Pécresse seule.
        "retailleau": ["Valérie Pécresse"],
        # Macron sortant LREM, "philippe" = centre gouvernemental ;
        # Lassalle RES centriste rural.
        "philippe":   ["Emmanuel Macron", "Jean Lassalle"],
        # EELV Jadot + PS Hidalgo.
        "glucksmann": ["Yannick Jadot", "Anne Hidalgo"],
        # LFI Mélenchon + PCF Roussel + NPA Poutou + LO Arthaud.
        "melenchon":  ["Jean-Luc Mélenchon", "Fabien Roussel",
                       "Philippe Poutou", "Nathalie Arthaud"],
    },
}


# --------------------------------------------------------------------
# Tailles de pools spécifiques à chaque année (v2 : empiriques).
# Calibrées sur les agrégats de la mapping ci-dessus. Inertias gardées
# constantes (ce sont des propriétés structurelles plus stables que la taille).
# Si une famille politique est faible (ex: gauche en 2017), la taille du pool
# est réduite ; les autres pools restent inchangés en tailles fixes.
# La somme = 100 (cas idéal, écart < 2 pts toléré).
# --------------------------------------------------------------------
YEAR_POOLS: dict[int, dict[str, dict[str, float]]] = {
    2002: {
        "rn":      {"size": 22.0, "inertia": 0.78},  # FN 17 + Mégret 2 + sympathisants
        "central": {"size":  9.0, "inertia": 0.40},  # Bayrou 7 + Lepage 2
        "gauche":  {"size": 33.0, "inertia": 0.50},  # PS+Verts+Chevènement+PRG
        "lfi":     {"size": 15.0, "inertia": 0.65},  # PCF+LO+LCR+PT
        "lr":      {"size": 21.0, "inertia": 0.60},  # RPR+DL+CPNT+FRS
        "indecis": {"size":  5.0, "inertia": 0.15},
    },
    2007: {
        "rn":      {"size": 13.0, "inertia": 0.78},  # FN au plus bas (10.4)
        "central": {"size": 21.0, "inertia": 0.40},  # Bayrou 18.6 + résiduels
        "gauche":  {"size": 30.0, "inertia": 0.50},  # PS Royal + Verts
        "lfi":     {"size": 10.0, "inertia": 0.65},
        "lr":      {"size": 32.0, "inertia": 0.60},  # Sarkozy 31.2 + Nihous
        "indecis": {"size":  5.0, "inertia": 0.15},
    },
    2012: {
        "rn":      {"size": 19.0, "inertia": 0.80},  # Marine LP 17.9
        "central": {"size": 10.0, "inertia": 0.40},  # Bayrou 9.1
        "gauche":  {"size": 33.0, "inertia": 0.50},  # PS Hollande + EELV
        "lfi":     {"size": 13.0, "inertia": 0.65},  # FG + NPA + LO
        "lr":      {"size": 29.0, "inertia": 0.60},  # Sarko 27.2 + DLR + Cheminade
        "indecis": {"size":  5.0, "inertia": 0.15},
    },
    2017: {
        "rn":      {"size": 26.0, "inertia": 0.82},  # Le Pen 21.3 + DA
        "central": {"size": 25.0, "inertia": 0.45},  # Macron 24 + Lassalle 1.2
        "gauche":  {"size":  7.0, "inertia": 0.50},  # Hamon 6.4 (PS effondré)
        "lfi":     {"size": 21.0, "inertia": 0.65},  # Mélenchon 19.6 + autres
        "lr":      {"size": 21.0, "inertia": 0.60},  # Fillon 20 + autres
        "indecis": {"size":  5.0, "inertia": 0.15},
    },
    2022: {
        "rn":      {"size": 32.0, "inertia": 0.85},  # Le Pen 23 + Zemmour 7 + DA
        "central": {"size": 31.0, "inertia": 0.45},  # Macron 27.9 + Lassalle 3.1
        "gauche":  {"size":  7.0, "inertia": 0.50},  # Jadot 4.6 + Hidalgo 1.8
        "lfi":     {"size": 26.0, "inertia": 0.65},  # Mélenchon 22 + PCF + NPA + LO
        "lr":      {"size":  5.0, "inertia": 0.60},  # Pécresse 4.8 (LR effondré)
        "indecis": {"size":  5.0, "inertia": 0.15},
    },
}


# --------------------------------------------------------------------
# Contexte Tier 1 par élection : estimations à dire d'expert (sources :
# sondages d'époque, indicateurs INSEE, événements géopolitiques, presse).
# --------------------------------------------------------------------
CONTEXT: dict[int, dict[str, float]] = {
    # Bayrou 2002 : post-11/9 modéré, UDF en survie.
    2002: {
        "crisis": 0.35, "central_collapse": 0.20, "volatility": 0.55,
        "anti_extreme_pressure": 0.30, "campaign_machine": 0.40,
        "thematic_breadth": 0.45, "media_performance": 0.50,
        "coalition_building": 0.30,
    },
    # Bayrou 2007 : apogée Bayrou (banlieues 2005, dette publique au cœur).
    2007: {
        "crisis": 0.40, "central_collapse": 0.30, "volatility": 0.60,
        "anti_extreme_pressure": 0.40, "campaign_machine": 0.55,
        "thematic_breadth": 0.65, "media_performance": 0.80,
        "coalition_building": 0.50,
    },
    # Bayrou 2012 : isolé, crise euro/dette, bipartisme PS/UMP.
    2012: {
        "crisis": 0.55, "central_collapse": 0.15, "volatility": 0.40,
        "anti_extreme_pressure": 0.45, "campaign_machine": 0.40,
        "thematic_breadth": 0.50, "media_performance": 0.55,
        "coalition_building": 0.20,
    },
    # Macron 2017 : tempête parfaite (terror, Penelopegate, PS effondré).
    2017: {
        "crisis": 0.55, "central_collapse": 0.85, "volatility": 0.75,
        "anti_extreme_pressure": 0.65, "campaign_machine": 0.80,
        "thematic_breadth": 0.85, "media_performance": 0.85,
        "coalition_building": 0.80,
    },
    # Pécresse 2022 : guerre Ukraine, campagne ratée (Zenith).
    2022: {
        "crisis": 0.70, "central_collapse": 0.30, "volatility": 0.45,
        "anti_extreme_pressure": 0.50, "campaign_machine": 0.35,
        "thematic_breadth": 0.45, "media_performance": 0.20,
        "coalition_building": 0.30,
    },
}


def get_actual_score(historical_df, year: int) -> float:
    """Score réel 1T du Villepin-équivalent pour cette année."""
    name = VILLEPIN_EQUIVALENT[year]
    row = historical_df[(historical_df["year"] == year) & (historical_df["candidate"] == name)]
    if row.empty:
        raise ValueError(f"Pas de ligne pour {name} en {year}")
    return float(row["pct_exprimes"].iloc[0])


def aggregate_competitor_bases(historical_df, year: int) -> dict[str, float]:
    """Bases agrégées par archétype pour une année (somme des scores
    historiques des candidats mappés).
    """
    mapping = ARCHETYPE_MAPPING[year]
    df_year = historical_df[historical_df["year"] == year]
    out: dict[str, float] = {}
    for arch, candidates in mapping.items():
        total = 0.0
        for c in candidates:
            row = df_year[df_year["candidate"] == c]
            if not row.empty:
                total += float(row["pct_exprimes"].iloc[0])
        out[arch] = max(total, 0.1)
    return out


def year_pools(year: int) -> dict[str, dict[str, float]] | None:
    """Tailles de pools spécifiques à l'année (None si pas défini)."""
    return YEAR_POOLS.get(year)
