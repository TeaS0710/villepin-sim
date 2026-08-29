"""Variante multiclasses du contexte historique : 8 archétypes (au lieu de 6),
représentatifs des vraies familles politiques observées 2002-2027.

Familles (8 archétypes) :
    extreme_droite     : FN/RN core + Reconquête + MNR
    souverainiste      : DLF, UPR, MPF, S&P (souverainistes hors-RN)
    droite_classique   : RPR/UMP/LR + alliés conservateurs (CPNT, DL, FRS)
    centre_gouv        : Renaissance/Horizons/EM (sortants gouvernementaux)
    centre_outsider    : UDF/MoDem/Cap21/RES/LFH (centristes hors gouvernement)
    gauche_socdem      : PS, EELV, PRG, MdC, alliés modérés
    gauche_radicale    : FG, LFI, PCF moderne
    extreme_gauche     : NPA, LO, LCR, PT, PCF trotskysants

Drop-in remplaçant pour `historical_context.py` (interface compatible avec
`calibration.py`, `historical_validation.py`, `physical_model.py`).
"""
from __future__ import annotations


# --------------------------------------------------------------------
# Candidat "Villepin-équivalent" par élection (centriste outsider).
# Identique à v6pools : Villepin est un centriste outsider.
# --------------------------------------------------------------------
VILLEPIN_EQUIVALENT: dict[int, str] = {
    2002: "François Bayrou",       # UDF, 6.84%
    2007: "François Bayrou",       # UDF/MoDem, 18.57%
    2012: "François Bayrou",       # MoDem, 9.13%
    2017: "Emmanuel Macron",       # EM, 24.01%
    2022: "Valérie Pécresse",      # LR-centre, 4.78% (proxy faute de mieux)
}


# --------------------------------------------------------------------
# Mapping 61 candidats -> 8 archétypes par élection.
# Sources : Wikipédia (résultats officiels) + classement politique.
# --------------------------------------------------------------------
ARCHETYPE_MAPPING: dict[int, dict[str, list[str]]] = {
    2002: {
        "extreme_droite":   ["Jean-Marie Le Pen", "Bruno Mégret"],
        # Pas de souverainiste majeur en 2002 hors Le Pen camp.
        "souverainiste":    [],
        # RPR (ancêtre direct UMP→LR), DL (libéral), CPNT (rural droite),
        # FRS (droite chrétienne).
        "droite_classique": ["Jacques Chirac", "Alain Madelin",
                              "Jean Saint-Josse", "Christine Boutin"],
        # Pas de centriste sortant en 2002.
        "centre_gouv":      [],
        # Bayrou + Lepage (Cap21).
        "centre_outsider":  ["François Bayrou", "Corinne Lepage"],
        # PS (Jospin) + Verts (Mamère) + PRG (Taubira) + MdC (Chevènement).
        "gauche_socdem":    ["Lionel Jospin", "Noël Mamère",
                              "Christiane Taubira", "Jean-Pierre Chevènement"],
        # PCF modéré (Hue) en gauche radicale.
        "gauche_radicale":  ["Robert Hue"],
        # LO + LCR + PT (trotskystes/extrême).
        "extreme_gauche":   ["Arlette Laguiller", "Olivier Besancenot",
                              "Daniel Gluckstein"],
    },
    2007: {
        "extreme_droite":   ["Jean-Marie Le Pen"],
        # MPF Villiers, souverainisme droite radicale.
        "souverainiste":    ["Philippe de Villiers"],
        # UMP Sarkozy + CPNT Nihous.
        "droite_classique": ["Nicolas Sarkozy", "Frédéric Nihous"],
        "centre_gouv":      [],
        # Bayrou seul centriste outsider.
        "centre_outsider":  ["François Bayrou"],
        # PS Royal + Verts Voynet + altermondialiste Bové (gauche écologiste).
        "gauche_socdem":    ["Ségolène Royal", "Dominique Voynet", "José Bové"],
        # PCF Buffet.
        "gauche_radicale":  ["Marie-George Buffet"],
        # LCR + LO + PT.
        "extreme_gauche":   ["Olivier Besancenot", "Arlette Laguiller",
                              "Gérard Schivardi"],
    },
    2012: {
        "extreme_droite":   ["Marine Le Pen"],
        # DLR Dupont-Aignan + S&P Cheminade (souverainistes).
        "souverainiste":    ["Nicolas Dupont-Aignan", "Jacques Cheminade"],
        # UMP Sarkozy sortant.
        "droite_classique": ["Nicolas Sarkozy"],
        "centre_gouv":      [],
        # Bayrou MoDem.
        "centre_outsider":  ["François Bayrou"],
        # PS Hollande + EELV Joly.
        "gauche_socdem":    ["François Hollande", "Eva Joly"],
        # Front de Gauche Mélenchon (PCF + Parti de Gauche).
        "gauche_radicale":  ["Jean-Luc Mélenchon"],
        # NPA Poutou + LO Arthaud.
        "extreme_gauche":   ["Philippe Poutou", "Nathalie Arthaud"],
    },
    2017: {
        "extreme_droite":   ["Marine Le Pen"],
        # DLF Dupont-Aignan (allié RN 2T) + UPR Asselineau + S&P Cheminade.
        "souverainiste":    ["Nicolas Dupont-Aignan", "François Asselineau",
                              "Jacques Cheminade"],
        # LR Fillon.
        "droite_classique": ["François Fillon"],
        # Macron EM, le centre vraiment gouvernemental (devenu président).
        "centre_gouv":      ["Emmanuel Macron"],
        # Lassalle RES (centriste rural).
        "centre_outsider":  ["Jean Lassalle"],
        # PS Hamon (centre-gauche écolo).
        "gauche_socdem":    ["Benoît Hamon"],
        # LFI Mélenchon.
        "gauche_radicale":  ["Jean-Luc Mélenchon"],
        # NPA Poutou + LO Arthaud.
        "extreme_gauche":   ["Philippe Poutou", "Nathalie Arthaud"],
    },
    2022: {
        # RN Le Pen + Reconquête Zemmour.
        "extreme_droite":   ["Marine Le Pen", "Éric Zemmour"],
        # DLF Dupont-Aignan.
        "souverainiste":    ["Nicolas Dupont-Aignan"],
        # LR Pécresse.
        "droite_classique": ["Valérie Pécresse"],
        # Macron sortant LREM.
        "centre_gouv":      ["Emmanuel Macron"],
        # Lassalle RES.
        "centre_outsider":  ["Jean Lassalle"],
        # EELV Jadot + PS Hidalgo.
        "gauche_socdem":    ["Yannick Jadot", "Anne Hidalgo"],
        # LFI Mélenchon + PCF Roussel.
        "gauche_radicale":  ["Jean-Luc Mélenchon", "Fabien Roussel"],
        # NPA Poutou + LO Arthaud.
        "extreme_gauche":   ["Philippe Poutou", "Nathalie Arthaud"],
    },
}


# --------------------------------------------------------------------
# Tailles de pools spécifiques à chaque année (8 pools).
# Calibrés sur les agrégats observés ; inerties stables.
# Somme = 100 (cas idéal).
# --------------------------------------------------------------------
YEAR_POOLS: dict[int, dict[str, dict[str, float]]] = {
    2002: {
        "extreme_droite":  {"size": 19.0, "inertia": 0.80},   # FN 17 + Mégret 2
        "souverainiste":   {"size":  2.0, "inertia": 0.55},   # marginal en 2002
        "droite_classique":{"size": 21.0, "inertia": 0.60},   # RPR 20 + DL 4 + CPNT 4 + FRS 1
        "centre_gouv":     {"size":  1.0, "inertia": 0.40},   # quasi vide en 2002
        "centre_outsider": {"size":  9.0, "inertia": 0.45},   # Bayrou 7 + Lepage 2
        "gauche_socdem":   {"size": 31.0, "inertia": 0.50},   # PS+Verts+Chevènement+PRG
        "gauche_radicale": {"size":  4.0, "inertia": 0.65},   # PCF Hue 3.4
        "extreme_gauche":  {"size": 11.0, "inertia": 0.70},   # LO 5.7 + LCR 4.3 + PT 0.5
        "indecis":         {"size":  2.0, "inertia": 0.15},
    },
    2007: {
        "extreme_droite":  {"size": 11.0, "inertia": 0.80},   # FN au plus bas (10.4)
        "souverainiste":   {"size":  3.0, "inertia": 0.55},   # MPF Villiers 2.2
        "droite_classique":{"size": 33.0, "inertia": 0.60},   # Sarkozy 31.2 + Nihous
        "centre_gouv":     {"size":  1.0, "inertia": 0.40},
        "centre_outsider": {"size": 19.0, "inertia": 0.45},   # Bayrou 18.6 + résiduels
        "gauche_socdem":   {"size": 30.0, "inertia": 0.50},   # PS Royal + Verts + Bové
        "gauche_radicale": {"size":  2.0, "inertia": 0.65},   # PCF Buffet
        "extreme_gauche":  {"size":  3.0, "inertia": 0.70},   # LCR + LO + PT
        "indecis":         {"size":  2.0, "inertia": 0.15},
    },
    2012: {
        "extreme_droite":  {"size": 18.0, "inertia": 0.82},   # Marine LP 17.9
        "souverainiste":   {"size":  3.0, "inertia": 0.55},   # DLR Dupont-Aignan + Cheminade
        "droite_classique":{"size": 27.0, "inertia": 0.60},   # Sarko 27.2
        "centre_gouv":     {"size":  1.0, "inertia": 0.40},
        "centre_outsider": {"size":  9.0, "inertia": 0.45},   # Bayrou 9.1
        "gauche_socdem":   {"size": 31.0, "inertia": 0.50},   # PS Hollande + EELV
        "gauche_radicale": {"size":  9.0, "inertia": 0.65},   # FG Mélenchon 11.1 (-marge)
        "extreme_gauche":  {"size":  2.0, "inertia": 0.70},   # NPA + LO
        "indecis":         {"size":  2.0, "inertia": 0.15},
    },
    2017: {
        "extreme_droite":  {"size": 21.0, "inertia": 0.85},   # Le Pen 21.3
        "souverainiste":   {"size":  6.0, "inertia": 0.55},   # DLF DA 4.7 + UPR + Cheminade
        "droite_classique":{"size": 21.0, "inertia": 0.60},   # Fillon 20
        "centre_gouv":     {"size": 24.0, "inertia": 0.45},   # Macron 24
        "centre_outsider": {"size":  1.5, "inertia": 0.45},   # Lassalle 1.2
        "gauche_socdem":   {"size":  7.0, "inertia": 0.50},   # Hamon 6.4 (PS effondré)
        "gauche_radicale": {"size": 19.0, "inertia": 0.70},   # Mélenchon 19.6
        "extreme_gauche":  {"size":  2.0, "inertia": 0.75},   # NPA + LO
        "indecis":         {"size":  2.5, "inertia": 0.15},
    },
    2022: {
        "extreme_droite":  {"size": 30.0, "inertia": 0.85},   # Le Pen 23 + Zemmour 7
        "souverainiste":   {"size":  2.0, "inertia": 0.55},   # DLF DA
        "droite_classique":{"size":  5.0, "inertia": 0.60},   # Pécresse 4.8 (LR effondré)
        "centre_gouv":     {"size": 28.0, "inertia": 0.45},   # Macron 27.9
        "centre_outsider": {"size":  3.0, "inertia": 0.45},   # Lassalle 3.1
        "gauche_socdem":   {"size":  7.0, "inertia": 0.50},   # Jadot 4.6 + Hidalgo 1.8
        "gauche_radicale": {"size": 24.0, "inertia": 0.70},   # Mélenchon 22 + Roussel 2.3
        "extreme_gauche":  {"size":  1.0, "inertia": 0.75},   # NPA + LO
        "indecis":         {"size":  2.0, "inertia": 0.15},
    },
}


# --------------------------------------------------------------------
# Contexte Tier 1 par élection (estimations à dire d'expert).
# Inchangé par rapport à v6pools : ces 8 dimensions sont indépendantes
# du nombre de pools.
# --------------------------------------------------------------------
CONTEXT: dict[int, dict[str, float]] = {
    2002: {
        "crisis": 0.35, "central_collapse": 0.20, "volatility": 0.55,
        "anti_extreme_pressure": 0.30, "campaign_machine": 0.40,
        "thematic_breadth": 0.45, "media_performance": 0.50,
        "coalition_building": 0.30,
    },
    2007: {
        "crisis": 0.40, "central_collapse": 0.30, "volatility": 0.60,
        "anti_extreme_pressure": 0.40, "campaign_machine": 0.55,
        "thematic_breadth": 0.65, "media_performance": 0.80,
        "coalition_building": 0.50,
    },
    2012: {
        "crisis": 0.55, "central_collapse": 0.15, "volatility": 0.40,
        "anti_extreme_pressure": 0.45, "campaign_machine": 0.40,
        "thematic_breadth": 0.50, "media_performance": 0.55,
        "coalition_building": 0.20,
    },
    2017: {
        "crisis": 0.55, "central_collapse": 0.85, "volatility": 0.75,
        "anti_extreme_pressure": 0.65, "campaign_machine": 0.80,
        "thematic_breadth": 0.85, "media_performance": 0.85,
        "coalition_building": 0.80,
    },
    2022: {
        "crisis": 0.70, "central_collapse": 0.30, "volatility": 0.45,
        "anti_extreme_pressure": 0.50, "campaign_machine": 0.35,
        "thematic_breadth": 0.45, "media_performance": 0.20,
        "coalition_building": 0.30,
    },
}


# --------------------------------------------------------------------
# Helpers (interface identique à historical_context.py)
# --------------------------------------------------------------------
def get_actual_score(historical_df, year: int) -> float:
    """Score réel 1T du Villepin-équivalent pour cette année."""
    name = VILLEPIN_EQUIVALENT[year]
    row = historical_df[(historical_df["year"] == year)
                        & (historical_df["candidate"] == name)]
    if row.empty:
        raise ValueError(f"Pas de ligne pour {name} en {year}")
    return float(row["pct_exprimes"].iloc[0])


def aggregate_competitor_bases(historical_df, year: int) -> dict[str, float]:
    """Bases agrégées par archétype pour une année (somme des scores réels
    des candidats mappés). Floor 0.1 pour préserver la signature numérique.
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
    """Tailles de pools spécifiques à l'année (None si non défini)."""
    return YEAR_POOLS.get(year)
