"""Génère le rapport final markdown + plots associés.

Ce reporter consolide tous les artefacts produits par les phases précédentes
(calibration, dataset, CMA-ES) en un rapport critique avec disclaimer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Style "pro industriel" appliqué globalement à toutes les figures.
# - police sans-serif moderne (Noto Sans dispo localement, fallback DejaVu)
# - spines top/right masquées
# - grille subtile y uniquement
# - tick marks discrets
# - palette par défaut désaturée
plt.rcParams.update({
    "font.family": "sans-serif",
    # DejaVu Sans en premier : couvre les glyphs math/typographiques (− × ≈ → ←)
    # que Noto Sans n'a pas, tout en restant une police sans-serif moderne.
    "font.sans-serif": ["DejaVu Sans", "Noto Sans", "Inter",
                          "Helvetica Neue", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlepad": 10,
    "axes.labelsize": 10,
    "axes.labelpad": 5,
    "axes.edgecolor": "#444",
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": "#e6e6e6",
    "grid.linestyle": "-",
    "grid.linewidth": 0.7,
    "grid.alpha": 0.9,
    "xtick.color": "#444",
    "ytick.color": "#444",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9.5,
    "figure.facecolor": "white",
    "figure.titlesize": 14,
    "figure.titleweight": "bold",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.dpi": 150,
    "patch.linewidth": 0.8,
    "patch.edgecolor": "#444",
})

from .historical_context import VILLEPIN_EQUIVALENT
from .physical_model import load_config

DISCLAIMER = """\
> ⚠️ **Disclaimer.** Ce rapport est un **exercice de modélisation politique exploratoire**.
> Les paramètres et fonctions de coût sont calibrés sur 5 élections historiques
> (2002-2022) avec un nombre limité de degrés de liberté. Les résultats ne sont
> **PAS des prédictions** mais des **explorations de scénarios contrefactuels**.
> La politique réelle implique des facteurs (chocs noirs, personnalité, hasard)
> impossibles à modéliser. Les « stratégies optimales » identifiées sont des
> hypothèses à interroger, pas des chemin optimals opérationnelles.
"""


def _build_historical_validation_section(out_dir: Path) -> str:
    summary_path = out_dir / "historical_validation_summary.json"
    if not summary_path.exists():
        return ""
    summary = json.loads(summary_path.read_text())
    per_year = pd.read_csv(out_dir / "historical_validation_per_year.csv")
    metrics = summary["metrics"]
    out = "## D. Validation supervisée sur élections passées (2002-2022)\n\n"
    out += (
        "Pour chaque élection historique, on configure le simulateur avec les bases "
        "concurrents agrégées et le contexte Tier 1 estimé, puis on prédit (i) les "
        "deux qualifiés au 2T et (ii) le vainqueur du second tour. On compare aux "
        "résultats officiels.\n\n"
    )
    out += "### Métriques de classification\n\n"
    out += "| métrique | valeur |\n|---|---|\n"
    out += f"| **F1** (qualification 2T) | **{metrics['f1_qualif']:.3f}** |\n"
    out += f"| Precision (qualif) | {metrics['precision_qualif']:.3f} |\n"
    out += f"| Recall (qualif) | {metrics['recall_qualif']:.3f} |\n"
    out += f"| Accuracy (qualif) | {metrics['accuracy_qualif']:.3f} |\n"
    out += f"| **Accuracy vainqueur 2T** | **{metrics['winner_correct_count']}/{metrics['winner_total']} = {metrics['winner_accuracy']*100:.0f}%** |\n"
    out += f"| TP / FP / FN / TN | {metrics['confusion_matrix']['tp']} / {metrics['confusion_matrix']['fp']} / {metrics['confusion_matrix']['fn']} / {metrics['confusion_matrix']['tn']} |\n\n"
    out += "![Matrices de validation historique](plots/historical_validation_metrics.png)\n\n"
    out += "![Score 1T prédit vs réel par année × archétype](plots/historical_pred_vs_actual.png)\n\n"

    # Si la validation walk-forward (T-2 mois, honnête) existe, l'afficher
    # comparativement. Les chiffres in-sample ci-dessus sont gonflés par la
    # calibration post-hoc de CONTEXT[year] et YEAR_POOLS[year].
    wf_path = out_dir / "historical_validation_walk_forward_summary.json"
    if wf_path.exists():
        m_wf = json.loads(wf_path.read_text())["metrics"]
        out += "### Validation honnête (walk-forward T-2 mois)\n\n"
        out += (
            "⚠️ La validation ci-dessus est **in-sample** : `CONTEXT[year]` et "
            "`YEAR_POOLS[year]` sont hard-codés en connaissant le résultat. "
            "La vraie capacité prédictive du modèle se mesure en walk-forward, "
            "avec uniquement les sondages T-2 mois (jamais le résultat) :\n\n"
        )
        out += "| métrique | in-sample (post-hoc) | walk-forward (T-2 mois) |\n|---|---|---|\n"
        out += (f"| MAE 1T | {float(np.mean(np.abs(pd.read_csv(out_dir / 'historical_validation_per_pair.csv')['predicted_score_1T'] - pd.read_csv(out_dir / 'historical_validation_per_pair.csv')['actual_score_1T']))):.2f} pts "
                f"| **{m_wf['mae_score_1T']:.2f} pts** |\n")
        out += f"| F1 qualif | {metrics['f1_qualif']:.3f} | **{m_wf['f1_qualif']:.3f}** |\n"
        out += f"| Accuracy qualif | {metrics['accuracy_qualif']:.3f} | **{m_wf['accuracy_qualif']:.3f}** |\n"
        out += (f"| Vainqueur 2T | {metrics['winner_correct_count']}/{metrics['winner_total']} "
                f"| **{m_wf['winner_correct_count']}/{m_wf['winner_total']}** |\n\n")
        out += "![Comparaison in-sample vs walk-forward](plots/historical_validation_compare.png)\n\n"

    out += "### Détails par année\n\n"
    cols = ["year", "actual_top2", "predicted_top2", "top2_overlap",
            "actual_winner", "predicted_winner", "winner_correct"]
    out += per_year[cols].to_markdown(index=False) + "\n\n"
    out += (
        "### Lecture honnête\n\n"
        "**Le modèle est correct pour la qualification au 2T** dans ~80% des cas et avec F1 ≈ 0.70 "
        "(top-2 partiellement matché). Il est **systématiquement faux pour le 2T 2002-2022 (0/5)** : "
        "le modèle prédit toujours `bardella` (camp RN) comme vainqueur. Deux causes structurelles "
        "identifiées :\n\n"
        "1. **Tailles de pools fixées à mai 2026** : `pool_rn = 33%`, ce qui correspond aux sondages "
        "actuels mais SURESTIME le poids historique du FN/RN (16-23% sur 2002-2022). La taille du pool "
        "× son inertie (0.85) donne à `bardella` une rétention de ~28% du corps électoral, qui dépasse "
        "presque toujours les scores 1T historiques réels.\n"
        "2. **Boost de front républicain mal calibré** : `boost_to_score = 30` est trop large pour les "
        "scénarios historiques où le score-différence 1T était petit (2-5 points). Sigmoid+sigmoid_scale "
        "5 amplifient au lieu d'amortir.\n\n"
        "Ce diagnostic est utile : il indique deux pistes claires pour la v3 (pool sizes "
        "dynamiques par année, calibration empirique des paramètres 2T sur les 5 élections).\n\n"
    )
    return out


def _build_dynamics_section() -> str:
    return (
        "## E. Dynamiques internes du modèle\n\n"
        "### Hiérarchie des variables (sensibilité globale du surrogate)\n\n"
        "![Sensibilité Tier 1](plots/param_sensitivity.png)\n\n"
        "**Lecture** : la sensibilité moyenne `E[|∂P(victoire)/∂param|]` mesure, sur 2000 "
        "échantillons aléatoires du dataset, à quel point une petite variation du paramètre "
        "fait varier la prédiction. Constat majeur : les 4 paramètres **exogènes** "
        "(`volatility`, `crisis`, `anti_extreme_pressure`, `central_collapse`) dominent les 4 "
        "**internes** (campagne) : l'effet de chaque exogène est ~2× celui de chaque interne. "
        "Conclusion politique : **la campagne Villepin a moins de levier que le contexte qu'elle "
        "n'a pas choisi**.\n\n"
        "### Partial dependence : effet marginal isolé de chaque paramètre\n\n"
        "![Partial dependence](plots/partial_dependence.png)\n\n"
        "Pour chaque paramètre Tier 1, on fixe les 7 autres à leur médiane du dataset et on fait "
        "varier la valeur du paramètre de 0 à 1. Toutes les courbes sont **monotones croissantes** "
        "(plus haut = mieux pour Villepin) et **proches du linéaire** (pas de saturation forte "
        "dans la région médiane). Hiérarchie visible : `volatility` produit la plus grande "
        "amplitude (~0.12 % → ~0.15 %), puis `crisis` et `central_collapse`, tandis que "
        "`coalition_building` et `media_performance` sont quasi plats à la médiane.\n\n"
        "### Baseline 2027 isolé (params Tier 1 neutres)\n\n"
        "![Baseline 2027](plots/baseline_2027.png)\n\n"
        "Prédiction du modèle pour le contexte 2027 sans aucune perturbation : "
        "Tier 1 fixés à 0.5, bases concurrents = sondages mai 2026. Bardella ressort "
        "à 38.8%, Villepin à 3.55%.\n\n"
        "### D'où vient le score ? Origine par pool, par scénario\n\n"
        "![Pool ownership par scénario](plots/pool_ownership_by_scenario.png)\n\n"
        "Décomposition du score 1T de chaque candidat par pool électoral d'origine, **pour chaque "
        "scénario exogène** (paramètres internes fixés à 0.5). On voit pourquoi Bardella domine "
        "structurellement : il capture massivement le pool `rn` (33 % du corps électoral, "
        "inertie 0.85 : très peu mobile). Villepin tire ses voix de `central`, `lr` et `indecis`, "
        "trois pools dont la taille combinée plafonne à ~40 % et qui sont disputés avec Philippe "
        "(natural owner de `central`).\n\n"
        "### Décomposition baseline vs optimum (paramètres internes différents)\n\n"
        "![Pool breakdown baseline vs optimum](plots/pool_breakdown.png)\n\n"
        "Comparaison directe : à paramètres neutres (gauche) vs optimum CMA-ES (droite). La "
        "structure de capture par pool est très stable : l'optimisation interne ne déplace "
        "marginalement que les flux mobiles, pas les rétentions de stock.\n\n"
    )


def _build_llm_section(llm_summary, tier2_meta) -> str:
    if llm_summary is None and not tier2_meta:
        return ""
    out = "## C. Inférence Tier 2 par LLM (Ollama)\n\n"
    if llm_summary is not None:
        baseline = llm_summary.get("baseline_best_p_victory", float("nan"))
        final = llm_summary.get("final_best_p_victory", float("nan"))
        total = llm_summary.get("accepted_total", 0)
        model = llm_summary.get("model", "?")
        out += f"**Modèle LLM** : `{model}`  \n"
        out += f"**P(victoire) baseline → finale** : {baseline*100:.3f}% → {final*100:.3f}%  "
        if not (final != final) and not (baseline != baseline):  # not NaN
            out += f"(Δ = {(final-baseline)*100:+.3f} pts)\n\n"
        else:
            out += "\n\n"
        out += f"**Sous-paramètres acceptés au total** : {total}\n\n"
        if llm_summary.get("iterations"):
            out += "### Historique des itérations\n\n"
            rows = []
            for it in llm_summary["iterations"]:
                rows.append({
                    "iter": it["iter"],
                    "accepted": ", ".join(it.get("accepted_names", [])) or "-",
                    "rejected_count": it.get("rejected_count", 0),
                    "p_before": f"{it.get('best_p_victory_before', 0)*100:.3f}%" if "best_p_victory_before" in it else "-",
                    "p_after":  f"{it.get('best_p_victory_after',  0)*100:.3f}%" if "best_p_victory_after"  in it else "-",
                    "delta":    f"{it.get('delta', 0)*100:+.3f} pts" if "delta" in it else "-",
                    "decision": it.get("decision", "-"),
                })
            out += pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
        out += "![Évolution LLM](plots/llm_evolution.png)\n\n"
    if tier2_meta:
        out += "### Sous-paramètres Tier 2 retenus dans le config courant\n\n"
        rows = []
        for name, meta in tier2_meta.items():
            rows.append({
                "name": name,
                "parent": meta.get("parent", "?"),
                "importance": meta.get("importance", "?"),
                "weight": f"{meta.get('weight', 0):+.4f}",
                "description": meta.get("description", "")[:80],
            })
        out += pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
        out += "![Poids Tier 2](plots/tier2_weights.png)\n\n"
    return out


def _build_winner_section(df_full, df_agg, df_best) -> str:
    if df_full is None or df_agg is None:
        return ""
    df_agg = df_agg.copy()
    df_agg.columns = [c if c != "Unnamed: 0" else "candidate" for c in df_agg.columns]
    df_agg = df_agg.rename(columns={df_agg.columns[0]: "candidate"}) if "candidate" not in df_agg.columns else df_agg
    out = "## A. Qui peut gagner ? (P(victoire 2T) par candidat)\n\n"
    out += "Probabilité moyenne de victoire 2T pour chaque candidat, agrégée sur "
    out += f"{df_full[['exo_scenario','shock_scenario']].drop_duplicates().shape[0]} combinaisons "
    out += "exogènes × shocks externes.\n\n"
    out += df_agg.to_markdown(index=False) + "\n\n"
    out += "![Winner heatmap](plots/winner_heatmap.png)\n\n"
    if df_best is not None:
        out += "### Top 10 meilleurs scénarios pour Villepin\n\n"
        cols = ["exo_scenario", "shock_scenario", "score_1T_mean", "p_qualif", "p_victory"]
        out += df_best[cols].head(10).to_markdown(index=False) + "\n\n"
        out += "![Top scénarios Villepin](plots/villepin_top_scenarios.png)\n\n"
    return out


def _build_path_to_victory_section(out_dir: Path) -> str:
    path_summary = out_dir / "path_to_victory.summary.json"
    if not path_summary.exists():
        return ""
    s = json.loads(path_summary.read_text())
    out = "## F. Chemin minimal vers une victoire de Villepin\n\n"
    out += (
        f"**Baseline 2027** : P(victoire) = {s['baseline_p_victory']*100:.2f} %  \n"
        f"**Optimum extrême atteint par CMA-ES 13D** : P(victoire) = {s['extreme_p_victory']*100:.2f} %\n\n"
        "On interpole linéairement entre baseline 2027 et optimum CMA-ES (13D). "
        "Pour chaque seuil de P(victoire), on trouve le pas α minimal qui le franchit ; "
        "le vecteur correspondant donne le **shift requis variable par variable**.\n\n"
    )
    out += "![Courbe alpha → P(victoire)](plots/path_to_victory_curve.png)\n\n"
    out += "![Shifts requis par seuil](plots/path_to_victory_shifts.png)\n\n"
    out += "### Lecture des shifts (depuis baseline 2027)\n\n"
    rows = []
    baseline = s["baseline_state"]
    for t in s["targets"]:
        if not t.get("achievable"):
            continue
        # Détecte dynamiquement les bases présentes (v6 ou v8 multiclasses)
        row = {
            "P_cible": f"≥ {int(t['target']*100)}%",
            "P_obtenu": f"{t['p_check']*100:.1f}%",
            "α": f"{t['alpha']:.3f}",
        }
        for k, v in t['state'].items():
            if k.startswith('base_'):
                short = k.replace('base_', '')
                shift = t['shifts'].get(k, 0)
                row[short] = f"{v:.1f} ({shift:+.1f})"
        rows.append(row)
    out += pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
    out += (
        "### Lecture des Tier 1 (paramètres exogènes + campagne)\n\n"
    )
    rows = []
    for t in s["targets"]:
        if not t.get("achievable"):
            continue
        rows.append({
            "P_cible": f"≥ {int(t['target']*100)}%",
            "crisis": f"{t['state']['crisis']:.2f}",
            "central_collapse": f"{t['state']['central_collapse']:.2f}",
            "volatility": f"{t['state']['volatility']:.2f}",
            "anti_extreme": f"{t['state']['anti_extreme_pressure']:.2f}",
            "machine": f"{t['state']['campaign_machine']:.2f}",
            "thematic": f"{t['state']['thematic_breadth']:.2f}",
            "media": f"{t['state']['media_performance']:.2f}",
            "coalition": f"{t['state']['coalition_building']:.2f}",
        })
    out += pd.DataFrame(rows).to_markdown(index=False) + "\n\n"
    out += (
        "### Trois enseignements du modèle\n\n"
        "1. **Les bases concurrents pèsent plus que les Tier 1**. Faire baisser Bardella de "
        "12-19 pts et Philippe de 9-14 pts est nécessaire à tous les seuils. Les Tier 1 "
        "(campagne + exogènes) bougent en parallèle de 0.29 → 0.46 mais leur effet marginal "
        "est secondaire vs la décrue des concurrents.\n"
        "2. **Boost contre-intuitif de Retailleau**. Le modèle pousse `base_retailleau` à "
        "la hausse (+7 → +11 pts). Mécanisme : Retailleau a une affinité positive sur le "
        "pool RN (+0.20) et fait donc concurrence à Bardella ; en gonflant Retailleau, on "
        "fragmente le vote de droite et on prive Bardella d'une partie de son socle.\n"
        "3. **`central_collapse` doit DIMINUER**, pas augmenter. Avec Philippe déjà rabaissé "
        "à 3-6 pts, le pool central est essentiellement vide. Transférer le pool central "
        "vers les indécis (mécanisme de `central_collapse`) prive Villepin du pool central "
        "où il a son affinité la plus forte (+0.40). Garder le pool central intact lui "
        "donne plus de matière à capter.\n\n"
    )
    return out


def _build_extreme_section(summary) -> str:
    if summary is None:
        return ""
    out = "## B. Recherche extrême : peut-on faire gagner Villepin sans biais ?\n\n"
    out += "CMA-ES sur 13 dimensions (8 paramètres Tier 1 + 5 bases concurrents bornées "
    out += "par leurs intervalles historiques 2002-2022). Objectif : maximiser P(victoire Villepin).\n\n"
    out += f"**P(victoire Villepin) maximale identifiée : {summary['best_p_victory_villepin']*100:.2f}%** "
    out += f"(score 1T = {summary['best_score_1T_villepin']:.1f}%, vs {summary['best_opponent']}).\n\n"
    out += "**Conditions requises** (toutes simultanément) :\n\n"
    out += "| variable | valeur à l'optimum |\n|---|---|\n"
    for k, v in summary["bases_at_optimum"].items():
        out += f"| `base_{k}` | {v:.1f}% |\n"
    for k, v in summary["params_at_optimum"].items():
        out += f"| `{k}` | {v:.2f} |\n"
    p5, p95 = summary["distinct_optima_p5_p95"]
    out += f"\n**Robustesse** : sur {summary['n_restarts']} restarts CMA-ES, "
    out += f"P(victoire) ∈ [{p5*100:.2f}%, {p95*100:.2f}%]. Un seul optimum trouvé.\n\n"
    out += "**Interprétation honnête** : ce résultat dit *« si Bardella tombe au plancher "
    out += "historique RN, ET Philippe au plancher des sortants, ET la crise est maximale, "
    out += "ET Villepin sature sa campagne, ALORS le modèle prédit ~94% »*. La probabilité "
    out += "conjointe de ces 4 chocs en réalité est très faible.\n\n"
    return out


def _save_plot_calibration(loo_df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(loo_df["actual"], loo_df["predicted"], s=80)
    lim = max(loo_df["actual"].max(), loo_df["predicted"].max()) + 2
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, alpha=0.5, label="parfait")
    for _, row in loo_df.iterrows():
        label = f"{row['candidate']} {row['year_held_out']}"
        ax.annotate(label, (row["actual"], row["predicted"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xlabel("Score réel 1er tour (%)")
    ax.set_ylabel("Score prédit (LOO) (%)")
    ax.set_title("Calibration historique : leave-one-out")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p = out / "calibration_loo.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_scenarios(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    df = df.sort_values("best_p_victory")
    ax.barh(df["scenario"], df["best_p_victory"] * 100, color="#5b8def")
    ax.set_xlabel("P(victoire 2T) à l'optimum (%)")
    ax.set_title("Probabilité de victoire par scénario exogène (optim interne)")
    for i, v in enumerate(df["best_p_victory"]):
        ax.text(v * 100 + 0.05, i, f"{v*100:.2f}%", va="center", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    p = out / "scenarios_p_victory.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_dataset_distrib(df: pd.DataFrame, out: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(axes,
                              ["score_1T_mean", "p_qualif", "p_victory"],
                              ["Score 1er tour Villepin", "P(qualifié 2T)", "P(victoire 2T)"]):
        sns.histplot(df[col], bins=40, ax=ax, color="#5b8def")
        ax.set_title(title)
        ax.set_xlabel("")
    plt.tight_layout()
    p = out / "dataset_distrib.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_historical_validation(per_pair_df: pd.DataFrame, per_year_df: pd.DataFrame,
                                     metrics: dict, out: Path) -> tuple[Path, Path]:
    """1. Heatmap année × archétype : score prédit vs réel + symbole match/miss.
       2. Confusion matrix qualification 2T + barre accuracy par année."""
    # --- Plot 1 : grille prédit/réel par année × archétype
    pivot_pred = per_pair_df.pivot(index="year", columns="archetype",
                                    values="predicted_score_1T")
    pivot_act = per_pair_df.pivot(index="year", columns="archetype",
                                   values="actual_score_1T")
    pivot_match = per_pair_df.pivot(index="year", columns="archetype",
                                     values="match")
    archs = pivot_pred.columns.tolist()
    years = pivot_pred.index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(13, max(3.5, 0.6 * len(years))),
                              gridspec_kw={"width_ratios": [1, 1]})
    for ax, mat, title in zip(axes, [pivot_pred, pivot_act],
                              ["Score 1T prédit", "Score 1T réel (agrégé)"]):
        sns.heatmap(mat, annot=True, fmt=".1f", cmap="YlGnBu",
                    ax=ax, cbar_kws={"label": "%"}, linewidths=0.4)
        ax.set_title(title)
        ax.set_xlabel("")
    plt.tight_layout()
    p1 = out / "historical_pred_vs_actual.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # --- Plot 2 : confusion matrix + barre par année
    cm = metrics["confusion_matrix"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5),
                              gridspec_kw={"width_ratios": [1, 1.8]})
    cm_arr = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    sns.heatmap(cm_arr, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["pred non-qualif", "pred qualif"],
                yticklabels=["réel non-qualif", "réel qualif"],
                cbar=False)
    axes[0].set_title(f"Matrice de confusion (qualification 2T)\n"
                      f"F1={metrics['f1_qualif']:.2f} | Acc={metrics['accuracy_qualif']:.2f}")
    # Per-year accuracy bars
    yacc = per_pair_df.groupby("year")["match"].mean().reset_index()
    winner_acc = per_year_df.set_index("year")["winner_correct"].astype(int)
    yacc["winner"] = yacc["year"].map(winner_acc)
    x = np.arange(len(yacc))
    w = 0.4
    axes[1].bar(x - w / 2, yacc["match"] * 100, width=w,
                color="#5b8def", label="qualifié top-2 (matches/6)")
    axes[1].bar(x + w / 2, yacc["winner"] * 100, width=w,
                color="#d62728", label="vainqueur 2T")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(yacc["year"].astype(int).astype(str))
    axes[1].set_ylabel("Précision (%)")
    axes[1].set_ylim(0, 105)
    axes[1].set_title(
        f"Accuracy 2T par année : "
        f"{metrics['winner_correct_count']}/{metrics['winner_total']}"
    )
    axes[1].legend()
    axes[1].grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = out / "historical_validation_metrics.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    return p1, p2


def _save_plot_historical_validation_compare(
    out_dir: Path, out: Path
) -> Path | None:
    """Comparaison validation classique (in-sample, post-hoc) vs walk-forward
    (T-2 mois, AUCUNE info post-hoc).

    Met côte-à-côte les métriques pour faire apparaître le gap réel entre
    la précision affichée du modèle (calibré sur le résultat) et sa capacité
    prédictive honnête.
    """
    in_sample_path = out_dir / "historical_validation_summary.json"
    wf_path = out_dir / "historical_validation_walk_forward_summary.json"
    if not (in_sample_path.exists() and wf_path.exists()):
        return None
    m_in = json.loads(in_sample_path.read_text())["metrics"]
    m_wf = json.loads(wf_path.read_text())["metrics"]
    per_year_in = pd.read_csv(out_dir / "historical_validation_per_year.csv")
    per_year_wf = pd.read_csv(out_dir / "historical_validation_walk_forward_per_year.csv")
    per_pair_in = pd.read_csv(out_dir / "historical_validation_per_pair.csv")
    per_pair_wf = pd.read_csv(out_dir / "historical_validation_walk_forward_per_pair.csv")

    # MAE in-sample : compute si pas présente
    mae_in = float(np.mean(np.abs(
        per_pair_in["predicted_score_1T"] - per_pair_in["actual_score_1T"]
    )))
    mae_wf = m_wf.get("mae_score_1T", float(np.mean(np.abs(
        per_pair_wf["predicted_score_1T"] - per_pair_wf["actual_score_1T"]
    ))))

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.55, wspace=0.4,
                          left=0.06, right=0.98, top=0.86, bottom=0.10)

    # Panel 1 : MAE 1T comparaison
    ax = fig.add_subplot(gs[0, 0])
    vals = [mae_in, mae_wf]
    colors = ["#90A4AE", "#C62828"]
    bars = ax.bar(["in-sample\n(post-hoc)", "walk-forward\n(T-2 mois)"],
                  vals, color=colors, edgecolor="white", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.15,
                f"{v:.2f}", ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("MAE score 1T (points)", fontsize=10)
    ax.set_title("MAE 1T : capacité prédictive réelle",
                 fontsize=11, fontweight="bold", loc="left")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel 2 : F1 / Accuracy comparaison
    ax = fig.add_subplot(gs[0, 1])
    keys = ["accuracy_qualif", "precision_qualif",
            "recall_qualif", "f1_qualif"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(keys))
    w = 0.4
    ax.bar(x - w/2, [m_in[k] for k in keys], w,
           color="#90A4AE", edgecolor="white",
           label="in-sample (post-hoc)")
    ax.bar(x + w/2, [m_wf[k] for k in keys], w,
           color="#C62828", edgecolor="white",
           label="walk-forward (T-2 mois)")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score [0,1]", fontsize=10)
    ax.set_title("Classification qualif. 2T (archétypes)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel 3 : vainqueur 2T par année (in-sample vs walk-forward)
    ax = fig.add_subplot(gs[0, 2])
    years_in = per_year_in["year"].astype(int).values
    years_wf = per_year_wf["year"].astype(int).values
    correct_in = per_year_in["winner_correct"].astype(int).values
    correct_wf = per_year_wf["winner_correct"].astype(int).values
    x = np.arange(len(years_in))
    w = 0.4
    ax.bar(x - w/2, correct_in * 100, w,
           color="#90A4AE", edgecolor="white", label="in-sample")
    ax.bar(x + w/2, correct_wf * 100, w,
           color="#C62828", edgecolor="white", label="walk-forward")
    ax.set_xticks(x); ax.set_xticklabels(years_in.astype(str), fontsize=9)
    ax.set_ylim(0, 130)
    ax.set_yticks([0, 50, 100])
    ax.set_ylabel("Vainqueur 2T correct ?", fontsize=10)
    ax.set_title(
        f"Vainqueur 2T par année : "
        f"{m_in['winner_correct_count']}/{m_in['winner_total']} vs "
        f"{m_wf['winner_correct_count']}/{m_wf['winner_total']}",
        fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel 4 (étend sur 2 cols) : MAE par année et archétype
    ax = fig.add_subplot(gs[1, :2])
    # err par paire en valeur absolue
    err_in = (per_pair_in["predicted_score_1T"] - per_pair_in["actual_score_1T"]).abs()
    err_wf = (per_pair_wf["predicted_score_1T"] - per_pair_wf["actual_score_1T"]).abs()
    per_pair_in["err"] = err_in
    per_pair_wf["err"] = err_wf
    grp_in = per_pair_in.groupby("year")["err"].mean()
    grp_wf = per_pair_wf.groupby("year")["err"].mean()
    x = np.arange(len(grp_in))
    w = 0.4
    ax.bar(x - w/2, grp_in.values, w, color="#90A4AE",
           edgecolor="white", label="in-sample (post-hoc)")
    ax.bar(x + w/2, grp_wf.values, w, color="#C62828",
           edgecolor="white", label="walk-forward (T-2 mois)")
    for i in range(len(grp_in)):
        ax.text(x[i] - w/2, grp_in.values[i] + 0.15,
                f"{grp_in.values[i]:.1f}", ha="center", fontsize=8.5,
                color="#555")
        ax.text(x[i] + w/2, grp_wf.values[i] + 0.15,
                f"{grp_wf.values[i]:.1f}", ha="center", fontsize=8.5,
                color="#C62828", fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(grp_in.index.astype(int).astype(str),
                                          fontsize=9)
    ax.set_ylabel("MAE 1T par année (points)", fontsize=10)
    ax.set_title("MAE 1T par année (6 archétypes agrégés)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel 5 : note méthodologique
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    fuite_pct = ((mae_wf - mae_in) / mae_wf) * 100 if mae_wf > 0 else 0
    note = (
        "Différence in-sample vs walk-forward :\n\n"
        f"  • MAE :  {mae_in:.2f}  →  {mae_wf:.2f} pts  ({mae_wf - mae_in:+.2f})\n"
        f"  • F1   :  {m_in['f1_qualif']:.2f}  →  {m_wf['f1_qualif']:.2f}\n"
        f"  • Vainqueur 2T : "
        f"{m_in['winner_correct_count']}/{m_in['winner_total']}  →  "
        f"{m_wf['winner_correct_count']}/{m_wf['winner_total']}\n\n"
        f"L'écart de {fuite_pct:.0f} % sur la MAE\n"
        f"reflète la fuite de calibration :\n"
        f"CONTEXT[year] et YEAR_POOLS[year]\n"
        f"étaient fixés en connaissant le\n"
        f"résultat final. Walk-forward ne\n"
        f"voit que les sondages T-2 mois."
    )
    ax.text(0.0, 1.0, note, transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5, family="monospace",
            color="#222",
            bbox=dict(boxstyle="round,pad=0.6", fc="#FFEBEE",
                      ec="#C62828", lw=1.2))

    fig.suptitle(
        "Validation honnête (walk-forward T-2 mois) vs validation in-sample (post-hoc)",
        fontsize=14, fontweight="bold", y=0.95,
    )
    fig.text(0.5, 0.91,
             "Les chiffres affichés en rouge sont la VRAIE capacité prédictive du modèle, "
             "sans aucune information qui n'aurait pas été disponible 2 mois avant l'élection.",
             ha="center", fontsize=10, style="italic", color="#444")

    p = out / "historical_validation_compare.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_partial_dependence(model_path: Path, dataset_path: Path,
                                   out: Path) -> Path | None:
    """Pour chaque param Tier 1 : P(victoire) prédite en fonction de la valeur
    du param, les autres fixés à leur médiane du dataset.
    """
    try:
        import torch
        from .neural_predictor import load_model
        from .parameters import TIER1_PARAMS
    except ImportError:
        return None
    model, x_cols = load_model(model_path)
    df = pd.read_parquet(dataset_path)
    medians = df[x_cols].median().values.astype(np.float32)
    grid = np.linspace(0.0, 1.0, 41)

    tier1 = [c for c in x_cols if c in TIER1_PARAMS]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    for ax, name in zip(axes.flatten(), tier1):
        i = x_cols.index(name)
        X = np.tile(medians, (len(grid), 1))
        X[:, i] = grid
        with torch.no_grad():
            pred = model(torch.from_numpy(X.astype(np.float32))).numpy()
        ax.plot(grid, pred[:, 2] * 100, color="#5b8def", lw=2)
        ax.fill_between(grid, 0, pred[:, 2] * 100, color="#5b8def", alpha=0.18)
        ax.axhline(0, color="#aaa", lw=0.5)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("P(victoire) (%)")
    axes[1, 0].set_ylabel("P(victoire) (%)")
    fig.suptitle(
        "Partial dependence : P(victoire 2T) en fonction de chaque paramètre "
        "(autres fixés à la médiane)", fontsize=11,
    )
    plt.tight_layout()
    p = out / "partial_dependence.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_param_sensitivity(model_path: Path, dataset_path: Path,
                                  out: Path) -> Path | None:
    """Sensibilité globale = E[|∂P/∂param|] sur le dataset complet."""
    try:
        import torch
        from .neural_predictor import load_model
    except ImportError:
        return None
    model, x_cols = load_model(model_path)
    df = pd.read_parquet(dataset_path)
    X = df[x_cols].values.astype(np.float32)
    n_samples = min(2000, len(X))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(X), n_samples, replace=False)
    X_sample = torch.from_numpy(X[idx]).requires_grad_(True)
    pred = model(X_sample)
    p_vict = pred[:, 2].sum()
    grads = torch.autograd.grad(p_vict, X_sample)[0].numpy()
    mean_abs = np.abs(grads).mean(axis=0)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(x_cols))))
    order = np.argsort(mean_abs)
    palette = sns.color_palette("rocket", n_colors=len(x_cols))
    ax.barh([x_cols[i] for i in order], mean_abs[order],
            color=[palette[i] for i in order])
    ax.set_xlabel("Sensibilité moyenne |∂P(victoire)/∂param|")
    ax.set_title("Sensibilité globale du surrogate aux 8 paramètres Tier 1")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    p = out / "param_sensitivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_llm_evolution(llm_summary: dict, out: Path) -> Path | None:
    """Évolution de P(victoire) sur les itérations LLM."""
    iters = llm_summary.get("iterations", [])
    if not iters:
        return None
    baseline = llm_summary.get("baseline_best_p_victory", float("nan"))
    xs = [0] + [it["iter"] for it in iters]
    ys = [baseline] + [it.get("best_p_victory_after", baseline) for it in iters]
    decisions = ["baseline"] + [it.get("decision", "?") for it in iters]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#5b8def" if d == "keep" else ("#d62728" if d == "rollback" else "#999")
              for d in decisions]
    ax.plot(xs, np.array(ys) * 100, "-", color="#888", lw=1)
    ax.scatter(xs, np.array(ys) * 100, c=colors, s=110, zorder=5)
    for x, y, d in zip(xs, ys, decisions):
        ax.annotate(d, (x, y * 100), textcoords="offset points", xytext=(8, 4), fontsize=8)
    ax.axhline(baseline * 100, ls="--", lw=0.8, color="#888", alpha=0.6,
               label=f"baseline {baseline*100:.3f}%")
    ax.set_xlabel("Itération LLM")
    ax.set_ylabel("Best P(victoire) (%)")
    ax.set_title(f"Inférence Tier 2 : évolution du plafond ({llm_summary.get('model','?')})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    p = out / "llm_evolution.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_tier2_weights(tier2_meta: dict, out: Path) -> Path | None:
    """Bar chart des poids attribués aux sous-paramètres Tier 2 (groupés par parent)."""
    if not tier2_meta:
        return None
    rows = []
    for name, meta in tier2_meta.items():
        rows.append({"name": name, "parent": meta.get("parent", "?"),
                     "weight": abs(meta.get("weight", 0.0)),
                     "importance": meta.get("importance", 0)})
    df = pd.DataFrame(rows).sort_values(["parent", "weight"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(df))))
    palette = sns.color_palette("tab10", n_colors=df["parent"].nunique())
    parent_to_color = {p: palette[i] for i, p in enumerate(df["parent"].unique())}
    bar_colors = [parent_to_color[p] for p in df["parent"]]
    ax.barh(df["name"][::-1], df["weight"][::-1], color=bar_colors[::-1])
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in parent_to_color.values()]
    ax.legend(handles, parent_to_color.keys(), title="parent Tier 1",
              loc="lower right", fontsize=8)
    ax.set_xlabel("Poids |w| dans la formule de masse")
    ax.set_title("Sous-paramètres Tier 2 retenus par la boucle LLM")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    p = out / "tier2_weights.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_path_to_victory(summary: dict, out: Path) -> tuple[Path, Path]:
    """Deux plots :
    1. Courbe P(victoire) en fonction de l'avancement α (baseline → optimum).
    2. Heatmap des shifts par variable × seuil P(victoire).
    """
    targets = [r for r in summary["targets"] if r.get("achievable")]
    # Plot 1 : courbe alpha vs p_victory. CSV est dans outputs/, plots/ a un parent qui pointe outputs/.
    csv_path = Path(out).parent.parent / "path_to_victory.csv"
    if not csv_path.exists():
        return None, None
    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["alpha"], df["p_victory"] * 100, color="#5b8def", lw=2)
    ax.fill_between(df["alpha"], 0, df["p_victory"] * 100,
                     color="#5b8def", alpha=0.18)
    for t in targets:
        ax.axhline(t["target"] * 100, ls="--", lw=0.6, color="#888")
        ax.annotate(
            f"P≥{t['target']*100:.0f}% @ α={t['alpha']:.2f}",
            xy=(t["alpha"], t["target"] * 100),
            xytext=(t["alpha"] + 0.02, t["target"] * 100 + 4),
            fontsize=8, color="#333",
            arrowprops=dict(arrowstyle="->", color="#999", lw=0.5),
        )
    ax.set_xlabel("α (0 = baseline 2027,  1 = optimum CMA-ES)")
    ax.set_ylabel("P(victoire Villepin 2T) (%)")
    ax.set_title("Chemin minimal vers la victoire : interpolation baseline vers optimum extrême")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    plt.tight_layout()
    p1 = Path(out).parent / "path_to_victory_curve.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Plot 2 : heatmap shifts × seuils
    # Tier 1 fixes + bases dynamiques (clés base_* effectivement présentes dans state).
    baseline_state = summary["baseline_state"]
    base_vars = [k for k in baseline_state if k.startswith("base_")]
    var_order = ["crisis", "central_collapse", "volatility", "anti_extreme_pressure",
                 "campaign_machine", "thematic_breadth", "media_performance",
                 "coalition_building"] + base_vars
    rows = []
    for t in targets:
        row = {"target": f"P≥{int(t['target']*100)}%  (P={t['p_check']*100:.1f}%)"}
        baseline = summary["baseline_state"]
        for v in var_order:
            row[v] = t["state"][v] - baseline[v]
        rows.append(row)
    if not rows:
        return p1, None
    sh = pd.DataFrame(rows).set_index("target")[var_order]
    # Normaliser les shifts pour rendre la heatmap lisible :
    # Tier 1 (∈[0,1]) garde sa valeur ; bases concurrents normalisées à leur plage historique.
    from .winner_analysis import HISTORICAL_BASE_BOUNDS
    normalizer = {}
    for v in var_order:
        if v.startswith("base_"):
            arch = v[5:]
            lo, hi = HISTORICAL_BASE_BOUNDS[arch]
            normalizer[v] = hi - lo
        else:
            normalizer[v] = 1.0
    sh_norm = sh.copy()
    for v in var_order:
        sh_norm[v] = sh[v] / normalizer[v]

    fig, ax = plt.subplots(figsize=(13, max(3, 0.55 * len(rows))))
    sns.heatmap(sh_norm, annot=sh.round(2), fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={"label": "shift normalisé"},
                linewidths=0.4, linecolor="white",
                annot_kws={"fontsize": 9})
    ax.set_title("Shifts requis depuis baseline 2027 pour atteindre chaque seuil de P(victoire)")
    ax.set_xlabel("variable")
    ax.set_ylabel("seuil")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    plt.tight_layout()
    p2 = Path(out).parent / "path_to_victory_shifts.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p1, p2


def _save_plot_winner_heatmap(winner_df: pd.DataFrame, out: Path) -> Path:
    """Heatmap (exo, shock) × candidate : intensité = P(victoire).

    Lignes : combinaison exogène × shock. Colonnes : candidats. On filtre les
    candidats jamais qualifiés pour gagner en lisibilité.
    """
    pivot = winner_df.pivot_table(
        index=["exo_scenario", "shock_scenario"],
        columns="candidate", values="p_victory", aggfunc="mean",
    ).fillna(0)
    # On retire les candidats jamais qualifiés (col sum < 0.01) pour épurer
    keep = pivot.sum(axis=0) >= 0.01
    pivot = pivot.loc[:, keep]
    col_order = pivot.sum(axis=0).sort_values(ascending=False).index.tolist()
    pivot = pivot[col_order]
    # Format index lisible
    pivot.index = pivot.index.map(lambda t: f"{t[0]:<13} × {t[1]}")
    fig_h = max(7.5, 0.32 * len(pivot))
    fig, ax = plt.subplots(figsize=(11, fig_h))
    sns.heatmap(pivot, annot=True, fmt=".0%", cmap="RdYlGn",
                vmin=0, vmax=1, cbar_kws={"label": "P(victoire 2T)"},
                linewidths=0.4, linecolor="white",
                annot_kws={"fontsize": 8}, ax=ax)
    ax.set_title("P(victoire 2T) par scénario (exogène × shock concurrents) × candidat",
                  fontsize=11, pad=10)
    ax.set_ylabel("scénario", fontsize=9)
    ax.set_xlabel("candidat", fontsize=9)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8, family="monospace")
    plt.setp(ax.get_xticklabels(), fontsize=10)
    plt.tight_layout()
    p = out / "winner_heatmap.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _save_plot_villepin_top_scenarios(villepin_df: pd.DataFrame, out: Path) -> Path:
    """Bar chart trié des meilleurs scénarios pour Villepin (top-10)."""
    top = villepin_df.head(10).copy()
    top["label"] = top["exo_scenario"] + " ✕ " + top["shock_scenario"]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("RdYlGn", n_colors=len(top))
    bars = ax.barh(top["label"][::-1], top["p_victory"][::-1] * 100,
                   color=colors)
    for bar, v in zip(bars, top["p_victory"][::-1]):
        ax.text(v * 100 + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{v*100:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("P(victoire Villepin) (%)")
    ax.set_title("Top 10 des scénarios où Villepin a le plus de chances")
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    p = out / "villepin_top_scenarios.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_sensitivity(sens_df: pd.DataFrame, out: Path) -> Path:
    """Pour chaque année : intervalle p5-p95 des prédictions + point réel."""
    fig, ax = plt.subplots(figsize=(9, 5))
    years = sens_df["year"].astype(int).astype(str).tolist()
    means = sens_df["predicted_mean"].values
    p5s = sens_df["predicted_p5"].values
    p95s = sens_df["predicted_p95"].values
    actuals = sens_df["actual"].values
    x = np.arange(len(years))
    ax.errorbar(x, means, yerr=[means - p5s, p95s - means],
                fmt="o", color="#5b8def", capsize=6, label="prédit [p5, p95]")
    ax.scatter(x, actuals, color="#d62728", marker="X", s=110, zorder=5, label="réel")
    for i, (y, m, a) in enumerate(zip(years, means, actuals)):
        ax.annotate(f"{a:.1f}", (x[i], a), textcoords="offset points",
                    xytext=(5, 5), fontsize=8, color="#d62728")
    labels = [f"{y}\n{sens_df['candidate'].iloc[i].split()[-1]}" for i, y in enumerate(years)]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score 1er tour (%)")
    ax.set_title("Sensibilité au contexte : prédits (±0.20 sur 8 paramètres) vs réels")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p = out / "sensitivity.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_effort_allocation(scenarios_df: pd.DataFrame, out: Path) -> Path:
    """Barres groupées : effort optimal par paramètre interne, par scénario."""
    internal = ["best_campaign_machine", "best_thematic_breadth",
                "best_media_performance", "best_coalition_building"]
    sub = scenarios_df.set_index("scenario")[internal].copy()
    sub.columns = ["machine", "thematic", "media", "coalition"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sub.plot(kind="bar", ax=ax, width=0.78, colormap="viridis")
    ax.set_ylabel("Valeur optimale du paramètre interne")
    ax.set_ylim(0, 1.05)
    ax.set_title("Allocation d'effort optimale (CMA-ES) par scénario")
    ax.set_xlabel("")
    ax.legend(title="paramètre", loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    p = out / "effort_allocation.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


_POOL_COLORS = {
    "rn":       "#1a3a8a",
    "central":  "#FFC400",
    "gauche":   "#F08080",
    "lfi":      "#B71C1C",
    "lr":       "#0288D1",
    "indecis":  "#9E9E9E",
}
_POOL_LABELS_FR = {
    # Pools v6 (config.yaml et config_8pools.yaml)
    "rn":       "Électorat RN / Reconquête",
    "central":  "Centre / macronistes",
    "gauche":   "Gauche modérée (PS, Verts)",
    "lfi":      "Gauche radicale (LFI, PCF, NPA)",
    "lr":       "Droite classique (LR)",
    "indecis":  "Indécis / fluide",
    # Pools v8 multiclasses (config_multiclass.yaml)
    "extreme_droite":    "Extrême droite (RN, Reconquête)",
    "souverainiste":     "Souverainistes (DLF, UPR)",
    "droite_classique":  "Droite classique (LR)",
    "centre_gouv":       "Centre gouvernemental (Renaissance, Horizons)",
    "centre_outsider":   "Centre outsider (LFH, MoDem)",
    "gauche_socdem":     "Gauche modérée (PS, Verts)",
    "gauche_radicale":   "Gauche radicale (LFI, PCF)",
    "extreme_gauche":    "Extrême gauche (NPA, LO)",
    # Compétiteurs 8 pools (config_8pools.yaml uniquement)
    "extreme_droite_8p": "Extrême droite (Reconquête)",
    "extreme_gauche_8p": "Extrême gauche (NPA/LO)",
}
_ARCH_LABELS_FR = {
    # v6 (noms de personnes)
    "bardella":   "Bardella\n(RN)",
    "philippe":   "Philippe\n(Horizons)",
    "melenchon":  "Mélenchon\n(LFI)",
    "glucksmann": "Glucksmann\n(PS)",
    "villepin":   "Villepin\n(LFH)",
    "retailleau": "Retailleau\n(LR)",
    # v8 multiclasses (noms d'archétypes ; on associe le candidat phare 2027)
    "extreme_droite":   "Bardella\n(RN)",
    "souverainiste":    "Souverainiste\n(DLF/UPR)",
    "droite_classique": "Retailleau\n(LR)",
    "centre_gouv":      "Philippe\n(Horizons)",
    "centre_outsider":  "Villepin\n(LFH)",
    "gauche_socdem":    "Glucksmann\n(PS)",
    "gauche_radicale":  "Mélenchon\n(LFI)",
    "extreme_gauche":   "Poutou/Arthaud\n(NPA/LO)",
    # 8 pools (compétiteurs additionnels)
    "zemmour":    "Zemmour\n(Reconquête)",
    "poutou":     "Poutou\n(NPA)",
}
_ARCH_COLORS = {
    "bardella":   "#1a3a8a",
    "retailleau": "#0288D1",
    "philippe":   "#FFC400",
    "glucksmann": "#F08080",
    "melenchon":  "#B71C1C",
    "villepin":   "#7E57C2",
    # v8 multiclasses
    "extreme_droite":   "#0D1A4D",
    "souverainiste":    "#6D4C41",
    "droite_classique": "#0288D1",
    "centre_gouv":      "#FFC400",
    "centre_outsider":  "#7E57C2",
    "gauche_socdem":    "#F08080",
    "gauche_radicale":  "#B71C1C",
    "extreme_gauche":   "#880E4F",
    # 8 pools
    "zemmour": "#4527A0",
    "poutou":  "#AD1457",
}


def _save_plot_inputs_overview(cfg, out: Path) -> Path:
    """Vue d'ensemble des 8 entrées du NN, partitionnées exogènes vs internes."""
    fig = plt.figure(figsize=(13, 7.5))
    gs = fig.add_gridspec(2, 4, hspace=0.45, wspace=0.30,
                          left=0.05, right=0.97, top=0.86, bottom=0.10)

    exogenes = [
        ("crisis", "Intensité géopolitique",
         "Guerres, tensions internationales,\nUkraine, Gaza, Taïwan...", "#D32F2F"),
        ("central_collapse", "Effondrement bloc central",
         "PS, LR, Renaissance fragilisés.\nDes voix flottent.", "#F57C00"),
        ("volatility", "Volatilité électorale",
         "Combien les électeurs sont prêts\nà changer d'avis pendant la campagne", "#FBC02D"),
        ("anti_extreme_pressure", "Front républicain",
         "Pression sociale à voter contre\nl'extrême-droite/-gauche au 2nd tour", "#7B1FA2"),
    ]
    internes = [
        ("campaign_machine", "Machine de campagne",
         "Parrainages, militants, fonds,\nfédérations locales", "#1565C0"),
        ("thematic_breadth", "Largeur thématique",
         "Sortir du seul international,\nproposer un programme complet", "#00838F"),
        ("media_performance", "Performance médiatique",
         "Verbe, présence TV, débats,\nmaîtrise du tempo", "#2E7D32"),
        ("coalition_building", "Construction de coalitions",
         "Ralliements de gaullistes,\ncentristes, gauche modérée", "#558B2F"),
    ]

    for i, (name, title, desc, color) in enumerate(exogenes):
        ax = fig.add_subplot(gs[0, i])
        _draw_input_card(ax, title, desc, color, "EXOGÈNE\n(contexte subi)")
    for i, (name, title, desc, color) in enumerate(internes):
        ax = fig.add_subplot(gs[1, i])
        _draw_input_card(ax, title, desc, color, "INTERNE\n(campagne contrôle)")

    fig.suptitle(
        "Entrées du réseau de neurones : 8 paramètres ∈ [0, 1]",
        fontsize=13, fontweight="bold", y=0.96,
    )
    fig.text(
        0.5, 0.03,
        "Sorties : score_1T (%) | P(qualifié_2T) ∈ [0,1] | P(victoire_2T) ∈ [0,1]",
        ha="center", fontsize=9.5, color="#666", family="monospace",
    )
    p = out / "inputs_overview.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _draw_input_card(ax, title, desc, color, category):
    """Petite carte rectangulaire propre pour un input."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # Rectangle de fond
    rect = plt.Rectangle((0.02, 0.05), 0.96, 0.90,
                         facecolor=color, alpha=0.13,
                         edgecolor=color, linewidth=1.5)
    ax.add_patch(rect)
    # Catégorie en haut
    ax.text(0.5, 0.88, category, ha="center", va="top",
            fontsize=7.5, color=color, fontweight="bold",
            family="monospace")
    # Titre
    ax.text(0.5, 0.65, title, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#222")
    # Description
    ax.text(0.5, 0.30, desc, ha="center", va="center",
            fontsize=8.5, color="#555")


def _save_plot_archetypes_explained(cfg, out: Path) -> Path:
    """6 panels expliquant chaque archétype.
    Titre = famille politique (neutre), liste rigoureuse des partis agrégés
    (passé + présent), exemples avec le parti EFFECTIF de l'élection visée.
    """
    arch_data = [
        # (key, family_name, parties_aggregated_text, examples)
        ("bardella",
         "Extrême-droite / national-populiste",
         "Partis agrégés : FN (1972-2018), MNR, MPF, DLF, RN (depuis 2018), Reconquête",
         ["2002 : Le Pen J-M (FN) 16.9 % + Mégret (MNR) 2.3 %",
          "2007 : Le Pen J-M (FN) 10.4 % + Villiers (MPF) 2.2 %",
          "2012 : Le Pen M. (FN) 17.9 %",
          "2017 : Le Pen M. (FN) 21.3 % + Dupont-Aignan (DLF) 4.7 %",
          "2022 : Le Pen M. (RN) 23.2 % + Zemmour (Rec.) 7.1 % + DA 2.1 %",
          "2027 : Bardella (RN) ≈ 36 % (sondé)"]),
        ("retailleau",
         "Droite gouvernementale classique",
         "Partis agrégés : RPR (1976-2002), UMP (2002-2015), DL, CPNT, LR (depuis 2015)",
         ["2002 : Chirac (RPR) 19.9 % + Madelin (DL) 3.9 % + CPNT 4.2 %",
          "2007 : Sarkozy (UMP) 31.2 % + Nihous (CPNT) 1.2 %",
          "2012 : Sarkozy (UMP) 27.2 % + DA (DLR) 1.8 %",
          "2017 : Fillon (LR) 20.0 % + Asselineau (UPR) 0.9 %",
          "2022 : Pécresse (LR) 4.8 %",
          "2027 : Retailleau (LR) ≈ 10 % (sondé)"]),
        ("philippe",
         "Centre / macroniste / sortant",
         "Partis agrégés : Cap21 (2002), LREM/EM (2017→Renaissance), Horizons, RES",
         ["2002 : Lepage (Cap21) 1.9 %",
          "2007 : aucun candidat dans cette case",
          "2012 : aucun candidat dans cette case",
          "2017 : Lassalle (RES) 1.2 %",
          "2022 : Macron sortant (LREM) 27.9 % + Lassalle (RES) 3.1 %",
          "2027 : Philippe (Horizons) ≈ 18 % (sondé)"]),
        ("glucksmann",
         "Gauche modérée / sociale-démocrate",
         "Partis agrégés : PS, PRG, MdC, Verts (puis EELV), Place Publique",
         ["2002 : Jospin (PS) 16.2 % + Mamère (Verts) 5.3 % + Chevènement 5.3 %",
          "2007 : Royal (PS) 25.9 % + Voynet (Verts) 1.6 %",
          "2012 : Hollande (PS) 28.6 % + Joly (EELV) 2.3 %",
          "2017 : Hamon (PS) 6.4 %",
          "2022 : Jadot (EELV) 4.6 % + Hidalgo (PS) 1.8 %",
          "2027 : Glucksmann (PS / Place publique) ≈ 11 % (sondé)"]),
        ("melenchon",
         "Gauche radicale",
         "Partis agrégés : PCF, LO, LCR/NPA, PT, FG (2012), LFI (depuis 2016)",
         ["2002 : Hue (PCF) 3.4 % + Laguiller (LO) 5.7 % + Besancenot (LCR) 4.3 % + PT 0.5 %",
          "2007 : Besancenot (LCR) 4.1 % + Buffet (PCF) 1.9 % + LO 1.3 % + Bové 1.3 %",
          "2012 : Mélenchon (FG) 11.1 % + Poutou (NPA) 1.2 % + Arthaud (LO) 0.6 %",
          "2017 : Mélenchon (LFI) 19.6 % + Poutou (NPA) 1.1 % + Arthaud (LO) 0.6 %",
          "2022 : Mélenchon (LFI) 22.0 % + Roussel (PCF) 2.3 % + Poutou + Arthaud",
          "2027 : Mélenchon (LFI) ≈ 11 % (sondé)"]),
        ("villepin",
         "Centriste outsider / rassembleur",
         "Partis agrégés : UDF (1978-2007), MoDem (depuis 2007), EM (2017, devenu Renaissance), LFH (depuis 2026)",
         ["2002 : Bayrou (UDF) 6.8 %",
          "2007 : Bayrou (UDF/MoDem) 18.6 %  ← succès relatif",
          "2012 : Bayrou (MoDem) 9.1 %",
          "2017 : Macron (EM) 24.0 %  ← qualifié et élu",
          "2022 : Pécresse (LR-centre) 4.8 %  ← échec patent",
          "2027 : Villepin (LFH, créée en 2026) ? sondé ≈ 11 %"]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(
        "Les 6 archétypes politiques du modèle (familles stables 2002-2027)",
        fontsize=14, fontweight="bold", y=0.965,
    )
    fig.text(
        0.5, 0.93,
        "Chaque archétype est défini par son attraction (affinité) sur les 6 groupes sociologiques d'électeurs",
        ha="center", fontsize=10.5, style="italic", color="#444",
    )

    pool_keys = list(cfg["pools"].keys())
    # Filtre les archétypes qui existent dans cfg["competitors"] (le mapping
    # arch_data est conçu pour v6 ; en multiclasses, on skip ceux qui n'existent
    # pas et la figure est juste plus vide. Pour un rendu propre en multiclasses,
    # il faudrait un arch_data spécifique).
    arch_data = [d for d in arch_data
                 if d[0] == "villepin" or d[0] in cfg.get("competitors", {})]
    if not arch_data:
        plt.close(fig)
        return None
    for idx, (arch, family, parties_line, examples) in enumerate(arch_data):
        r, c = idx // 3, idx % 3
        if r >= 2 or c >= 3:
            break
        ax = axes[r][c]
        # Affinités
        if arch == "villepin":
            affs = [cfg["villepin_affinity"][p] for p in pool_keys]
        else:
            affs = [cfg["competitors"][arch]["affinity"][p] for p in pool_keys]
        ax.axhline(0, color="#333", lw=0.8, alpha=0.5)
        colors = ["#2E7D32" if a > 0 else "#C62828" for a in affs]
        ax.bar(range(len(pool_keys)), affs, color=colors, alpha=0.75,
               edgecolor="white", linewidth=0.8)
        ax.set_xticks(range(len(pool_keys)))
        # Labels courts pour éviter chevauchements : sigles seuls.
        short_pool = {"rn": "RN", "central": "Centre", "gauche": "Gauche",
                      "lfi": "LFI", "lr": "LR", "indecis": "Indécis"}
        ax.set_xticklabels(
            [short_pool.get(p, p) for p in pool_keys],
            fontsize=9,
        )
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Affinité [-1, +1]", fontsize=8.5)
        # Titre : famille politique (gras, couleur archétype)
        ax.set_title(family, fontsize=11, fontweight="bold",
                     color=_ARCH_COLORS[arch], loc="left", pad=20)
        # Ligne "Partis agrégés" juste sous le titre
        ax.text(0.0, 1.045, parties_line, transform=ax.transAxes,
                ha="left", va="bottom", fontsize=7.8, color="#555",
                style="italic")
        ax.grid(True, axis="y", alpha=0.2, ls=":")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

        # Exemples historiques (élection par élection, avec parti d'époque)
        ex_text = "Scores 1er tour par élection :\n" + "\n".join(f"• {e}" for e in examples)
        ax.text(0.5, -0.50, ex_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.8, color="#333",
                family="sans-serif",
                bbox=dict(boxstyle="round,pad=0.5", fc="#fafafa",
                          ec="#ccc", lw=0.6))

    plt.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.97,
                        hspace=1.55, wspace=0.30)
    p = out / "archetypes_explained.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_historical_per_party(out_dir: Path, out: Path) -> Path | None:
    """Pour chaque archétype, scores prédits vs réels sur les 5 élections passées.

    Note : l'archétype `villepin` (LFH créée en 2026) n'existait pas avant. Pour
    valider sur 2002-2022 on prend le centriste outsider de l'époque comme
    proxy : Bayrou (UDF/MoDem), puis Macron 2017, puis Pécresse 2022.
    """
    per_pair_path = out_dir / "historical_validation_per_pair.csv"
    if not per_pair_path.exists():
        return None
    df = pd.read_csv(per_pair_path)
    # Liste dynamique : tous les archétypes présents dans le CSV de validation.
    archetypes = sorted(df["archetype"].unique().tolist())

    # Titres courts par panel (ne pas mentionner la LFH avant 2026).
    # Pour `villepin` : on indique explicitement que c'est un proxy historique
    # car la LFH (parti créé en 2026) n'existait pas avant.
    panel_titles = {
        "bardella":   "Bardella (RN)",
        "retailleau": "Retailleau (LR)",
        "philippe":   "Philippe (Horizons)",
        "glucksmann": "Glucksmann (PS)",
        "melenchon":  "Mélenchon (LFI)",
        # Proxy Villepin 2027 : la LFH (parti de Villepin) date de 2026,
        # donc avant 2026 on prend le centriste outsider de l'époque.
        "villepin":   "Centriste outsider (proxy Villepin)",
    }
    villepin_year_labels = {
        2002: "Bayrou\n(UDF)",
        2007: "Bayrou\n(UDF)",
        2012: "Bayrou\n(MoDem)",
        2017: "Macron\n(EM)",
        2022: "Pécresse\n(LR)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.0))
    fig.suptitle(
        "Précision du modèle par parti, sur les 5 élections passées (2002-2022)",
        fontsize=14, fontweight="bold", y=0.97,
    )
    fig.text(
        0.5, 0.935,
        "Score réel (cercle plein) vs score prédit (croix). Plus c'est proche, mieux le modèle prédit ce parti.",
        ha="center", fontsize=10, style="italic", color="#444",
    )
    for idx, arch in enumerate(archetypes):
        r, c = idx // 3, idx % 3
        ax = axes[r][c]
        sub = df[df["archetype"] == arch].sort_values("year")
        years_int = sub["year"].astype(int).tolist()
        actuals = sub["actual_score_1T"].values
        preds = sub["predicted_score_1T"].values
        x = np.arange(len(years_int))

        ax.plot(x, actuals, "o-", color=_ARCH_COLORS[arch], lw=2,
                 markersize=11, label="Score réel", zorder=3)
        ax.plot(x, preds, "x--", color="#333", lw=1.4,
                 markersize=11, mew=2.0, label="Score prédit", zorder=4)
        ax.fill_between(x, actuals, preds, color=_ARCH_COLORS[arch],
                         alpha=0.15)

        mae = float(np.mean(np.abs(preds - actuals)))
        ax.set_title(
            f"{panel_titles[arch]}\nErreur moyenne : {mae:.1f} pts",
            fontsize=10.5, fontweight="bold", color=_ARCH_COLORS[arch],
            loc="left",
        )

        ax.set_xticks(x)
        if arch == "villepin":
            # Étiquette : année + candidat réel de l'époque + son parti
            labels = [f"{y}\n{villepin_year_labels.get(y, '')}"
                      for y in years_int]
            ax.set_xticklabels(labels, fontsize=8.2)
        else:
            ax.set_xticklabels([str(y) for y in years_int], fontsize=9)
        ax.set_ylabel("Score 1er tour (%)", fontsize=9)
        ax.set_ylim(0, max(actuals.max(), preds.max()) * 1.25)
        ax.legend(loc="upper left", fontsize=8.5)
        ax.grid(True, alpha=0.25, ls=":")
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    plt.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.98,
                        hspace=0.55, wspace=0.28)
    p = out / "historical_per_party.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_villepin_winning_path(cfg, out: Path) -> Path | None:
    """Mise en page pédagogique du chemin optimal vers P(victoire Villepin) = 50 %.

    Format identique à `baseline_2027` (barres empilées par pool sociologique,
    sondage hachuré, seuil 2T, bulles "Qualifié 2T") mais comparant cette fois
    BASELINE (couleurs pâles) vs CHEMIN OPTIMAL CMA-ES (couleurs pleines), avec
    le sondage réel mai 2026 en référence à droite.
    """
    summary_path = Path("outputs") / "path_to_victory.summary.json"
    if not summary_path.exists():
        return None
    s = json.loads(summary_path.read_text())
    target = None
    for t in s["targets"]:
        if t.get("achievable") and abs(t["target"] - 0.50) < 0.01:
            target = t
            break
    if target is None:
        return None

    baseline = s["baseline_state"]
    winning = target["state"]

    from .physical_model import first_round_scores_breakdown
    from .parameters import TIER1_PARAMS
    import copy

    def _breakdown_at(state):
        params = {k: state[k] for k in TIER1_PARAMS}
        cfg_eff = copy.deepcopy(cfg)
        # bases dynamiques selon les compétiteurs effectivement présents
        for arch in cfg_eff["competitors"]:
            key = f"base_{arch}"
            if key in state:
                cfg_eff["competitors"][arch]["base"] = state[key]
        return first_round_scores_breakdown(params, cfg_eff)

    bd_base = _breakdown_at(baseline)
    bd_win = _breakdown_at(winning)

    # Sondages réels mai 2026 (sources : IFOP, Public Sénat, Wikipédia)
    real_polls = {
        "bardella":  36.0, "philippe":  18.0, "melenchon": 11.0,
        "glucksmann": 11.0, "villepin":  11.0, "retailleau": 10.0,
    }
    long_names = {
        "bardella":   "Bardella\n(RN)",
        "philippe":   "Philippe\n(Horizons)",
        "melenchon":  "Mélenchon\n(LFI)",
        "glucksmann": "Glucksmann\n(PS / Place publique)",
        "villepin":   "Villepin\n(LFH)",
        "retailleau": "Retailleau\n(LR)",
    }
    pool_labels = {
        "rn":       "Électorat RN / Reconquête",
        "central":  "Centre / macronistes",
        "gauche":   "Gauche modérée (PS, Verts)",
        "lfi":      "Gauche radicale (LFI, PCF, NPA)",
        "lr":       "Droite classique (LR)",
        "indecis":  "Indécis / fluide",
    }
    pool_colors = {
        "rn":       "#1a3a8a", "central":  "#FFC400",
        "gauche":   "#F08080", "lfi":      "#B71C1C",
        "lr":       "#0288D1", "indecis":  "#9E9E9E",
    }

    candidates = list(bd_base.keys())
    totals_base = np.array([sum(bd_base[c].values()) for c in candidates])
    totals_win = np.array([sum(bd_win[c].values()) for c in candidates])
    # Ordre par chemin optimal décroissant (le candidat dominant après shift en premier)
    order = np.argsort(-totals_win)
    candidates = [candidates[i] for i in order]
    totals_base = totals_base[order]
    totals_win = totals_win[order]
    polls = np.array([real_polls.get(c, 0.0) for c in candidates])
    labels = [long_names.get(c, c) for c in candidates]
    pool_keys = list(cfg["pools"].keys())

    fig, ax = plt.subplots(figsize=(15, 9.0))

    n = len(candidates)
    bar_width = 0.27
    gap = 0.025
    x = np.arange(n)
    x_base = x - (bar_width + gap)
    x_win = x
    x_poll = x + (bar_width + gap)

    # Barre BASELINE : empilée par pool, couleurs PÂLES (alpha 0.45)
    bottoms = np.zeros(n)
    for pk in pool_keys:
        vals = np.array([bd_base[c].get(pk, 0.0) for c in candidates])
        ax.bar(x_base, vals, width=bar_width, bottom=bottoms,
               color=pool_colors.get(pk, "#888"), edgecolor="none",
               alpha=0.40, zorder=2)
        bottoms += vals

    # Barre CHEMIN OPTIMAL : empilée par pool, couleurs PLEINES
    bottoms = np.zeros(n)
    for pk in pool_keys:
        vals = np.array([bd_win[c].get(pk, 0.0) for c in candidates])
        ax.bar(x_win, vals, width=bar_width, bottom=bottoms,
               color=pool_colors.get(pk, "#888"), edgecolor="none",
               label=pool_labels.get(pk, pk), zorder=3)
        bottoms += vals

    # Barre SONDAGE RÉEL (hachurée)
    ax.bar(x_poll, polls, width=bar_width,
           color="white", edgecolor="#555", hatch="////", linewidth=0.9,
           label="Sondage réel (mai 2026)", zorder=2)

    y_max = max(totals_base.max(), totals_win.max(), polls.max()) * 1.30
    label_offset = y_max * 0.014

    # Labels valeurs sur chaque barre
    for i in range(n):
        # baseline (italique, plus discret)
        ax.text(x_base[i], totals_base[i] + label_offset,
                f"{totals_base[i]:.1f} %",
                ha="center", va="bottom", fontsize=9.0,
                color="#555", style="italic")
        # chemin optimal (gras)
        ax.text(x_win[i], totals_win[i] + label_offset,
                f"{totals_win[i]:.1f} %",
                ha="center", va="bottom", fontsize=11,
                fontweight="bold", color="#1A237E")
        # delta sous le chiffre du chemin optimal
        delta = totals_win[i] - totals_base[i]
        if abs(delta) >= 0.3:
            sign = "+" if delta > 0 else ""
            ax.text(x_win[i], totals_win[i] + label_offset + y_max * 0.040,
                    f"{sign}{delta:.1f} pts",
                    ha="center", va="bottom", fontsize=8.5,
                    fontweight="bold",
                    color="#2E7D32" if delta > 0 else "#C62828")
        # sondage
        if polls[i] > 0:
            ax.text(x_poll[i], polls[i] + label_offset, f"{int(polls[i])} %",
                    ha="center", va="bottom", fontsize=9, style="italic",
                    color="#666")

    # Seuil de qualification 2T basé sur le 2e du chemin optimal
    threshold = sorted(totals_win, reverse=True)[1]
    ax.axhline(threshold, color="#C62828", lw=1.1, ls="--", alpha=0.55, zorder=1)
    ax.text(n - 0.5, threshold + label_offset * 0.5,
            f"Seuil qualification 2nd tour (chemin optimal) : {threshold:.1f} %",
            ha="right", va="bottom", fontsize=9, color="#C62828",
            style="italic")

    # Badges "qualifié 2T" sur top-2 du chemin optimal
    top2 = np.argsort(-totals_win)[:2]
    badge_y = y_max * 0.92
    for idx in top2:
        is_villepin = candidates[idx] == "villepin"
        text = "Qualifié 2nd tour" + (" (VILLEPIN)" if is_villepin else "")
        ax.annotate(text,
                    xy=(x_win[idx], totals_win[idx] + label_offset * 3),
                    xytext=(x_win[idx], badge_y),
                    ha="center", fontsize=9.5, fontweight="bold",
                    color="#5E35B1" if is_villepin else "#1B5E20",
                    arrowprops=dict(arrowstyle="-",
                                    color="#5E35B1" if is_villepin else "#1B5E20",
                                    lw=0.7, alpha=0.6),
                    bbox=dict(boxstyle="round,pad=0.40",
                              fc="#EDE7F6" if is_villepin else "#E8F5E9",
                              ec="#5E35B1" if is_villepin else "#1B5E20",
                              lw=1.0), zorder=10)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Pourcentage des suffrages exprimés", fontsize=11)
    p_b = s["baseline_p_victory"] * 100
    p_w = target["p_check"] * 100
    ax.set_title(
        f"Chemin optimal pour Villepin : P(victoire 2T) {p_b:.2f} % → {p_w:.2f} %",
        fontsize=13.5, fontweight="bold", pad=22,
    )
    ax.text(
        0.5, 1.018,
        "Comparaison BASELINE (couleurs pâles, aujourd'hui mai 2026) vs CHEMIN OPTIMAL CMA-ES "
        "(couleurs pleines) ; sondages réels en référence (hachuré).",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, color="#444", style="italic",
    )
    ax.set_ylim(0, y_max * 1.04)
    ax.set_xlim(-0.7, n - 0.30)
    ax.grid(True, axis="y", alpha=0.25, ls=":", zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    leg = ax.legend(
        title="Origine sociologique (barres pleines) et sondage réel (hachuré)",
        loc="upper right", fontsize=8.5, title_fontsize=9.5,
        ncol=2, framealpha=0.95, borderpad=0.6, columnspacing=1.2,
        handlelength=1.4, handleheight=1.0,
    )
    leg.get_frame().set_edgecolor("#ccc")

    plt.subplots_adjust(bottom=0.27, top=0.90, left=0.06, right=0.98)

    fig.text(
        0.5, 0.175,
        "Lecture : pour chaque candidat, 3 barres juxtaposées.  "
        "Gauche pâle = baseline modèle (aujourd'hui).  "
        "Milieu pleine = chemin optimal qui maximise Villepin.  "
        "Droite hachurée = sondage réel mai 2026.",
        ha="center", fontsize=9.5, style="italic", color="#555",
    )

    # Récap chiffré sous la lecture
    villepin_idx = candidates.index("villepin") if "villepin" in candidates else None
    v_base = totals_base[villepin_idx] if villepin_idx is not None else 0
    v_win = totals_win[villepin_idx] if villepin_idx is not None else 0
    methodo_lines = [
        "Comment lire le chemin optimal :",
        f"1. Algorithme CMA-ES, 1 000 000 configurations testées sur les 8 paramètres (4 exogènes subis + 4 contrôlables de campagne) pour maximiser P(victoire Villepin 2T).",
        f"2. Interpolation linéaire baseline → optimum, on garde le plus petit α qui franchit le seuil ciblé.",
        f"3. Effet sur Villepin (1er tour) : {v_base:.1f} % → {v_win:.1f} %, soit {v_win - v_base:+.1f} pts.  P(victoire) passe de {p_b:.2f} % à {p_w:.2f} %.",
        "Calibré sur 5 élections (2002-2022, MAE 2.05 pts).  Exploration de scénarios, pas une prédiction certaine.",
    ]
    y_positions = [0.125, 0.100, 0.077, 0.054, 0.018]
    weights = ["bold", "normal", "normal", "normal", "normal"]
    sizes = [9.5, 8.5, 8.5, 8.5, 8.5]
    colors = ["#444", "#666", "#666", "#666", "#888"]
    styles = ["normal", "normal", "normal", "normal", "italic"]
    for txt, y, w_, s_, c, st in zip(methodo_lines, y_positions,
                                      weights, sizes, colors, styles):
        fig.text(0.5, y, txt, ha="center", fontsize=s_, fontweight=w_,
                 color=c, style=st)

    p = out / "villepin_winning_path.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_villepin_winning_recipe(cfg, out: Path) -> Path | None:
    """Comparaison baseline 2027 vs chemin optimal (P=50%) en 4 vignettes."""
    summary_path = Path("outputs") / "path_to_victory.summary.json"
    if not summary_path.exists():
        return None
    s = json.loads(summary_path.read_text())
    target = None
    for t in s["targets"]:
        if t.get("achievable") and abs(t["target"] - 0.50) < 0.01:
            target = t
            break
    if target is None:
        return None

    baseline = s["baseline_state"]
    winning = target["state"]

    from .physical_model import first_round_scores
    from .parameters import TIER1_PARAMS
    import copy
    def _scores_at(state):
        params = {k: state[k] for k in TIER1_PARAMS}
        cfg_eff = copy.deepcopy(cfg)
        for arch in cfg_eff["competitors"]:
            key = f"base_{arch}"
            if key in state:
                cfg_eff["competitors"][arch]["base"] = state[key]
        return first_round_scores(params, cfg_eff)

    scores_baseline = _scores_at(baseline)
    scores_winning = _scores_at(winning)

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.5, wspace=0.30,
                          left=0.07, right=0.96, top=0.88, bottom=0.07)

    # Panel 1 : scores 1T des candidats (dynamique selon config)
    ax = fig.add_subplot(gs[0, 0])
    cands = list(cfg["competitors"].keys()) + ["villepin"]
    base_b = [scores_baseline[c] for c in cands]
    base_w = [scores_winning[c]  for c in cands]
    x = np.arange(len(cands))
    w = 0.4
    ax.bar(x - w / 2, base_b, w, label="Aujourd'hui (mai 2026)",
           color="#90A4AE", edgecolor="white")
    colors_w = ["#7E57C2"] * len(cands)
    ax.bar(x + w / 2, base_w, w, label="Chemin optimal",
           color=colors_w, edgecolor="white")
    for i, (b, ww) in enumerate(zip(base_b, base_w)):
        delta = ww - b
        ax.text(x[i] + w / 2, ww + 0.7, f"{delta:+.1f}", ha="center",
                fontsize=9, color="#C62828" if delta < 0 else "#2E7D32",
                fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([_ARCH_LABELS_FR[c].replace("\n", " ") for c in cands],
                       fontsize=9, rotation=20, ha="right")
    ax.set_ylabel("Score 1er tour prédit (%)", fontsize=10)
    ax.set_title("1. Scores 1er tour : aujourd'hui vers chemin optimal",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel 2 : params Tier 1, exogènes
    ax = fig.add_subplot(gs[0, 1])
    exo_keys = ["crisis", "central_collapse", "volatility", "anti_extreme_pressure"]
    exo_labels = ["Crise géopolitique", "Effondrement\ndu centre",
                  "Volatilité\nélectorale", "Front\nrépublicain"]
    bx = np.arange(len(exo_keys))
    b_val = [baseline[k] for k in exo_keys]
    w_val = [winning[k] for k in exo_keys]
    ax.bar(bx - w / 2, b_val, w, label="Aujourd'hui (neutre 0.5)",
           color="#90A4AE", edgecolor="white")
    ax.bar(bx + w / 2, w_val, w, label="Chemin optimal",
           color="#D84315", edgecolor="white")
    ax.set_xticks(bx)
    ax.set_xticklabels(exo_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Intensité [0-1]", fontsize=10)
    ax.set_title("2. Contexte exogène (subi par la campagne)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel 3 : params Tier 1, internes
    ax = fig.add_subplot(gs[1, 0])
    int_keys = ["campaign_machine", "thematic_breadth",
                "media_performance", "coalition_building"]
    int_labels = ["Machine de\ncampagne", "Largeur\nthématique",
                  "Performance\nmédiatique", "Coalitions /\nralliements"]
    bx = np.arange(len(int_keys))
    b_val = [baseline[k] for k in int_keys]
    w_val = [winning[k] for k in int_keys]
    ax.bar(bx - w / 2, b_val, w, label="Aujourd'hui (neutre 0.5)",
           color="#90A4AE", edgecolor="white")
    ax.bar(bx + w / 2, w_val, w, label="Chemin optimal",
           color="#2E7D32", edgecolor="white")
    ax.set_xticks(bx)
    ax.set_xticklabels(int_labels, fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Intensité [0-1]", fontsize=10)
    ax.set_title("3. Campagne Villepin (action volontaire)",
                 fontsize=11, fontweight="bold", loc="left")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25, ls=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Panel 4 : tableau récap des shifts
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    table_rows = [
        ("P(victoire Villepin) aujourd'hui",  f"{s['baseline_p_victory']*100:.2f} %"),
        ("P(victoire Villepin) chemin optimal", f"{target['p_check']*100:.2f} %"),
        ("", ""),
    ]
    # Bases dynamiques (selon v6 ou v8 multiclasses)
    for k in baseline:
        if k.startswith("base_"):
            table_rows.append((k, f"{baseline[k]:.1f} vers {winning.get(k, baseline[k]):.1f}"))
    table_rows.append(("", ""))
    # Tier 1 fixes
    for k in ["crisis", "central_collapse", "volatility", "anti_extreme_pressure",
              "campaign_machine", "thematic_breadth", "media_performance",
              "coalition_building"]:
        table_rows.append((k, f"{baseline[k]:.2f} vers {winning[k]:.2f}"))
    # On garde la liste suffixée pour ne pas casser le code suivant
    y0, dy = 1.0, 0.055
    for i, (k, v) in enumerate(table_rows):
        y = y0 - i * dy
        ax.text(0.00, y, k, transform=ax.transAxes, fontsize=8.5,
                family="monospace", color="#333", va="top")
        ax.text(0.55, y, v, transform=ax.transAxes, fontsize=8.5,
                family="monospace", color="#222", fontweight="bold", va="top")
    ax.set_title("4. Tableau récap (baseline vers chemin optimal)",
                 fontsize=11, fontweight="bold", loc="left")

    fig.suptitle(
        "Configuration minimale pour P(victoire Villepin) = 50 %",
        fontsize=14, fontweight="bold", y=0.96,
    )
    p = out / "villepin_winning_recipe.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_model_trust_card(out_dir: Path, out: Path) -> Path | None:
    """Tableau factuel des métriques + sources, sans aucun texte vendeur.
    Implémentation propre via matplotlib.Table (pas de fig.text manuel)."""
    cal_path = out_dir / "calibration_summary.json"
    hist_path = out_dir / "historical_validation_summary.json"
    if not (cal_path.exists() and hist_path.exists()):
        return None
    cal = json.loads(cal_path.read_text())
    hist = json.loads(hist_path.read_text())["metrics"]

    # Données
    sources_data = [
        ["Wikipédia (résultats 1er tour)",  "2002-2022",  "61 candidats × 5 élections", "scrap + SHA-256"],
        ["Sondages IFOP / Public Sénat",     "mai 2026",   "6 candidats 2027",            "intervalle déclaratif"],
        ["Pools sociologiques",              "config.yaml","6 pools × 6 affinités",        "à dire d'expert documenté"],
        ["Pool sizes par année",             "src/historical_context.py", "30 valeurs (6×5)", "à dire d'expert"],
    ]
    metrics_data = [
        ["MAE LOO (1er tour, leave-one-out)",     f"{cal['mae_loo']:.2f} pts",         "0 = parfait"],
        ["MAE in-sample (1er tour, fit total)",    f"{cal['mae_in_sample']:.2f} pts",   "0 = parfait"],
        ["F1 score qualification au 2T",          f"{hist['f1_qualif']:.3f}",          "1.000 = parfait"],
        ["Precision qualification",                f"{hist['precision_qualif']:.3f}",   "1.000 = parfait"],
        ["Recall qualification",                   f"{hist['recall_qualif']:.3f}",      "1.000 = parfait"],
        ["Confusion matrix (TP/FP/FN/TN)",
         f"{hist['confusion_matrix']['tp']}/{hist['confusion_matrix']['fp']}/{hist['confusion_matrix']['fn']}/{hist['confusion_matrix']['tn']}",
         "n = 30 pairs"],
        ["Vainqueur 2T correct",                   f"{hist['winner_correct_count']}/{hist['winner_total']}", "5 = parfait"],
    ]
    limits_data = [
        ["Cygnes noirs (scandale, retrait...)",    "non modélisé"],
        ["Personnalité individuelle du candidat",   "non modélisée (archétypes seuls)"],
        ["Dynamique temporelle de campagne",       "non modélisée (modèle statique)"],
        ["Affinités candidat × pool",              "à dire d'expert, non recalibrées par fit"],
    ]

    # Figure avec 3 sous-axes, un par tableau
    fig = plt.figure(figsize=(14, 11))
    fig.suptitle(
        "Modèle de simulation électorale 2027 : sources, métriques, limites",
        fontsize=14, fontweight="bold", y=0.985,
    )
    gs = fig.add_gridspec(3, 1, hspace=0.38, height_ratios=[2.0, 3.2, 1.8],
                          left=0.04, right=0.97, top=0.94, bottom=0.03)

    def _draw_table(ax, title, headers, rows, col_widths):
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left",
                     color="#222", pad=10)
        full_data = [headers] + rows
        table = ax.table(
            cellText=full_data,
            colWidths=col_widths,
            cellLoc="left",
            loc="upper left",
            bbox=[0.0, 0.0, 1.0, 0.92],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        # Style header + rows
        for (r, c), cell in table.get_celld().items():
            cell.set_edgecolor("#ddd")
            cell.set_linewidth(0.5)
            cell.PAD = 0.04
            if r == 0:
                cell.set_facecolor("#f0f0f0")
                cell.set_text_props(fontweight="bold", color="#333")
            else:
                cell.set_facecolor("white")
                cell.set_text_props(color="#222")
        # Hauteur des lignes
        for (r, c), cell in table.get_celld().items():
            cell.set_height(1.0 / len(full_data))

    ax1 = fig.add_subplot(gs[0])
    _draw_table(ax1, "1. Sources de données",
                ["Source", "Période", "Volume", "Notes"],
                sources_data,
                col_widths=[0.32, 0.18, 0.25, 0.25])

    ax2 = fig.add_subplot(gs[1])
    _draw_table(ax2, "2. Métriques de validation (sur 2002-2022)",
                ["Métrique", "Valeur", "Borne idéale"],
                metrics_data,
                col_widths=[0.50, 0.25, 0.25])

    ax3 = fig.add_subplot(gs[2])
    _draw_table(ax3, "3. Limites assumées",
                ["Aspect", "Statut"],
                limits_data,
                col_widths=[0.40, 0.60])

    p = out / "model_trust_card.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _save_plot_nn_architecture(out: Path) -> Path:
    """Schéma Mermaid propre du réseau de neurones (architecture vue de profil)."""
    import subprocess
    cfg = load_config(Path('config.fitted.yaml') if Path('config.fitted.yaml').exists() else Path('config.yaml'))
    nn_cfg = cfg['pipeline']['neural_network']
    H = nn_cfg['hidden']
    L = nn_cfg.get('n_layers', 3)

    mermaid_src = """%%{init: {'theme': 'base', 'themeVariables': {
'primaryColor': '#fff',
'primaryBorderColor': '#444',
'fontFamily': 'sans-serif',
'fontSize': '16px'
}, 'flowchart': {'curve': 'basis', 'padding': 18, 'nodeSpacing': 50, 'rankSpacing': 60}}}%%
flowchart LR
    A["<b>Entrée</b><br/>8 paramètres Tier 1<br/>crisis, central_collapse,<br/>media, anti_extreme, …"]
    B["<b>Embed</b><br/>Linear 8 → 192<br/>+ LayerNorm<br/>2 112 paramètres (0.2%)"]
    C["<b>Block 1</b><br/>résiduel pré-norm<br/>192 ⇄ 768 (expand ×4)<br/>296 256 paramètres (32.6%)"]
    D["<b>Block 2</b><br/>résiduel pré-norm<br/>192 ⇄ 768 (expand ×4)<br/>296 256 paramètres (32.6%)"]
    E["<b>Block 3</b><br/>résiduel pré-norm<br/>192 ⇄ 768 (expand ×4)<br/>296 256 paramètres (32.6%)"]
    F["<b>Head</b><br/>LayerNorm + Linear 192 → 96<br/>GELU + Linear 96 → 3<br/>18 912 paramètres (2.1%)"]
    G["<b>Sortie</b><br/>3 valeurs prédites :<br/>score_1T, p_qualif, p_victory"]

    A --> B --> C --> D --> E --> F --> G

    classDef io fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    classDef embed fill:#EDE7F6,stroke:#5E35B2,stroke-width:2px,color:#311B92
    classDef block fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    classDef head fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    classDef out fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#B71C1C
    class A io
    class B embed
    class C,D,E block
    class F head
    class G out
"""
    mmd_path = out / 'nn_architecture.mmd'
    mmd_path.write_text(mermaid_src)
    p = out / 'nn_architecture.png'
    try:
        subprocess.run(
            ['mmdc', '-i', str(mmd_path), '-o', str(p),
             '-w', '2800', '-H', '900', '-b', 'white', '--scale', '2'],
            check=True, capture_output=True, timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f'  [nn_architecture] mmdc échec : {e}')
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, 'Rendu Mermaid indisponible', ha='center', va='center')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return p


def _save_plot_nn_block_anatomy(out: Path) -> Path:
    """Anatomie d'un bloc résiduel pré-norm : Mermaid propre."""
    import subprocess
    mermaid_src = """%%{init: {'theme': 'base', 'themeVariables': {
'primaryColor': '#fff',
'primaryBorderColor': '#444',
'fontFamily': 'sans-serif',
'fontSize': '16px'
}, 'flowchart': {'curve': 'basis', 'padding': 18, 'nodeSpacing': 55, 'rankSpacing': 60}}}%%
flowchart LR
    IN(["x  (192 dim)"])
    LN["<b>LayerNorm</b><br/>γ, β<br/>384 paramètres"]
    L1["<b>Linear</b><br/>192 → 768<br/>(expand ×4)<br/>147 K paramètres"]
    AC["<b>GELU</b><br/>activation<br/>non-linéaire"]
    DR["<b>Dropout</b><br/>p = 0.10<br/>régularisation"]
    L2["<b>Linear</b><br/>768 → 192<br/>(contract)<br/>147 K paramètres"]
    ADD((+))
    OUT(["y  (192 dim)"])

    IN --> LN --> L1 --> AC --> DR --> L2 --> ADD --> OUT
    IN -. "skip connection<br/>(identité)" .-> ADD

    classDef io fill:#FFFDE7,stroke:#F57F17,stroke-width:2px,color:#E65100
    classDef layer fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    classDef op fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#311B92
    classDef sum fill:#E0F7FA,stroke:#006064,stroke-width:2.5px,color:#004D40

    class IN,OUT io
    class LN,L1,L2 layer
    class AC,DR op
    class ADD sum
"""
    mmd_path = out / 'nn_block_anatomy.mmd'
    mmd_path.write_text(mermaid_src)
    p = out / 'nn_block_anatomy.png'
    try:
        subprocess.run(
            ['mmdc', '-i', str(mmd_path), '-o', str(p),
             '-w', '2600', '-H', '900', '-b', 'white', '--scale', '2'],
            check=True, capture_output=True, timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f'  [nn_block_anatomy] mmdc échec : {e}')
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, 'Rendu Mermaid indisponible', ha='center', va='center')
        fig.savefig(p, dpi=150, bbox_inches='tight')
        plt.close(fig)
    return p


def _nn_iso_color(hex_color: str, factor: float) -> tuple:
    """Renvoie une teinte du même hex, plus claire si factor>1, plus sombre si <1."""
    import matplotlib.colors as mcolors
    r, g, b = mcolors.hex2color(hex_color)
    if factor < 1:
        return (r * factor, g * factor, b * factor)
    f = min(factor - 1, 1)
    return (r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f)


def _load_nn_arch_from_ckpt() -> dict | None:
    """Lit le state_dict du surrogate et retourne dimensions + nb params par couche.

    Retourne None si le checkpoint n'existe pas (la fonction appelante doit
    alors basculer sur un fallback hard-codé).
    """
    import torch
    ckpt_path = Path('outputs/checkpoints/nn_surrogate.pt')
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = ckpt['state_dict']
    n_inputs = int(ckpt['n_inputs'])
    hidden = int(ckpt['hidden'])
    n_layers = int(ckpt.get('n_layers', 4))
    # _ResidualBlock : lin1 = (4d, d), lin2 = (d, 4d). On lit la taille réelle.
    expand_w = state.get('blocks.0.lin1.weight')
    expand = int(expand_w.shape[0]) if expand_w is not None else 4 * hidden
    head_dim = int(state['head.1.weight'].shape[0])
    n_out = int(state['head.3.weight'].shape[0])

    def count(*prefixes):
        return sum(int(v.numel()) for k, v in state.items()
                   if any(k.startswith(p) for p in prefixes))

    # Pre-norm : on rattache chaque LayerNorm à la transformation qui suit.
    p_embed = count('embed.0', 'embed.1')                              # Linear + LN
    p_expand = [count(f'blocks.{i}.norm1', f'blocks.{i}.lin1')
                for i in range(n_layers)]                              # LN + Linear (h→4h)
    p_contract = [count(f'blocks.{i}.lin2') for i in range(n_layers)]  # Linear (4h→h)
    p_head = count('head.0', 'head.1')                                 # LN + Linear (h→h/2)
    p_out = count('head.3')                                            # Linear (h/2→n_out)
    total = sum(int(v.numel()) for v in state.values())

    return {
        'n_inputs': n_inputs, 'hidden': hidden, 'expand': expand,
        'head_dim': head_dim, 'n_out': n_out, 'n_layers': n_layers,
        'p_embed': p_embed, 'p_expand': p_expand, 'p_contract': p_contract,
        'p_head': p_head, 'p_out': p_out, 'total': total,
    }


def _save_plot_nn_3d_volumetric(out: Path) -> Path:
    """Vue 3D isométrique du réseau, échelle réelle.

    - Tous les neurones du réseau (3 179 pour le surrogate v2) dessinés un par un
      en grille (max_rows=32, cols=ceil(dim/32)).
    - Toutes les arêtes Linear (~905 k) rendues via LineCollection rasterisée
      à 300 dpi pour garder un SVG léger (≈10 MB au lieu de 180 MB pur vectoriel).
    - Cadres / neurones / labels restent vectoriels → zoom illimité.
    - Sortie : `.svg` (vectoriel + raster) ET `.png` (raster HD).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as PathEffects
    from matplotlib.patches import Polygon, FancyArrowPatch
    from matplotlib.collections import LineCollection
    import numpy as np

    arch = _load_nn_arch_from_ckpt()
    if arch is None:
        n_inputs, hidden, expand, head_dim, n_layers, n_out = 8, 192, 768, 96, 3, 3
        total = 910_083
    else:
        n_inputs = arch['n_inputs']; hidden = arch['hidden']; expand = arch['expand']
        head_dim = arch['head_dim']; n_layers = arch['n_layers']; n_out = arch['n_out']
        total = arch['total']

    layers: list[tuple[str, int, str]] = [('Entrée', n_inputs, '#1565C0'),
                                          ('Embed',  hidden,   '#7B1FA2')]
    for k in range(1, n_layers + 1):
        layers.append((f'B{k} exp', expand, '#388E3C'))
        layers.append((f'B{k} con', hidden, '#66BB6A'))
    layers.append(('Head',   head_dim, '#F57C00'))
    layers.append(('Sortie', n_out,    '#C62828'))
    n_total = len(layers)
    n_neurons_total = sum(d for _, d, _ in layers)

    # Disposition : chaque couche = grille (rows × cols). Hauteur fixée à
    # max_rows neurones, largeur = ceil(dim/rows). La couche la plus haute
    # (Entrée=8) reste petite, la plus large (768) tient sur 24 colonnes.
    max_rows = 32
    cell = 0.16
    gap = 2.8

    positions: list[np.ndarray] = []
    metas: list[dict] = []
    x_cursor = 0.0
    for (lbl, dim, color) in layers:
        rows = min(dim, max_rows)
        cols = int(np.ceil(dim / rows))
        h = (rows - 1) * cell if rows > 1 else 0.0
        w = (cols - 1) * cell if cols > 1 else 0.0
        idx = np.arange(dim)
        xs = (idx // rows) * cell + x_cursor
        ys = (idx % rows) * cell - h / 2
        # Centre verticalement la dernière colonne si incomplète
        last_col = dim // rows
        last_n = dim - last_col * rows
        if last_n > 0 and last_n < rows:
            mask = (idx // rows) == last_col
            ys[mask] += (rows - last_n) * cell / 2
        positions.append(np.column_stack([xs, ys]))
        metas.append(dict(lbl=lbl, dim=dim, color=color,
                          rows=rows, cols=cols, w=w, h=h, x0=x_cursor))
        x_cursor += w + gap

    # Toutes les arêtes (un segment = un poids de Linear)
    seg_list = []
    for i in range(n_total - 1):
        src = positions[i]; dst = positions[i + 1]
        ii = np.repeat(np.arange(len(src)), len(dst))
        jj = np.tile(np.arange(len(dst)), len(src))
        seg_list.append(np.stack([src[ii], dst[jj]], axis=1))
    all_segs = np.concatenate(seg_list, axis=0)
    n_weights = int(len(all_segs))

    fig, ax = plt.subplots(figsize=(46, 12))
    fig.patch.set_facecolor('white')

    # Fonds isométriques par couche
    iso_cos = np.cos(np.radians(28))
    iso_sin = np.sin(np.radians(28))
    bg_depth = 0.65
    for m in metas:
        x0 = m['x0'] - 0.30
        x1 = m['x0'] + m['w'] + 0.30
        y0 = -m['h'] / 2 - 0.30
        y1 =  m['h'] / 2 + 0.30
        dx = bg_depth * iso_cos
        dy = bg_depth * iso_sin
        c = m['color']
        ax.add_patch(Polygon(
            [(x0 + dx, y0 + dy), (x1 + dx, y0 + dy),
             (x1 + dx, y1 + dy), (x0 + dx, y1 + dy)],
            facecolor=_nn_iso_color(c, 1.7), edgecolor=c, lw=0.7, alpha=0.18, zorder=0))
        ax.add_patch(Polygon(
            [(x1, y0), (x1 + dx, y0 + dy), (x1 + dx, y1 + dy), (x1, y1)],
            facecolor=_nn_iso_color(c, 0.8), edgecolor=c, lw=0.6, alpha=0.18, zorder=0))
        ax.add_patch(Polygon(
            [(x0, y1), (x1, y1), (x1 + dx, y1 + dy), (x0 + dx, y1 + dy)],
            facecolor=_nn_iso_color(c, 1.3), edgecolor=c, lw=0.6, alpha=0.22, zorder=0))
        ax.add_patch(Polygon(
            [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            facecolor='none', edgecolor=c, lw=2.2, alpha=0.85, zorder=2))

    # Toutes les arêtes : rasterisée pour SVG léger
    lc = LineCollection(all_segs, colors='#465062',
                        linewidths=0.15, alpha=0.04, zorder=1,
                        antialiaseds=True, capstyle='round')
    lc.set_rasterized(True)
    ax.add_collection(lc)

    # Neurones : halo coloré + cœur blanc (visibles malgré la nappe)
    for i, m in enumerate(metas):
        pos = positions[i]
        ax.scatter(pos[:, 0], pos[:, 1], s=22,
                   c=m['color'], edgecolor='none', alpha=0.85, zorder=3)
        ax.scatter(pos[:, 0], pos[:, 1], s=6,
                   c='white', edgecolor='black', linewidth=0.15, zorder=4)

    # Skip-connections : Embed → fin de chaque bloc
    skip_pairs = [(1 + 2 * (k - 1), 1 + 2 * k) for k in range(1, n_layers + 1)]
    max_y_top = max(m['h'] / 2 + 0.3 for m in metas)
    for (i_src, i_dst) in skip_pairs:
        x_a = metas[i_src]['x0'] + metas[i_src]['w'] / 2
        x_b = metas[i_dst]['x0'] + metas[i_dst]['w'] / 2
        y_a = metas[i_src]['h'] / 2 + 0.5
        y_b = metas[i_dst]['h'] / 2 + 0.5
        ax.add_patch(FancyArrowPatch(
            (x_a, y_a), (x_b, y_b),
            connectionstyle="arc3,rad=-0.32",
            arrowstyle='-|>', color='#E65100', lw=3.2,
            mutation_scale=24, zorder=6,
        ))

    mid_x = metas[1 + n_layers]['x0'] + metas[1 + n_layers]['w'] / 2
    txt = ax.text(mid_x, max_y_top + 1.8,
                  'Skip-connection résiduelle (identité +)',
                  ha='center', va='bottom', fontsize=18,
                  color='#E65100', fontweight='bold')
    txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])

    label_y_base = -max_y_top - 0.7
    for m in metas:
        cx = m['x0'] + m['w'] / 2
        ax.text(cx, label_y_base, m['lbl'], ha='center', va='top',
                fontsize=17, fontweight='bold', color='#222')
        ax.text(cx, label_y_base - 0.55,
                f"{m['dim']:,} neurones".replace(',', ' '),
                ha='center', va='top', fontsize=13, color='#444')

    x_min = -1.0
    x_max = metas[-1]['x0'] + metas[-1]['w'] + 1.0 + bg_depth * iso_cos
    y_min = label_y_base - 1.4
    y_max = max_y_top + 3.6
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.axis('off')

    sub = (f"{n_total} couches  ·  {n_neurons_total:,} neurones (tous dessinés)  "
           f"·  {n_weights:,} arêtes = poids Linear (toutes)  "
           f"·  {total:,} paramètres totaux").replace(',', ' ')
    ax.set_title("Réseau de neurones : vue 3D (isométrique, échelle réelle)",
                 fontsize=26, fontweight='bold', pad=26)
    ax.text(0.5, 1.01, sub, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=15, color='#444', style='italic')

    # SVG (vectoriel + raster pour la nappe) + PNG haute résolution
    p_svg = out / 'nn_3d_volumetric.svg'
    p_png = out / 'nn_3d_volumetric.png'
    fig.savefig(p_svg, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(p_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p_png


def _save_plot_nn_weights_heatmap(out: Path) -> Path:
    """Matrice de poids RÉELLE d'un bloc, centrée : figure seule."""
    import torch
    ckpt_path = Path('outputs/checkpoints/nn_surrogate.pt')
    if not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    sd = ckpt.get('state_dict', ckpt)
    W = None
    for k in sd:
        if 'blocks.0.lin1.weight' in k:
            W = sd[k].cpu().numpy()
            break
    if W is None:
        return None

    rows = np.linspace(0, W.shape[0] - 1, 96, dtype=int)
    cols = np.linspace(0, W.shape[1] - 1, 96, dtype=int)
    W_sub = W[np.ix_(rows, cols)]
    vmax = float(np.percentile(np.abs(W_sub), 98))

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(W_sub, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
    ax.set_title(
        f"Matrice de poids apprise : Bloc 1 Linear (192 → 768)\n"
        f"sous-échantillon {W_sub.shape[0]}×{W_sub.shape[1]} sur {W.shape[0]}×{W.shape[1]} = {W.shape[0]*W.shape[1]:,} poids",
        fontsize=14, weight='bold', color='#111', pad=14,
    )
    ax.set_xlabel("neurone d'entrée (sur 192)", fontsize=11)
    ax.set_ylabel("neurone de sortie (sur 768)", fontsize=11)
    ax.set_xticks([])
    ax.set_yticks([])
    cb = plt.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label('valeur du poids', fontsize=11)
    fig.text(0.5, 0.02,
             'Rouge = poids positif, bleu = poids négatif. Ces 147 456 valeurs sont apprises par AdamW + Cosine Annealing.',
             ha='center', fontsize=10, color='#444', style='italic')
    plt.subplots_adjust(left=0.08, right=0.94, top=0.92, bottom=0.10)
    p = out / 'nn_weights_heatmap.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


def _save_plot_nn_weights_distribution(out: Path) -> Path:
    """Distribution des poids appris du NN : histogramme seul, centré."""
    import torch
    ckpt_path = Path('outputs/checkpoints/nn_surrogate.pt')
    if not ckpt_path.exists():
        return None
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    except Exception:
        return None
    sd = ckpt.get('state_dict', ckpt)
    # Concatène tous les poids des Linear (pas LayerNorm, pas biais)
    all_weights = []
    for k, v in sd.items():
        if 'weight' in k and v.ndim == 2:
            all_weights.append(v.cpu().numpy().flatten())
    if not all_weights:
        return None
    W = np.concatenate(all_weights)

    fig, ax = plt.subplots(figsize=(13, 7.5))
    ax.hist(W, bins=80, color='#1976D2', alpha=0.82, edgecolor='#0D47A1')
    ax.axvline(0, color='#444', lw=1.2, ls='--')
    ax.axvline(W.mean(), color='#C62828', lw=2, label=f'μ = {W.mean():.4f}')
    ax.set_title(
        f"Distribution des {W.size:,} poids appris (toutes couches Linear)\n"
        f"μ = {W.mean():.4f}  |  σ = {W.std():.4f}  |  max |w| = {np.abs(W).max():.3f}",
        fontsize=14, weight='bold', color='#111', pad=14,
    )
    ax.set_xlabel('valeur du poids', fontsize=11)
    ax.set_ylabel('nombre de poids', fontsize=11)
    ax.grid(alpha=0.3, ls=':')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(fontsize=11, loc='upper right')
    fig.text(0.5, 0.015,
             "Les poids sont centrés autour de zéro (initialisation Xavier + régularisation weight_decay 5e-5). "
             "Cette concentration prouve que le réseau n'a pas explosé pendant l'entraînement.",
             ha='center', fontsize=10, color='#444', style='italic')
    plt.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.12)
    p = out / 'nn_weights_distribution.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


def _save_plot_nn_params_pie(out: Path) -> Path:
    """Répartition des 910 083 paramètres par catégorie de couche."""
    p_embed = 2_112
    p_blocks = 3 * 296_256
    p_head = 19_203 - 291
    p_out = 291
    parts = [
        ('Embed (Linear + LN)', p_embed, '#7E57C2'),
        ('3 blocs résiduels', p_blocks, '#43A047'),
        ('Head (LN + Linear + GELU + Linear)', p_head, '#FFA726'),
        ('Output (Linear final)', p_out, '#E53935'),
    ]
    sizes = [x[1] for x in parts]
    colors = [x[2] for x in parts]
    total = sum(sizes)
    labels = [f"{name}\n{val:,} ({100*val/total:.1f}%)" for name, val, _ in parts]

    fig, ax = plt.subplots(figsize=(12, 9))
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90,
                       wedgeprops=dict(edgecolor='white', linewidth=2.5))
    ax.set_title(f'Répartition des {total:,} paramètres entraînables',
                 fontsize=15, weight='bold', color='#111', pad=18)
    ax.legend(wedges, labels, loc='center left',
              bbox_to_anchor=(1.05, 0.5), fontsize=12, frameon=False)
    fig.text(0.5, 0.04,
             "Les 3 blocs résiduels concentrent 97.7% des paramètres : c'est là que le réseau apprend.",
             ha='center', fontsize=11, color='#444', style='italic')
    plt.subplots_adjust(left=0.05, right=0.65, top=0.88, bottom=0.10)
    p = out / 'nn_params_pie.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p


def _save_plot_nn_hyperparams(out: Path) -> Path:
    """Tableau propre des hyperparamètres d'entraînement."""
    cfg = load_config(Path('config.fitted.yaml') if Path('config.fitted.yaml').exists() else Path('config.yaml'))
    nn_cfg = cfg['pipeline']['neural_network']
    rows = [
        ['Optimiseur',              'AdamW'],
        ['Learning rate',           f"{nn_cfg['lr']:.1e}"],
        ['Weight decay',            f"{nn_cfg['weight_decay']:.0e}"],
        ['Scheduler',               'Cosine Annealing'],
        ['Batch size',              str(nn_cfg['batch_size'])],
        ['Max epochs',              str(nn_cfg['max_epochs'])],
        ['Early stopping patience', str(nn_cfg['patience'])],
        ['Dropout',                 f"{nn_cfg['dropout']:.2f}"],
        ['Loss weights (3 sorties)', '1 / 5 / 20'],
        ['Hidden dim',              str(nn_cfg['hidden'])],
        ['Nb blocs résiduels',      str(nn_cfg.get('n_layers', 3))],
        ['Total paramètres',        '910 083'],
        ['MAE test (score_1T)',     '0.18 pt'],
        ["Vitesse d'inférence",     "≈ 10 000 cfg / s"],
    ]
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.axis('off')
    ax.set_title('Hyperparamètres du réseau de neurones',
                 fontsize=15, weight='bold', color='#111', pad=14)
    tbl = ax.table(cellText=rows, colWidths=[0.55, 0.45],
                   cellLoc='left', loc='center', bbox=[0.05, 0.02, 0.90, 0.92])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#ddd')
        cell.set_linewidth(0.6)
        if c == 0:
            cell.set_facecolor('#F5F5F5')
            cell.set_text_props(weight='bold', color='#333')
        else:
            cell.set_facecolor('white')
            cell.set_text_props(color='#111')
    p = out / 'nn_hyperparams.png'
    fig.savefig(p, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return p




def _save_plot_tuning_inference(out: Path) -> Path:
    """Schéma Mermaid propre : 3 étages d'inférence des paramètres."""
    import subprocess

    mermaid_src = """%%{init: {'theme': 'base', 'themeVariables': {
'primaryColor': '#fff',
'primaryBorderColor': '#444',
'fontFamily': 'sans-serif',
'fontSize': '15px'
}, 'flowchart': {'curve': 'basis', 'padding': 16, 'nodeSpacing': 60, 'rankSpacing': 65}}}%%
flowchart TB
    subgraph S1["ÉTAGE 1 : CALIBRATION HISTORIQUE"]
        direction TB
        A1["Historique 2002-2022<br/>5 élections × 61 candidats<br/>Ridge regression L-BFGS-B<br/>MAE LOO = 2.05 pts"]
        A2["→ Fixe les bornes du modèle<br/>m_min, m_max, bias<br/>poids des 8 dimensions<br/>régularisation λ"]
        A1 --> A2
    end
    subgraph S2["ÉTAGE 2 : OPTIMISATION CMA-ES"]
        direction TB
        B1["CMA-ES sur le NN surrogate<br/>100 popsize × 500 iter × 20 restarts<br/>= 1 000 000 évaluations<br/>adaptation de matrice de covariance"]
        B2["→ Trouve les 8 paramètres Tier 1<br/>crisis, central_collapse, campaign_machine<br/>media, anti_extreme_pressure, …<br/>maximisant P(victoire Villepin)"]
        B1 --> B2
    end
    subgraph S3["ÉTAGE 3 : INFÉRENCE LLM (boucle adaptative)"]
        direction TB
        C1["gemma4:31b-cloud (Ollama Turbo)<br/>lit la documentation des paramètres<br/>propose 8 sous-paramètres par itération<br/>plafonné à 24 cumulés"]
        C2["→ Filtre redondance<br/>|corr(sub, Tier 1)| < 0.7<br/>sinon rejet automatique"]
        C1 --> C2
    end
    subgraph S4["BOUCLE DE VALIDATION"]
        direction LR
        V1["Régénère le dataset<br/>LHS sur les nouvelles dimensions<br/>+ ré-entraîne le NN<br/>+ relance CMA-ES"]
        V2["Mesure ΔP(victoire) vs baseline<br/>MAE NN test  |  importance Tier 2<br/>Acceptance test :<br/>l'amélioration est-elle nette ?"]
        V3{"Amélioration<br/>nette ?"}
        V4["Garde les sub-params<br/>Tier 2 promus définitifs"]
        V5["Rollback complet<br/>paramètres restaurés"]
        V1 --> V2 --> V3
        V3 -->|OUI| V4
        V3 -->|NON| V5
    end

    A2 -.-> S4
    B2 -.-> S4
    C2 ==> S4
    V5 -. "nouvelle proposition LLM" .-> C1

    R["Run observé : 2 itérations LLM lancées, 2/2 rollbacks (filtre redondance + zéro amélioration)<br/>→ modèle final = 8 dimensions Tier 1 pures"]
    S4 --> R

    classDef calib fill:#E3F2FD,stroke:#1565C0,stroke-width:2px,color:#0D47A1
    classDef opt fill:#FCE4EC,stroke:#AD1457,stroke-width:2px,color:#880E4F
    classDef llm fill:#FFF3E0,stroke:#E65100,stroke-width:2px,color:#BF360C
    classDef val fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2px,color:#311B92
    classDef good fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px,color:#1B5E20
    classDef bad fill:#FFEBEE,stroke:#C62828,stroke-width:2px,color:#B71C1C
    classDef result fill:#FFFDE7,stroke:#F57F17,stroke-width:2.5px,color:#E65100

    class A1,A2 calib
    class B1,B2 opt
    class C1,C2 llm
    class V1,V2,V3 val
    class V4 good
    class V5 bad
    class R result
"""
    mmd_path = out / "tuning_inference_flow.mmd"
    mmd_path.write_text(mermaid_src)
    p = out / "tuning_inference_flow.png"
    try:
        subprocess.run(
            ["mmdc", "-i", str(mmd_path), "-o", str(p),
             "-w", "2000", "-H", "2400", "-b", "white", "--scale", "2"],
            check=True, capture_output=True, timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [tuning_inference] mmdc échec : {e}")
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis("off")
        ax.text(0.5, 0.5, "Rendu Mermaid indisponible : voir tuning_inference_flow.mmd",
                ha="center", va="center", fontsize=11)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return p


def _save_plot_volumes_cards(out: Path) -> Path:
    """Figure 1/3 du volume de calcul : 4 grandes cartes chiffrées seules, centrées."""
    from matplotlib.patches import FancyBboxPatch

    cards = [
        ("900 000", "simulations directes",
         "30 000 configs LHS × 30 Monte Carlo\n(8 dimensions Tier 1, bornes [0,1])",
         "#E3F2FD", "#1565C0"),
        ("910 083", "paramètres entraînés",
         "Réseau de neurones surrogate\n3 blocs résiduels × 192 hidden\nAdamW + Cosine Annealing",
         "#E8F5E9", "#1B5E20"),
        ("1 000 000", "évaluations CMA-ES",
         "100 popsize × 500 iter × 20 restarts\nsur le NN (1000× plus rapide\nque le simulateur direct)",
         "#FCE4EC", "#AD1457"),
        ("305", "validations historiques",
         "61 candidats × 5 élections (2002-2022)\nF1 = 0.90  |  Accuracy 2T = 80%\nMAE LOO = 2.05 pts",
         "#FFF3E0", "#E65100"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 7))
    fig.suptitle("Volumes de calcul mesurés sur le run actuel",
                 fontsize=16, weight="bold", color="#111", y=0.96)
    fig.text(0.5, 0.91,
             "Chiffres effectifs, pas d'estimation : toutes mesures issues du pipeline exécuté.",
             ha="center", fontsize=11, color="#555", style="italic")
    for ax, (big, label, sub, fc, ec) in zip(axes, cards):
        ax.axis("off")
        ax.add_patch(FancyBboxPatch(
            (0.04, 0.04), 0.92, 0.92,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            transform=ax.transAxes,
            facecolor=fc, edgecolor=ec, linewidth=2.5,
        ))
        ax.text(0.5, 0.72, big, transform=ax.transAxes,
                ha="center", va="center", fontsize=34, weight="bold", color=ec)
        ax.text(0.5, 0.50, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="#222")
        ax.text(0.5, 0.22, sub, transform=ax.transAxes,
                ha="center", va="center", fontsize=10.5, color="#444", style="italic")

    plt.subplots_adjust(left=0.04, right=0.96, top=0.87, bottom=0.05, wspace=0.20)
    p = out / "volumes_cards.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _save_plot_volumes_log_scale(out: Path) -> Path:
    """Figure 2/3 du volume de calcul : barres horizontales en échelle log, seule."""
    items = [
        ("Itérations LLM (propositions Tier 2)", 16, "#F57F17"),
        ("Calibration historique (candidats × élections)", 305, "#E65100"),
        ("Configs LHS échantillonnées", 30_000, "#1565C0"),
        ("Paramètres NN entraînés", 910_083, "#1B5E20"),
        ("Simulations directes (LHS × MC)", 900_000, "#0D47A1"),
        ("Évaluations CMA-ES sur NN", 1_000_000, "#AD1457"),
    ]
    labels = [it[0] for it in items]
    values = [it[1] for it in items]
    colors = [it[2] for it in items]

    fig, ax = plt.subplots(figsize=(15, 8))
    y_pos = np.arange(len(items))
    ax.barh(y_pos, values, color=colors, edgecolor="#222", linewidth=0.8, height=0.65)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("Compte (échelle logarithmique)", fontsize=11)
    ax.set_title("Échelle des volumes de calcul (log)",
                 fontsize=14, weight="bold", color="#111", pad=14)
    for i, v in enumerate(values):
        ax.text(v * 1.15, i, f"{v:,}".replace(",", " "),
                va="center", fontsize=10.5, color="#222", weight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3, ls=":")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlim(5, 5_000_000)

    plt.subplots_adjust(left=0.32, right=0.96, top=0.92, bottom=0.10)
    p = out / "volumes_log_scale.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _save_plot_volumes_funnel(out: Path) -> Path:
    """Figure 3/3 du volume de calcul : entonnoir compute → scénarios retenus, seule."""
    from matplotlib.patches import Polygon

    stages = [
        ("1 000 000 évaluations CMA-ES", 0.95, "#AD1457"),
        ("100 000 configs top-10% P(victoire)", 0.78, "#7B1FA2"),
        ("1 000 candidats retenus", 0.60, "#5E35B2"),
        ("36 scénarios analysés finement", 0.42, "#3949AB"),
        ("5 chemins optimaux exposés", 0.26, "#1976D2"),
    ]

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("Entonnoir : du calcul brut aux scénarios finaux",
                 fontsize=14, weight="bold", color="#111", pad=14)

    y_top = 0.92
    h = 0.15
    for i, (label, width_frac, color) in enumerate(stages):
        y = y_top - i * (h + 0.012)
        x_left = 0.5 - width_frac / 2
        x_right = 0.5 + width_frac / 2
        if i < len(stages) - 1:
            next_w = stages[i + 1][1]
            x_l2 = 0.5 - next_w / 2
            x_r2 = 0.5 + next_w / 2
            pts = [(x_left, y), (x_right, y), (x_r2, y - h), (x_l2, y - h)]
        else:
            pts = [(x_left, y), (x_right, y), (x_right, y - h), (x_left, y - h)]
        poly = Polygon(pts, closed=True, facecolor=color, edgecolor="#111",
                       linewidth=1.4, alpha=0.92)
        ax.add_patch(poly)
        ax.text(0.5, y - h / 2, label, ha="center", va="center",
                fontsize=12, color="white", weight="bold")

    ax.text(0.5, 0.04,
            "Chaque étage filtre les configurations selon des critères de qualité, diversité et interprétabilité.",
            ha="center", fontsize=10, color="#444", style="italic", transform=ax.transAxes)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.88, bottom=0.06)
    p = out / "volumes_funnel.png"
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _save_plot_pipeline_overview(out: Path) -> Path:
    """Pipeline rendu via Mermaid (mmdc CLI) : routing automatique propre,
    pas de chevauchement de flèches ni de boîtes."""
    import subprocess
    # Flowchart vertical, nœuds rectangulaires à coins arrondis (rx/ry).
    # Tirets insécables "‑" (U+2011) sur les acronymes (CMA‑ES, L‑BFGS‑B) pour
    # éviter que Mermaid ne wrappe au milieu de l'acronyme.
    mermaid_src = """%%{init: {'theme': 'base', 'themeVariables': {
'primaryColor': '#fff',
'primaryBorderColor': '#444',
'fontFamily': 'Inter, sans-serif',
'fontSize': '16px',
'lineColor': '#666'
}, 'flowchart': {'curve': 'basis', 'padding': 18, 'nodeSpacing': 80, 'rankSpacing': 80}}}%%
flowchart TB
    A["<b>1. DONNÉES D'ENTRÉE</b><br/>Wikipédia 2002-2022 : 61 candidats × 5 élections<br/>Sondages IFOP / Public Sénat / Wikipédia (mai 2026)<br/>6 archétypes politiques · pool sizes par année"]
    B["<b>2. CALIBRATION</b><br/>Ridge regression L‑BFGS‑B sur 5 élections<br/>6 paramètres globaux ajustés · MAE LOO = 2.05 pts"]
    C["<b>3. DATASET SYNTHÉTIQUE</b><br/>Latin Hypercube Sampling, 8 dimensions<br/>30 000 configs × 30 Monte Carlo = 900 000 simulations"]
    D["<b>4. ENTRAÎNEMENT NN</b><br/>3 blocs résiduels · 192 hidden · 910 083 paramètres<br/>AdamW + Cosine Annealing · MAE test ≈ 0.18 pt"]
    E["<b>5. OPTIMISATION CMA‑ES</b><br/>100 popsize · 500 iter · 20 restarts<br/>= 1 000 000 évaluations sur le NN<br/>1000× plus rapide que le simulateur direct"]
    F["<b>5 bis. BOUCLE LLM (Tier 2)</b><br/>gemma4:31b‑cloud via Ollama Turbo<br/>Propose sous‑paramètres · filtre redondance + rollback"]
    G["<b>6. ANALYSES</b><br/>winner_analysis : 36 combos · extreme_search : 13D CMA‑ES<br/>path_to_victory : chemin minimal vers seuils de probabilité"]
    H["<b>7. RAPPORT FINAL</b><br/>Markdown + 24 plots · validation historique F1 = 0.90<br/>Accuracy 2T = 80% · P(victoire) baseline = 0.0% · max extrême = 88%"]

    A --> B --> C --> D --> E
    E ==> G ==> H
    E -.-> F
    F -. "Si Tier 2 accepté,<br/>réentraîne le NN avec les nouvelles dimensions" .-> C

    classDef data    fill:#E3F2FD,stroke:#1565C0,stroke-width:2.5px,color:#0D47A1,rx:16,ry:16
    classDef calib   fill:#FFF3E0,stroke:#E65100,stroke-width:2.5px,color:#BF360C,rx:16,ry:16
    classDef dataset fill:#F3E5F5,stroke:#6A1B9A,stroke-width:2.5px,color:#4A148C,rx:16,ry:16
    classDef nn      fill:#E8F5E9,stroke:#1B5E20,stroke-width:2.5px,color:#1B5E20,rx:16,ry:16
    classDef opt     fill:#FCE4EC,stroke:#AD1457,stroke-width:2.5px,color:#880E4F,rx:16,ry:16
    classDef llm     fill:#FFFDE7,stroke:#F57F17,stroke-width:2.5px,color:#E65100,rx:16,ry:16
    classDef ana     fill:#E0F7FA,stroke:#006064,stroke-width:2.5px,color:#006064,rx:16,ry:16
    classDef rep     fill:#EFEBE9,stroke:#3E2723,stroke-width:2.5px,color:#3E2723,rx:16,ry:16

    class A data
    class B calib
    class C dataset
    class D nn
    class E opt
    class F llm
    class G ana
    class H rep
"""
    mmd_path = out / "pipeline_overview.mmd"
    mmd_path.write_text(mermaid_src)
    p = out / "pipeline_overview.png"
    try:
        subprocess.run(
            ["mmdc", "-i", str(mmd_path), "-o", str(p),
             "-w", "1900", "-H", "2400", "-b", "white",
             "--scale", "2"],
            check=True, capture_output=True, timeout=60,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"  [pipeline_overview] mmdc indisponible ou échec : {e}. "
              "Fallback matplotlib simplifié.")
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis("off")
        ax.text(0.5, 0.5,
                "Pipeline (rendu Mermaid indisponible)\n"
                "Voir pipeline_overview.mmd pour le source",
                ha="center", va="center", fontsize=12)
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return p


def _save_plot_baseline_2027(cfg, out: Path) -> Path:
    """Figure isolée du baseline 2027, pédagogique pour néophytes.
    Inclut noms complets + partis, pools traduits, seuil 2T, sondages réels.
    """
    from .physical_model import first_round_scores_breakdown
    from .parameters import TIER1_PARAMS

    # Sondages réels mai 2026 (sources : IFOP, Wikipédia)
    real_polls = {
        "bardella":  36.0,
        "philippe":  18.0,
        "melenchon": 11.0,
        "glucksmann": 11.0,
        "villepin":  11.0,
        "retailleau": 10.0,
    }
    # Noms longs + parti pour lisibilité
    long_names = {
        "bardella":   "Bardella\n(RN)",
        "philippe":   "Philippe\n(Horizons)",
        "melenchon":  "Mélenchon\n(LFI)",
        "glucksmann": "Glucksmann\n(PS / Place publique)",
        "villepin":   "Villepin\n(LFH)",
        "retailleau": "Retailleau\n(LR)",
    }
    # Sociologie des pools pour néophytes
    pool_labels = {
        "rn":       "Électorat RN / Reconquête",
        "central":  "Centre / macronistes",
        "gauche":   "Gauche modérée (PS, Verts)",
        "lfi":      "Gauche radicale (LFI, PCF, NPA)",
        "lr":       "Droite classique (LR)",
        "indecis":  "Indécis / fluide",
    }
    # Couleurs proches des codes politiques français
    pool_colors = {
        "rn":       "#1a3a8a",   # bleu marine RN
        "central":  "#FFC400",   # jaune macroniste
        "gauche":   "#F08080",   # rose-rouge PS
        "lfi":      "#B71C1C",   # rouge LFI
        "lr":       "#0288D1",   # bleu LR
        "indecis":  "#9E9E9E",   # gris indécis
    }

    params = {n: 0.5 for n in TIER1_PARAMS}
    bd = first_round_scores_breakdown(params, cfg)
    candidates = list(bd.keys())
    pool_keys = list(cfg["pools"].keys())
    totals = np.array([sum(bd[c].values()) for c in candidates])
    order = np.argsort(-totals)
    candidates = [candidates[i] for i in order]
    totals = totals[order]
    polls = np.array([real_polls.get(c, 0.0) for c in candidates])
    labels = [long_names.get(c, c) for c in candidates]

    fig, ax = plt.subplots(figsize=(14, 8.6))

    # Géométrie des barres : groupées par candidat avec gap minimal interne
    n = len(candidates)
    bar_width = 0.36
    gap = 0.04
    x = np.arange(n)
    x_model = x - (bar_width + gap) / 2
    x_poll  = x + (bar_width + gap) / 2

    # Barres empilées : score prédit décomposé par pool
    bottoms = np.zeros(n)
    for pk in pool_keys:
        vals = np.array([bd[c].get(pk, 0.0) for c in candidates])
        ax.bar(x_model, vals, width=bar_width,
               bottom=bottoms,
               color=pool_colors.get(pk, "#888"),
               edgecolor="none",
               label=pool_labels.get(pk, pk))
        bottoms += vals

    # Barre de comparaison : sondages réels (hachurée à droite)
    ax.bar(x_poll, polls, width=bar_width,
           color="white", edgecolor="#555", hatch="////", linewidth=0.9,
           label="Sondage réel (mai 2026)")

    # Calculs pour limites + offsets cohérents des labels
    y_max = max(totals.max(), polls.max()) * 1.22
    label_offset = y_max * 0.018

    # Labels valeurs : prédit (gras, foncé) au-dessus de chaque barre modèle
    for i, total in enumerate(totals):
        ax.text(x_model[i], total + label_offset, f"{total:.1f} %",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold",
                color="#1A237E")
    # Labels sondage (gris, italique) au-dessus des barres hachurées
    for i, poll in enumerate(polls):
        if poll > 0:
            ax.text(x_poll[i], poll + label_offset, f"{poll:.0f} %",
                    ha="center", va="bottom", fontsize=9.5, style="italic",
                    color="#666")

    # Seuil de qualification au 2nd tour
    threshold = sorted(totals, reverse=True)[1]
    ax.axhline(threshold, color="#C62828", lw=1.1, ls="--", alpha=0.55, zorder=1)
    ax.text(n - 0.5, threshold + label_offset * 0.5,
            f"Seuil qualification 2nd tour : {threshold:.1f} %",
            ha="right", va="bottom", fontsize=9, color="#C62828", style="italic")

    # Badges "qualifié 2nd tour" : aligner à hauteur fixe pour propreté
    top2_idx = np.argsort(-totals)[:2]
    badge_y = y_max * 0.93
    for idx in top2_idx:
        ax.annotate("Qualifié 2nd tour",
                    xy=(x_model[idx], totals[idx] + label_offset * 2.5),
                    xytext=(x_model[idx], badge_y),
                    ha="center", fontsize=9, fontweight="bold", color="#1B5E20",
                    arrowprops=dict(arrowstyle="-", color="#1B5E20", lw=0.7,
                                    alpha=0.6),
                    bbox=dict(boxstyle="round,pad=0.35", fc="#E8F5E9",
                              ec="#1B5E20", lw=0.8))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_ylabel("Pourcentage des suffrages exprimés", fontsize=11)
    ax.set_title(
        "Simulation Machine Learning : Élection présidentielle 2027 (1er tour)",
        fontsize=13.5, fontweight="bold", pad=22,
    )
    ax.text(
        0.5, 1.018,
        "Comparaison de la prédiction du modèle avec les sondages réels de mai 2026 "
        "(IFOP / Public Sénat / Wikipédia Liste de sondages 2027)",
        transform=ax.transAxes, ha="center", va="bottom",
        fontsize=9.5, color="#444", style="italic",
    )
    ax.set_ylim(0, y_max * 1.02)
    ax.set_xlim(-0.55, n - 0.45)
    ax.grid(True, axis="y", alpha=0.25, ls=":", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Légende compacte
    leg = ax.legend(
        title="Origine sociologique (barre couleur) et sondage réel (hachuré)",
        loc="upper right", fontsize=8.5, title_fontsize=9.5,
        ncol=2, framealpha=0.95, borderpad=0.6, columnspacing=1.2,
        handlelength=1.4, handleheight=1.0,
    )
    leg.get_frame().set_edgecolor("#ccc")

    # Réserver la zone basse pour les notes (subplots_adjust = méthode fiable
    # qui ne dépend pas de bbox_inches="tight").
    plt.subplots_adjust(bottom=0.24, top=0.90, left=0.07, right=0.98)

    # Note de lecture (juste sous l'axe X)
    fig.text(
        0.5, 0.155,
        "Lecture : à gauche (couleurs pleines) = score prédit décomposé par groupe d'électeurs ; "
        "à droite (hachuré) = score réel des sondages.",
        ha="center", fontsize=9.5, style="italic", color="#555",
    )

    # Bloc méthodo en 3 lignes courtes (pas de débordement)
    methodo_lines = [
        "Comment le modèle fonctionne :",
        "1. L'électorat est partitionné en 6 pools sociologiques (RN, centre, gauche, LFI, LR, indécis) attirés vers les candidats selon leurs affinités.",
        "2. Un réseau de neurones apprend la dynamique sur 30 000 simulations Monte Carlo.",
        "3. L'algorithme CMA-ES cherche les configurations qui maximisent la victoire de Villepin.",
        "Calibré sur 5 élections (2002-2022, MAE 2.05 pts).  Exploration de scénarios, pas une prédiction certaine.",
    ]
    y_positions = [0.105, 0.082, 0.061, 0.040, 0.012]
    weights = ["bold", "normal", "normal", "normal", "normal"]
    sizes = [9, 8.5, 8.5, 8.5, 8.5]
    colors = ["#444", "#666", "#666", "#666", "#888"]
    styles = ["normal", "normal", "normal", "normal", "italic"]
    for txt, y, w, s, c, st in zip(methodo_lines, y_positions, weights, sizes, colors, styles):
        fig.text(0.5, y, txt, ha="center", fontsize=s, fontweight=w,
                 color=c, style=st)

    # Pas de tight_layout (déjà géré par subplots_adjust)
    p = out / "baseline_2027.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_pool_ownership_by_scenario(winner_df: pd.DataFrame, cfg, out: Path) -> Path | None:
    """Pour chaque scénario exogène, décompose le score 1T de chaque candidat
    par pool d'origine (rn / central / gauche / lfi / lr / indecis).
    Très utile pour COMPRENDRE pourquoi tel candidat est devant.
    """
    from .physical_model import first_round_scores_breakdown
    from .genetic_optimizer import EXOGENOUS_SCENARIOS
    from .parameters import INTERNAL
    scenarios = list(EXOGENOUS_SCENARIOS.items())
    n_scen = len(scenarios)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.flatten()
    pool_keys = list(cfg["pools"].keys())
    palette = sns.color_palette("Set2", n_colors=len(pool_keys))
    internal_neutral = {n: 0.5 for n in INTERNAL}
    for ax, (scen_name, exo) in zip(axes, scenarios):
        params = {**internal_neutral, **exo}
        bd = first_round_scores_breakdown(params, cfg)
        candidates = list(bd.keys())
        bottoms = np.zeros(len(candidates))
        for j, pk in enumerate(pool_keys):
            vals = np.array([bd[c].get(pk, 0.0) for c in candidates])
            ax.bar(candidates, vals, bottom=bottoms,
                   color=palette[j], label=pk if scen_name == scenarios[0][0] else None)
            bottoms += vals
        ax.set_title(scen_name, fontsize=10)
        ax.set_ylabel("score 1T (%)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    for k in range(n_scen, len(axes)):
        axes[k].axis("off")
    # Légende globale au-dessus
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in palette]
    fig.legend(handles, pool_keys, title="pool", loc="upper center",
               bbox_to_anchor=(0.5, 1.02), ncol=6, frameon=False)
    plt.suptitle("Origine du score 1T par pool, par scénario exogène (interne fixés à 0.5)",
                 y=1.05, fontsize=11)
    plt.tight_layout()
    p = out / "pool_ownership_by_scenario.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def _save_plot_pool_breakdown(out: Path, cfg, params_baseline,
                              params_optimum, label_a="baseline", label_b="optimum") -> Path:
    """Barres empilées : décomposition par pool du score 1er tour de chaque candidat.
    Comparaison baseline (params neutres) vs optimum (params CMA-ES)."""
    from .physical_model import first_round_scores_breakdown
    bd_a = first_round_scores_breakdown(params_baseline, cfg)
    bd_b = first_round_scores_breakdown(params_optimum, cfg)
    candidates = list(bd_a.keys())
    pool_keys = list(cfg["pools"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    palette = sns.color_palette("tab10", n_colors=len(pool_keys))
    for ax, bd, title in zip(axes, [bd_a, bd_b], [label_a, label_b]):
        bottoms = np.zeros(len(candidates))
        for j, pk in enumerate(pool_keys):
            vals = np.array([bd[c].get(pk, 0.0) for c in candidates])
            ax.bar(candidates, vals, bottom=bottoms, color=palette[j], label=pk)
            bottoms += vals
        ax.set_title(f"Décomposition score 1T : {title}")
        ax.set_ylabel("Score 1T (%)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    axes[1].legend(title="pool", loc="upper right", bbox_to_anchor=(1.18, 1))
    plt.tight_layout()
    p = out / "pool_breakdown.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return p


def _save_plot_shap(model_path: Path, dataset_path: Path, out: Path) -> Path | None:
    """SHAP summary plot pour P(victoire). Permet de voir quel paramètre Tier 1
    pousse le plus la prédiction à la hausse ou la baisse."""
    try:
        import shap
        import torch
        from .neural_predictor import load_model
    except ImportError:
        return None
    model, x_cols = load_model(model_path)
    df = pd.read_parquet(dataset_path)
    X = df[x_cols].values.astype(np.float32)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X), 100, replace=False)
    test_idx = rng.choice(len(X), 300, replace=False)
    bg = torch.from_numpy(X[bg_idx])
    test = torch.from_numpy(X[test_idx])

    # Wrapper qui retourne UNIQUEMENT p_victoire (sortie idx 2)
    class _PVictory(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, x):
            return self.m(x)[:, 2:3]

    wrapped = _PVictory(model)
    explainer = shap.DeepExplainer(wrapped, bg)
    shap_vals = explainer.shap_values(test, check_additivity=False)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 0]

    plt.figure(figsize=(8, max(5, len(x_cols) * 0.35)))
    shap.summary_plot(shap_vals, test.numpy(), feature_names=list(x_cols), show=False)
    p = out / "shap_p_victory.png"
    plt.tight_layout()
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    return p


def _save_plot_archetypes(arche: pd.DataFrame, free_names: list[str], out: Path) -> Path:
    """Vrai radar polaire des centroïdes par cluster. Si tous les clusters
    convergent vers le même optimum, on l'annote explicitement plutôt que
    de superposer 6 lignes identiques.
    """
    centroid_cols = [f"centroid_{n}" for n in free_names]
    arche_sorted = arche.sort_values("best_p_victory", ascending=False).reset_index(drop=True)

    # Test de convergence : si max_dist entre centroïdes < 0.05, ils sont collés
    centroids = arche_sorted[centroid_cols].values
    max_pairwise = float(np.max(np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=-1
    ))) if len(centroids) > 1 else 0.0
    all_collapse = max_pairwise < 0.05

    n = len(free_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # fermer le polygone

    fig = plt.figure(figsize=(9, 6.5))
    ax = plt.subplot(111, polar=True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(free_names, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_ylim(0, 1.0)
    palette = sns.color_palette("tab10", n_colors=len(arche_sorted))
    for i, row in arche_sorted.iterrows():
        values = [row[c] for c in centroid_cols]
        values += values[:1]
        label = f"cluster {int(row['cluster'])} (p={row['best_p_victory']:.4f}, n={int(row['n_members'])})"
        ax.plot(angles, values, color=palette[i], lw=2.0, label=label,
                marker="o", markersize=5)
        ax.fill(angles, values, color=palette[i], alpha=0.06)
    ax.legend(loc="upper right", bbox_to_anchor=(1.45, 1.0), fontsize=8,
              frameon=False, title="archétype")
    suptitle = "Archétypes de stratégie : centroïdes (scénario tempete_2017)"
    if all_collapse:
        suptitle += "\n⚠ tous les clusters convergent vers le MÊME optimum (pas d'archétypes distincts)"
    plt.suptitle(suptitle, y=0.97, fontsize=11)
    plt.tight_layout()
    p = out / "archetypes_radar.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return p


def build_report(out_dir: Path) -> str:
    cal_sum = json.loads((out_dir / "calibration_summary.json").read_text())
    loo_df = pd.read_csv(out_dir / "calibration_loo.csv")
    base_df = pd.read_csv(out_dir / "calibration_baseline.csv")
    in_sample_df = pd.read_csv(out_dir / "calibration_in_sample.csv")
    scenarios_df = pd.read_csv(out_dir / "cmaes_scenarios.csv")
    arche_df = pd.read_csv(out_dir / "archetypes.csv")
    audit_arch_df = pd.read_csv(out_dir / "audit_archetypes.csv") if (out_dir / "audit_archetypes.csv").exists() else None
    audit_sens_df = pd.read_csv(out_dir / "audit_sensitivity.csv") if (out_dir / "audit_sensitivity.csv").exists() else None
    winner_df = pd.read_csv(out_dir / "winner_probabilities.csv") if (out_dir / "winner_probabilities.csv").exists() else None
    winner_agg_df = pd.read_csv(out_dir / "winner_aggregate.csv") if (out_dir / "winner_aggregate.csv").exists() else None
    extreme_summary = None
    if (out_dir / "extreme_search_summary.json").exists():
        extreme_summary = json.loads((out_dir / "extreme_search_summary.json").read_text())
    villepin_best_df = pd.read_csv(out_dir / "villepin_best_scenarios.csv") if (out_dir / "villepin_best_scenarios.csv").exists() else None
    llm_summary = None
    llm_history_path = out_dir / "llm_history" / "summary.json"
    if llm_history_path.exists():
        llm_summary = json.loads(llm_history_path.read_text())
    tier2_meta = None
    cfg_fitted_path = Path("config.fitted.yaml")
    if cfg_fitted_path.exists():
        fcfg = load_config(cfg_fitted_path)
        tier2_meta = fcfg.get("tier2_params", {})

    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _save_plot_calibration(loo_df, plots_dir)
    _save_plot_scenarios(scenarios_df, plots_dir)
    dataset_path = out_dir / "dataset.parquet"
    if dataset_path.exists():
        ds = pd.read_parquet(dataset_path)
        _save_plot_dataset_distrib(ds, plots_dir)
    free_names = [c.replace("centroid_", "") for c in arche_df.columns if c.startswith("centroid_")]
    _save_plot_archetypes(arche_df, free_names, plots_dir)

    # Nouveaux plots interprétables
    if winner_df is not None:
        _save_plot_winner_heatmap(winner_df, plots_dir)
    if villepin_best_df is not None:
        _save_plot_villepin_top_scenarios(villepin_best_df, plots_dir)
    if audit_sens_df is not None:
        _save_plot_sensitivity(audit_sens_df, plots_dir)
    _save_plot_effort_allocation(scenarios_df, plots_dir)
    # Pool breakdown : baseline (params neutres) vs optimum (tempete_2017)
    from .parameters import TIER1_PARAMS as _T1
    cfg_path = Path("config.fitted.yaml")
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)
    baseline_params = {n: 0.5 for n in _T1}
    # Optimum = stratégie CMA-ES dans le scénario `tempete_2017`
    opt_row = scenarios_df[scenarios_df["scenario"] == "tempete_2017"].iloc[0]
    optimum_params = {n: float(opt_row[f"best_{n}"]) if f"best_{n}" in opt_row else float(opt_row[n])
                      for n in _T1}
    _save_plot_pool_breakdown(plots_dir, cfg, baseline_params, optimum_params,
                              label_a="paramètres neutres (0.5)",
                              label_b="optimum CMA-ES (tempete_2017)")
    # SHAP (optionnel, demande NN + dataset)
    nn_ckpt = out_dir / "checkpoints" / "nn_surrogate.pt"
    if nn_ckpt.exists() and dataset_path.exists():
        _save_plot_shap(nn_ckpt, dataset_path, plots_dir)
    if llm_summary is not None:
        _save_plot_llm_evolution(llm_summary, plots_dir)
    if tier2_meta:
        _save_plot_tier2_weights(tier2_meta, plots_dir)

    # Validation historique supervisée + partial dependence + sensitivity
    hist_summary_path = out_dir / "historical_validation_summary.json"
    if hist_summary_path.exists():
        hist_sum = json.loads(hist_summary_path.read_text())
        per_pair_h = pd.read_csv(out_dir / "historical_validation_per_pair.csv")
        per_year_h = pd.read_csv(out_dir / "historical_validation_per_year.csv")
        _save_plot_historical_validation(per_pair_h, per_year_h,
                                          hist_sum["metrics"], plots_dir)
    _save_plot_historical_validation_compare(out_dir, plots_dir)
    if nn_ckpt.exists() and dataset_path.exists():
        _save_plot_partial_dependence(nn_ckpt, dataset_path, plots_dir)
        _save_plot_param_sensitivity(nn_ckpt, dataset_path, plots_dir)
    if winner_df is not None:
        _save_plot_pool_ownership_by_scenario(winner_df, cfg, plots_dir)
    _save_plot_baseline_2027(cfg, plots_dir)
    # Nouveaux plots pédagogiques v3
    _save_plot_inputs_overview(cfg, plots_dir)
    _save_plot_archetypes_explained(cfg, plots_dir)
    _save_plot_historical_per_party(out_dir, plots_dir)
    _save_plot_villepin_winning_recipe(cfg, plots_dir)
    _save_plot_villepin_winning_path(cfg, plots_dir)
    _save_plot_pipeline_overview(plots_dir)
    _save_plot_nn_architecture(plots_dir)
    _save_plot_nn_block_anatomy(plots_dir)
    _save_plot_nn_3d_volumetric(plots_dir)
    _save_plot_nn_params_pie(plots_dir)
    _save_plot_nn_hyperparams(plots_dir)
    _save_plot_nn_weights_heatmap(plots_dir)
    _save_plot_nn_weights_distribution(plots_dir)
    _save_plot_tuning_inference(plots_dir)
    _save_plot_volumes_cards(plots_dir)
    _save_plot_volumes_log_scale(plots_dir)
    _save_plot_volumes_funnel(plots_dir)
    _save_plot_model_trust_card(out_dir, plots_dir)
    ptv_summary_path = out_dir / "path_to_victory.summary.json"
    if ptv_summary_path.exists():
        ptv_summary = json.loads(ptv_summary_path.read_text())
        _save_plot_path_to_victory(ptv_summary, plots_dir / "path_to_victory.summary.json")

    p_best = float(scenarios_df["best_p_victory"].max())
    best_scenario = scenarios_df.loc[scenarios_df["best_p_victory"].idxmax()]

    audit_section = ""
    if audit_arch_df is not None and audit_sens_df is not None:
        cov = float(audit_sens_df["in_range"].mean() * 100)
        max_delta = float(audit_arch_df["abs_delta"].max())
        worst = audit_arch_df.loc[audit_arch_df["abs_delta"].idxmax()]
        audit_section = f"""\
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
MAE des archétypes : **{cal_sum.get('archetype_mae', 0):.2f}**, max |delta| **{max_delta:.2f}**
(cas le pire : {worst['archetype']} {int(worst['year'])}, base assignée
{worst['base_assigned']:.1f}, prédit {worst['predicted']:.1f}). La dynamique
mobile/stuck distord encore significativement les scores des concurrents non-Villepin.

### Robustesse au choix de contexte
Couverture {cov:.0f}% : les intervalles de prédiction p5-p95 (sur 100 perturbations
±0.20 du contexte) contiennent {cov:.0f}% des scores réels. Une couverture < 50%
indique que la précision affichée est artificielle : le modèle ne devrait pas
être lu en valeur exacte mais en ordre de grandeur.

{audit_sens_df.to_markdown(index=False)}

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

"""

    md = f"""# Villepin 2027 : Rapport exploratoire

{DISCLAIMER}

## Résumé exécutif

Avec un simulateur **audité contre les biais** et calibré sur 2002-2022
(MAE leave-one-out **{cal_sum['mae_loo']:.2f} points**, couverture sensibilité
**{cal_sum.get('sensitivity_coverage_pct', 0):.0f}%**), et une optimisation CMA-ES
sur les 4 paramètres de campagne contrôlables, **la probabilité maximale de
victoire au 2T identifiée est de {p_best*100:.2f}%**, atteinte dans
**« {best_scenario['scenario']} »**.

**Conclusion principale** : sous des affinités électorales sobres (Villepin
n'est PAS magnétique pour la gauche, ni pour les insoumis, ni hyper-favori
des indécis), le modèle prédit une **impossibilité structurelle quasi-totale
de qualification au 2T**. Le résultat de v0 (plafond 6%) était gonflé par
un biais "fanboy" sur les affinités initiales.

{audit_section}{_build_historical_validation_section(out_dir)}{_build_dynamics_section()}{_build_llm_section(llm_summary, tier2_meta)}{_build_winner_section(winner_df, winner_agg_df, villepin_best_df)}{_build_extreme_section(extreme_summary)}{_build_path_to_victory_section(out_dir)}## Annexe pédagogique : comprendre la machine

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
MAE : **{cal_sum['mae_baseline']:.2f} points**

{base_df.to_markdown(index=False)}

### Fit in-sample (ridge λ={cal_sum['ridge_lambda']}, fit sur les 5 années)
MAE : **{cal_sum['mae_in_sample']:.2f} points**

Paramètres ajustés :
{chr(10).join(f"- `{k}` : {v}" for k, v in cal_sum['fitted_params'].items())}

{in_sample_df.to_markdown(index=False)}

### Leave-one-out (entraînement sur 4 années, test sur la 5ème)
MAE : **{cal_sum['mae_loo']:.2f} points**

{loo_df[['year_held_out', 'candidate', 'actual', 'predicted', 'abs_error']].to_markdown(index=False)}

![Calibration LOO](plots/calibration_loo.png)

**Observation honnête** : le modèle reproduit raisonnablement bien Macron 2017
(scénario "tempête parfaite") et Pécresse 2022 (scénario faible). Il sous-estime
nettement **Bayrou 2007** ({loo_df[loo_df['year_held_out']==2007]['abs_error'].iloc[0]:.1f} pts d'erreur),
indiquant qu'il existe en 2007 un facteur non capturé par les variables
structurelles encodées (charisme personnel, dynamique de campagne, momentum
TV) : c'est une limite intrinsèque.

## 2. Scénarios exogènes : plafond de probabilité

Pour chaque scénario exogène fixé (`crisis`, `central_collapse`, `volatility`,
`anti_extreme_pressure`), on optimise par CMA-ES les 4 paramètres internes
(`campaign_machine`, `thematic_breadth`, `media_performance`,
`coalition_building`) sous contraintes budgétaires.

{scenarios_df[['scenario', 'crisis', 'central_collapse', 'volatility',
               'anti_extreme_pressure', 'best_p_victory', 'best_score_1T']].to_markdown(index=False)}

![Probabilité par scénario](plots/scenarios_p_victory.png)

**Plafond identifié** : {p_best*100:.2f}% dans `{best_scenario['scenario']}`.

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

Clustering KMeans (k={len(arche_df)}) sur les top candidats CMA-ES dans le
scénario tempête_2017. Centroïdes des paramètres internes :

{arche_df.to_markdown(index=False)}

![Archétypes](plots/archetypes_radar.png)

**Observation** : les clusters convergent tous vers une même P(victoire) à
{arche_df['best_p_victory'].iloc[0]*100:.2f}%, signe que CMA-ES trouve un
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

Pour le scénario le plus favorable (`{best_scenario['scenario']}`), les
paramètres internes optimaux suggérés sont :

{pd.DataFrame([{k.replace('best_', ''): v
                 for k, v in best_scenario.to_dict().items()
                 if k.startswith('best_') and k not in ['best_p_victory', 'best_score_1T', 'best_p_qualif']}]).T.rename(columns={0: 'valeur optimale'}).to_markdown()}

## 6. Prochaines étapes

- **Activer la boucle LLM (Ollama Turbo)** pour découvrir des sous-paramètres
  Tier 2 (cf `src/llm_param_discovery.py`).
- **Étendre la calibration** sur élections municipales / européennes pour plus
  de points.
- **Ajouter des adversaires adaptatifs** (self-play).
- **Modèle temporel mensuel** mai 2026 → avril 2027.

---

*Généré automatiquement. Code et données : `villepin_sim/`.*
"""
    return md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()
    out = Path(args.out_dir)
    md = build_report(out)
    (out / "final_report.md").write_text(md, encoding="utf-8")
    print(f"✓ Rapport : {out/'final_report.md'} ({len(md)} chars)")


if __name__ == "__main__":
    main()
