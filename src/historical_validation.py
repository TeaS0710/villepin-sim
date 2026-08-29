"""Validation supervisée du modèle sur les 5 élections historiques.

Pour chaque élection :
1. Configure le simulateur avec :
   - bases concurrents = scores réels agrégés par archétype
   - CONTEXT[year] = paramètres Tier 1 estimés
2. Lance `first_round_scores` -> 6 scores prédits (villepin + 5 archétypes)
3. Identifie le top-2 prédit ; compare au top-2 réel
4. Lance `all_candidates_2T_probabilities` -> P(victoire) par candidat top-2
5. Identifie le vainqueur prédit ; compare au vainqueur réel

Sortie : matrice de confusion + precision/recall/F1 sur la classification
binaire "qualifié au 2T", + accuracy sur la prédiction du vainqueur 2T.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from .calibration import build_year_config
from .historical_context import (
    ARCHETYPE_MAPPING, CONTEXT, VILLEPIN_EQUIVALENT,
    aggregate_competitor_bases,
)
from .physical_model import (
    all_candidates_2T_probabilities, first_round_scores, load_config,
)

# Résultats officiels du 2nd tour 2002-2022 (vainqueur, perdant, %s)
# https://www.interieur.gouv.fr/Elections/Les-resultats/Presidentielles
SECOND_ROUND_ACTUAL: dict[int, dict] = {
    # 2002 : Chirac (RPR) vs Le Pen (FN) → Chirac écrasement 82-18
    # Mapping rigoureux : RPR = retailleau (LR/UMP ancestor), pas philippe.
    2002: {"winner": "retailleau", "winner_pct": 82.21,
           "loser":  "bardella",   "loser_pct":  17.79,
           "winner_actual_name": "Jacques Chirac",
           "loser_actual_name":  "Jean-Marie Le Pen"},
    # 2007 : Sarkozy (UMP) vs Royal (PS) → Sarkozy 53-47
    # UMP = retailleau (LR ancestor), pas philippe.
    2007: {"winner": "retailleau", "winner_pct": 53.06,
           "loser":  "glucksmann", "loser_pct":  46.94,
           "winner_actual_name": "Nicolas Sarkozy",
           "loser_actual_name":  "Ségolène Royal"},
    # 2012 : Hollande (PS) vs Sarkozy (UMP) → Hollande 51.6-48.4
    2012: {"winner": "glucksmann", "winner_pct": 51.64,
           "loser":  "retailleau", "loser_pct":  48.36,
           "winner_actual_name": "François Hollande",
           "loser_actual_name":  "Nicolas Sarkozy"},
    # 2017 : Macron (EM) vs Le Pen (FN) → Macron 66-34
    # Macron 2017 = villepin (centriste outsider), pas philippe (sortant).
    2017: {"winner": "villepin",   "winner_pct": 66.10,
           "loser":  "bardella",   "loser_pct":  33.90,
           "winner_actual_name": "Emmanuel Macron",
           "loser_actual_name":  "Marine Le Pen"},
    # 2022 : Macron (LREM sortant) vs Le Pen (RN) → Macron 58-42
    # Macron 2022 = philippe (centre sortant), différent de 2017.
    2022: {"winner": "philippe",   "winner_pct": 58.55,
           "loser":  "bardella",   "loser_pct":  41.45,
           "winner_actual_name": "Emmanuel Macron",
           "loser_actual_name":  "Marine Le Pen"},
}


def actual_top2(year: int, hist_df: pd.DataFrame) -> set[str]:
    """Top 2 archétypes réels au 1T pour une année donnée.

    Stratégie : pour chaque archétype, somme des scores des candidats mappés.
    Top-2 = les 2 plus gros (incluant le Villepin-équivalent comme `villepin`).
    """
    mapping = ARCHETYPE_MAPPING[year]
    villepin_name = VILLEPIN_EQUIVALENT[year]
    df = hist_df[hist_df["year"] == year]

    arch_scores: dict[str, float] = {}
    for arch, candidates in mapping.items():
        total = sum(float(df[df["candidate"] == c]["pct_exprimes"].iloc[0])
                    for c in candidates
                    if c in df["candidate"].values)
        arch_scores[arch] = total
    # Score Villepin-équivalent
    villepin_score = float(df[df["candidate"] == villepin_name]["pct_exprimes"].iloc[0])
    arch_scores["villepin"] = villepin_score

    top2 = sorted(arch_scores.items(), key=lambda kv: -kv[1])[:2]
    return {name for name, _ in top2}, arch_scores


def predict_for_year(year: int, base_cfg: dict, hist_df: pd.DataFrame) -> dict:
    """Prédiction du modèle pour une année historique : scores 1T + top-2 + vainqueur 2T."""
    cfg_y = build_year_config(base_cfg, year, hist_df)
    params = dict(CONTEXT[year])
    scores = first_round_scores(params, cfg_y)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    pred_top2 = {ranked[0][0], ranked[1][0]}

    probs = all_candidates_2T_probabilities(params, cfg_y)
    qualified = [n for n, d in probs.items() if d["qualified"]]
    if len(qualified) == 2:
        a, b = qualified
        if probs[a]["p_victory"] > probs[b]["p_victory"]:
            pred_winner = a
        else:
            pred_winner = b
    else:
        pred_winner = ranked[0][0]
    return {
        "year": year,
        "scores": scores,
        "predicted_top2": pred_top2,
        "predicted_winner": pred_winner,
        "all_probs": probs,
    }


def evaluate(base_cfg: dict, hist_df: pd.DataFrame) -> dict:
    """Lance la validation sur les 5 élections. Retourne dict avec :
    - per_year_details
    - confusion matrix qualified vs not
    - precision/recall/F1 par archétype et global
    - 2T winner accuracy
    """
    candidates_set = ("villepin", "bardella", "philippe", "melenchon", "retailleau", "glucksmann")
    y_true_q, y_pred_q = [], []
    per_year_rows = []
    per_pair_rows = []
    winner_correct = []

    for year in CONTEXT:
        pred = predict_for_year(year, base_cfg, hist_df)
        actual_top2_set, actual_arch_scores = actual_top2(year, hist_df)
        winner_real = SECOND_ROUND_ACTUAL[year]["winner"]

        # Per-archetype classification (qualified ?)
        for arch in candidates_set:
            actual = arch in actual_top2_set
            pred_q = arch in pred["predicted_top2"]
            y_true_q.append(int(actual))
            y_pred_q.append(int(pred_q))
            per_pair_rows.append({
                "year": year,
                "archetype": arch,
                "predicted_score_1T": round(pred["scores"].get(arch, 0.0), 2),
                "actual_score_1T":    round(actual_arch_scores.get(arch, 0.0), 2),
                "predicted_qualified": pred_q,
                "actual_qualified":    actual,
                "match": pred_q == actual,
            })

        winner_match = pred["predicted_winner"] == winner_real
        winner_correct.append(int(winner_match))
        per_year_rows.append({
            "year": year,
            "predicted_top2":      ", ".join(sorted(pred["predicted_top2"])),
            "actual_top2":         ", ".join(sorted(actual_top2_set)),
            "top2_exact_match":    pred["predicted_top2"] == actual_top2_set,
            "top2_overlap":        len(pred["predicted_top2"] & actual_top2_set),
            "predicted_winner":    pred["predicted_winner"],
            "actual_winner":       winner_real,
            "winner_correct":      winner_match,
            "predicted_winner_real_name": SECOND_ROUND_ACTUAL[year]["winner_actual_name"]
                if winner_match else "(modèle prédit "+pred["predicted_winner"]+")",
        })

    y_true_q = np.array(y_true_q)
    y_pred_q = np.array(y_pred_q)

    # Métriques classification "qualifié au 2T"
    metrics = {
        "n_pairs": int(len(y_true_q)),
        "accuracy_qualif":   float(accuracy_score(y_true_q, y_pred_q)),
        "precision_qualif":  float(precision_score(y_true_q, y_pred_q, zero_division=0)),
        "recall_qualif":     float(recall_score(y_true_q, y_pred_q, zero_division=0)),
        "f1_qualif":         float(f1_score(y_true_q, y_pred_q, zero_division=0)),
        "winner_accuracy":   float(np.mean(winner_correct)),
        "winner_correct_count": int(sum(winner_correct)),
        "winner_total":      int(len(winner_correct)),
    }
    cm = confusion_matrix(y_true_q, y_pred_q, labels=[0, 1])
    metrics["confusion_matrix"] = {
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
    }
    return {
        "metrics": metrics,
        "per_year": pd.DataFrame(per_year_rows),
        "per_pair": pd.DataFrame(per_pair_rows),
        "classification_report": classification_report(
            y_true_q, y_pred_q, target_names=["non_qualif", "qualif"],
            output_dict=True, zero_division=0,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.fitted.yaml")
    ap.add_argument("--data", default="data/historical_elections.csv")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)
    hist = pd.read_csv(args.data)
    result = evaluate(cfg, hist)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["per_year"].to_csv(out / "historical_validation_per_year.csv", index=False)
    result["per_pair"].to_csv(out / "historical_validation_per_pair.csv", index=False)
    with open(out / "historical_validation_summary.json", "w") as f:
        json.dump({"metrics": result["metrics"],
                   "classification_report": result["classification_report"]},
                  f, indent=2, ensure_ascii=False)

    print("=== Métriques classification 'qualifié au 2T' ===")
    m = result["metrics"]
    print(f"  N pairs (year × archétype) : {m['n_pairs']}")
    print(f"  Accuracy        : {m['accuracy_qualif']:.3f}")
    print(f"  Precision       : {m['precision_qualif']:.3f}")
    print(f"  Recall          : {m['recall_qualif']:.3f}")
    print(f"  F1              : {m['f1_qualif']:.3f}")
    print(f"  Confusion matrix : TP={m['confusion_matrix']['tp']}  FN={m['confusion_matrix']['fn']}  "
          f"FP={m['confusion_matrix']['fp']}  TN={m['confusion_matrix']['tn']}")
    print(f"\n=== Accuracy prédiction du vainqueur 2T ===")
    print(f"  {m['winner_correct_count']}/{m['winner_total']} = {m['winner_accuracy']*100:.0f}%")

    print(f"\n=== Détails par année ===")
    print(result["per_year"][["year", "actual_top2", "predicted_top2",
                              "actual_winner", "predicted_winner",
                              "winner_correct"]].to_string(index=False))
    print(f"\n✓ outputs/historical_validation_*.csv & .json")


if __name__ == "__main__":
    main()
