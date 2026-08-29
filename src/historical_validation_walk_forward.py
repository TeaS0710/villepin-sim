"""Validation walk-forward stricte : aucune information post-hoc.

Contrairement à `historical_validation.py` qui utilise CONTEXT[year] et
YEAR_POOLS[year] hard-codés (donc connaissant le résultat final), ce module :

1. Remplace YEAR_POOLS[year] par les sommes des sondages T-2 mois par pool
   (`historical_context_blind.derive_year_pools_ex_ante`).
2. Remplace CONTEXT[year] par 0.5 partout (aucune info exogène disponible
   à T-2 mois sans ajout de sources externes).
3. Remplace les `bases` concurrents par les sondages T-2 mois agrégés par
   archétype (au lieu des résultats finaux).

Sortie : `outputs/historical_validation_walk_forward_*.csv` et `.json`
avec les mêmes colonnes que la validation standard, mais avec le suffixe
qui les distingue clairement comme "honnête" / ex-ante.
"""
from __future__ import annotations

import argparse
import copy
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)

from .historical_context import ARCHETYPE_MAPPING, VILLEPIN_EQUIVALENT
from .historical_context_blind import (
    all_context_ex_ante, all_year_pools_ex_ante,
    load_polls_T2, _norm, _strip_party_suffix,
)
from .historical_validation import (
    SECOND_ROUND_ACTUAL, actual_top2,
)
from .physical_model import (
    all_candidates_2T_probabilities, first_round_scores, load_config,
)


def aggregate_competitor_bases_ex_ante(
    year: int, polls_df: pd.DataFrame
) -> dict[str, float]:
    """Bases concurrents pour une année à partir des SONDAGES T-2 mois (jamais
    des résultats finaux).
    """
    sub = polls_df[polls_df["year"] == year]
    cand_to_pct = dict(zip(sub["candidate_norm_clean"], sub["mean_pct"]))

    def _match(name: str) -> float:
        n = _norm(_strip_party_suffix(name))
        if n in cand_to_pct:
            return float(cand_to_pct[n])
        for k, pct in cand_to_pct.items():
            if n in k or k in n:
                return float(pct)
        return 0.0

    out: dict[str, float] = {}
    for arch, candidates in ARCHETYPE_MAPPING[year].items():
        total = sum(_match(c) for c in candidates)
        # plancher 0.1 pour conserver la signature numérique du modèle
        out[arch] = max(total, 0.1)
    return out


def build_year_config_ex_ante(
    base_cfg: dict, year: int, polls_df: pd.DataFrame,
    pools_ex_ante: dict, context_ex_ante: dict,
) -> dict:
    """Comme `build_year_config` mais TOUTES les features sont ex-ante T-2 mois."""
    cfg = copy.deepcopy(base_cfg)
    bases = aggregate_competitor_bases_ex_ante(year, polls_df)
    for arch, base in bases.items():
        if arch in cfg["competitors"]:
            cfg["competitors"][arch]["base"] = base
    pools = pools_ex_ante.get(year)
    if pools is not None:
        for pool_key, vals in pools.items():
            if pool_key in cfg["pools"]:
                cfg["pools"][pool_key]["size"] = vals["size"]
                cfg["pools"][pool_key]["inertia"] = vals["inertia"]
    return cfg


def predict_for_year_ex_ante(
    year: int, base_cfg: dict, polls_df: pd.DataFrame,
    pools_ex_ante: dict, context_ex_ante: dict,
) -> dict:
    cfg_y = build_year_config_ex_ante(
        base_cfg, year, polls_df, pools_ex_ante, context_ex_ante
    )
    params = dict(context_ex_ante[year])
    scores = first_round_scores(params, cfg_y)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    pred_top2 = {ranked[0][0], ranked[1][0]}

    probs = all_candidates_2T_probabilities(params, cfg_y)
    qualified = [n for n, d in probs.items() if d["qualified"]]
    if len(qualified) == 2:
        a, b = qualified
        pred_winner = a if probs[a]["p_victory"] > probs[b]["p_victory"] else b
    else:
        pred_winner = ranked[0][0]
    return {
        "year": year,
        "scores": scores,
        "predicted_top2": pred_top2,
        "predicted_winner": pred_winner,
        "all_probs": probs,
    }


def evaluate_walk_forward(
    base_cfg: dict, hist_df: pd.DataFrame, polls_df: pd.DataFrame,
) -> dict:
    pools_ex_ante = all_year_pools_ex_ante(polls_df)
    context_ex_ante = all_context_ex_ante(polls_df)

    candidates_set = ("villepin", "bardella", "philippe",
                       "melenchon", "retailleau", "glucksmann")
    y_true_q, y_pred_q = [], []
    per_year_rows = []
    per_pair_rows = []
    winner_correct = []

    for year in context_ex_ante:
        pred = predict_for_year_ex_ante(
            year, base_cfg, polls_df, pools_ex_ante, context_ex_ante
        )
        actual_top2_set, actual_arch_scores = actual_top2(year, hist_df)
        winner_real = SECOND_ROUND_ACTUAL[year]["winner"]

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
            "predicted_top2":   ", ".join(sorted(pred["predicted_top2"])),
            "actual_top2":      ", ".join(sorted(actual_top2_set)),
            "top2_exact_match": pred["predicted_top2"] == actual_top2_set,
            "top2_overlap":     len(pred["predicted_top2"] & actual_top2_set),
            "predicted_winner": pred["predicted_winner"],
            "actual_winner":    winner_real,
            "winner_correct":   winner_match,
        })

    y_true_q = np.array(y_true_q)
    y_pred_q = np.array(y_pred_q)
    cm = confusion_matrix(y_true_q, y_pred_q, labels=[0, 1])
    metrics = {
        "n_pairs": int(len(y_true_q)),
        "accuracy_qualif":   float(accuracy_score(y_true_q, y_pred_q)),
        "precision_qualif":  float(precision_score(y_true_q, y_pred_q, zero_division=0)),
        "recall_qualif":     float(recall_score(y_true_q, y_pred_q, zero_division=0)),
        "f1_qualif":         float(f1_score(y_true_q, y_pred_q, zero_division=0)),
        "winner_accuracy":   float(np.mean(winner_correct)),
        "winner_correct_count": int(sum(winner_correct)),
        "winner_total":      int(len(winner_correct)),
        "confusion_matrix":  {"tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
                              "fn": int(cm[1, 0]), "tp": int(cm[1, 1])},
    }
    # MAE 1T sur les 30 paires (5 années × 6 archétypes)
    pp = pd.DataFrame(per_pair_rows)
    metrics["mae_score_1T"] = float(np.mean(
        np.abs(pp["predicted_score_1T"] - pp["actual_score_1T"])
    ))
    return {
        "metrics": metrics,
        "per_year": pd.DataFrame(per_year_rows),
        "per_pair": pp,
        "classification_report": classification_report(
            y_true_q, y_pred_q, target_names=["non_qualif", "qualif"],
            output_dict=True, zero_division=0,
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.fitted.yaml")
    ap.add_argument("--data", default="data/historical_elections.csv")
    ap.add_argument("--polls", default="data/historical_polls_T2.csv")
    ap.add_argument("--out-dir", default="outputs")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)
    hist = pd.read_csv(args.data)
    polls = load_polls_T2(args.polls)

    result = evaluate_walk_forward(cfg, hist, polls)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result["per_year"].to_csv(
        out / "historical_validation_walk_forward_per_year.csv", index=False)
    result["per_pair"].to_csv(
        out / "historical_validation_walk_forward_per_pair.csv", index=False)
    with open(out / "historical_validation_walk_forward_summary.json", "w") as f:
        json.dump({"metrics": result["metrics"],
                   "classification_report": result["classification_report"]},
                  f, indent=2, ensure_ascii=False)

    m = result["metrics"]
    print("\n=== Validation walk-forward (T-2 mois, AUCUNE info post-hoc) ===")
    print(f"  N pairs : {m['n_pairs']}")
    print(f"  MAE score 1T   : {m['mae_score_1T']:.2f} pts")
    print(f"  Accuracy qualif : {m['accuracy_qualif']:.3f}")
    print(f"  Precision      : {m['precision_qualif']:.3f}")
    print(f"  Recall         : {m['recall_qualif']:.3f}")
    print(f"  F1             : {m['f1_qualif']:.3f}")
    print(f"  Confusion matrix : TP={m['confusion_matrix']['tp']}  "
          f"FN={m['confusion_matrix']['fn']}  FP={m['confusion_matrix']['fp']}  "
          f"TN={m['confusion_matrix']['tn']}")
    print(f"  Vainqueur 2T correct : "
          f"{m['winner_correct_count']}/{m['winner_total']} "
          f"= {m['winner_accuracy']*100:.0f}%")

    print(f"\n=== Détails par année ===")
    print(result["per_year"][["year", "actual_top2", "predicted_top2",
                               "actual_winner", "predicted_winner",
                               "winner_correct"]].to_string(index=False))


if __name__ == "__main__":
    main()
