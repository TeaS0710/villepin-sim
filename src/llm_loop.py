"""Orchestre la boucle LLM complète :
  pour iteration 1..N :
    1. lire top stratégies CMA-ES de l'itération précédente
    2. appeler LLM, filtrer, intégrer sous-paramètres
    3. régénérer dataset, ré-entraîner NN, ré-optimiser
    4. comparer best p_victoire vs avant : keep si amélioration, sinon rollback

Sauvegarde l'évolution dans `outputs/llm_history/`.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from .llm_param_discovery import run_iteration
from .physical_model import load_config


def _run(cmd: list[str]) -> None:
    print(f"\n  $ {' '.join(cmd)}")
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)


def best_p_victory_from(scenarios_csv: Path) -> float:
    df = pd.read_csv(scenarios_csv)
    return float(df["best_p_victory"].max())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-in",  default="config.fitted.yaml")
    ap.add_argument("--config-out", default="config.fitted.yaml")
    ap.add_argument("--out-dir",    default="outputs")
    ap.add_argument("--history-dir", default="outputs/llm_history")
    ap.add_argument("--iterations", type=int, default=None,
                    help="override max_iterations from config")
    ap.add_argument("--mode", choices=["quick", "full"], default="full")
    args = ap.parse_args()

    cfg_path = Path(args.config_in)
    if not cfg_path.exists():
        cfg_path = Path("config.yaml")
    cfg = load_config(cfg_path)
    llm_cfg = cfg["pipeline"]["llm"]
    if not llm_cfg.get("enabled", False):
        print("LLM désactivé dans config.pipeline.llm.enabled. Stop.")
        return

    history = Path(args.history_dir)
    history.mkdir(parents=True, exist_ok=True)
    n_iter = args.iterations if args.iterations is not None else llm_cfg["max_iterations"]
    model = llm_cfg["model"]
    threshold = llm_cfg["redundancy_threshold"]
    n_sp = llm_cfg["params_per_iteration"]
    patience = llm_cfg["early_stop_patience"]
    max_total = llm_cfg["max_total_subparams"]
    py = sys.executable

    # Cache de la "best p_victory" courante
    scenarios_csv = Path(args.out_dir) / "cmaes_scenarios.csv"
    if not scenarios_csv.exists():
        print(f"Avant la boucle LLM, run de base manquant ({scenarios_csv}).")
        print("Lance d'abord `python main.py` ou la cmd genetic_optimizer.")
        sys.exit(1)
    current_best = best_p_victory_from(scenarios_csv)
    print(f"📍 Best p_victoire baseline = {current_best:.4f}")

    log = {"iterations": [], "baseline_best_p_victory": current_best,
           "model": model, "config_input": str(cfg_path)}

    bad_streak = 0
    accepted_total = 0

    for it in range(1, n_iter + 1):
        print(f"\n========== Itération LLM {it}/{n_iter} ==========")
        # Lit top stratégies de l'itération précédente
        top_csv = Path(args.out_dir) / "cmaes_top_candidates.csv"
        if not top_csv.exists():
            print(f"[stop] manque {top_csv}")
            break
        top_df = pd.read_csv(top_csv).sort_values("p_victory", ascending=False).head(20)

        # Appel LLM
        seed_iter = 1000 + it
        new_cfg, accepted, rejected = run_iteration(
            cfg, top_df, model=model, n_subparams=n_sp,
            redundancy_threshold=threshold, seed=seed_iter,
        )
        if not accepted:
            print("  [stop] aucun sous-paramètre accepté à cette itération.")
            log["iterations"].append({"iter": it, "accepted": [], "rejected_count": len(rejected)})
            break

        # Snapshot avant intégration
        snap_dir = history / f"iter_{it:02d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        with open(snap_dir / "config_before.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        with open(snap_dir / "accepted.json", "w") as f:
            json.dump([a.to_dict() for a in accepted], f, indent=2, ensure_ascii=False)
        with open(snap_dir / "rejected.json", "w") as f:
            json.dump([{"sp": sp.to_dict(), "reason": r} for sp, r in rejected],
                      f, indent=2, ensure_ascii=False)

        # Pré-check : limite max_total
        n_existing_t2 = len(cfg.get("tier2_params", {}))
        if n_existing_t2 + len(accepted) > max_total:
            keep = max_total - n_existing_t2
            print(f"  [budget] tronque à {keep} (max_total={max_total})")
            accepted = accepted[:keep]
            new_cfg, _, _ = run_iteration(
                cfg, top_df.head(0), model=model, n_subparams=0,
                redundancy_threshold=threshold, seed=seed_iter,
            )
            from .llm_param_discovery import apply_subparams_to_cfg
            new_cfg = apply_subparams_to_cfg(cfg, accepted)
            if not accepted:
                break

        # Persiste le nouveau config et relance la pipeline (dataset → NN → CMA-ES)
        cfg_tmp = Path(args.out_dir) / f"config_iter_{it:02d}.yaml"
        with open(cfg_tmp, "w") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)

        # Rerun pipeline avec ce config
        # 1. Dataset
        _run([py, "-m", "src.dataset_generator", "--config", str(cfg_tmp),
              "--mode", args.mode])
        # 2. NN
        _run([py, "-m", "src.neural_predictor", "--config", str(cfg_tmp)])
        # 3. CMA-ES (avec --quick si mode quick)
        cmd = [py, "-m", "src.genetic_optimizer", "--config", str(cfg_tmp)]
        if args.mode == "quick":
            cmd.append("--quick")
        _run(cmd)

        new_best = best_p_victory_from(scenarios_csv)
        delta = new_best - current_best
        improved = delta > 0.001  # seuil min 0.1pt

        with open(snap_dir / "config_after.yaml", "w") as f:
            yaml.safe_dump(new_cfg, f, sort_keys=False, allow_unicode=True)
        # Snapshot scenarios
        shutil.copy(scenarios_csv, snap_dir / "cmaes_scenarios.csv")

        log["iterations"].append({
            "iter": it,
            "accepted_names": [a.name for a in accepted],
            "rejected_count": len(rejected),
            "best_p_victory_before": current_best,
            "best_p_victory_after": new_best,
            "delta": delta,
            "decision": "keep" if improved else "rollback",
        })

        print(f"\n  best p_victoire : {current_best:.4f} → {new_best:.4f}  (Δ={delta:+.4f})")
        if improved:
            print(f"  ✓ keep ({len(accepted)} sub-params intégrés)")
            cfg = new_cfg
            current_best = new_best
            bad_streak = 0
            accepted_total += len(accepted)
            # Persiste le config principal "fitted"
            with open(args.config_out, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
        else:
            print(f"  ✗ rollback (pas d'amélioration significative)")
            bad_streak += 1
            if bad_streak >= patience:
                print(f"\n  [stop] {bad_streak} itérations consécutives sans gain. "
                      f"Patience={patience}.")
                break

    log["final_best_p_victory"] = current_best
    log["accepted_total"] = accepted_total
    log_path = history / "summary.json"
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False))
    print(f"\n✓ Résumé : {log_path}")
    print(f"  Best p_victoire finale = {current_best:.4f}")
    print(f"  Sub-params totaux acceptés = {accepted_total}")


if __name__ == "__main__":
    main()
