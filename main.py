"""Point d'entrée unique du pipeline DDV_ML.

Usage :
    python main.py --mode quick        # 3k configs, NN, CMA-ES, rapport
    python main.py --mode full         # 30k configs, plus de restarts
    python main.py --skip-calibration  # passe la phase 1C (si déjà faite)
    python main.py --only-report       # régénère seulement le rapport
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--skip-scrape", action="store_true")
    ap.add_argument("--skip-calibration", action="store_true")
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--skip-nn", action="store_true")
    ap.add_argument("--skip-cmaes", action="store_true")
    ap.add_argument("--skip-winner", action="store_true")
    ap.add_argument("--skip-extreme", action="store_true")
    ap.add_argument("--llm-iterations", type=int, default=0,
                    help="N>0 active la boucle LLM Tier 2 après le premier CMA-ES")
    ap.add_argument("--only-report", action="store_true")
    args = ap.parse_args()

    py = sys.executable

    if args.only_report:
        run([py, "-m", "src.reporter"])
        return

    # Phase 0bis : scraping (idempotent grâce au cache HTML)
    if not args.skip_scrape and not Path("data/historical_elections.csv").exists():
        run([py, "-m", "src.historical_data"])

    # Phase 1C : calibration
    if not args.skip_calibration:
        run([py, "-m", "src.calibration", "--write-fitted-config"])

    # Phase 2A : dataset
    if not args.skip_dataset:
        run([py, "-m", "src.dataset_generator", "--mode", args.mode])

    # Phase 2B : NN
    if not args.skip_nn:
        run([py, "-m", "src.neural_predictor"])

    # Phase 3 : CMA-ES
    if not args.skip_cmaes:
        cmaes_cmd = [py, "-m", "src.genetic_optimizer"]
        if args.mode == "quick":
            cmaes_cmd.append("--quick")
        run(cmaes_cmd)

    # Phase 3bis : validation honnête walk-forward (sondages T-2 mois, sans aucune
    # info post-hoc). Mesure la vraie capacité prédictive du modèle.
    if not args.skip_calibration:
        run([py, "-m", "src.historical_polls_scraper"])
        run([py, "-m", "src.historical_validation"])
        run([py, "-m", "src.historical_validation_walk_forward"])

    # Phase 4 : boucle LLM Tier 2 (optionnelle).
    # On force mode=quick pour les itérations LLM, sinon ré-entraîner le NN
    # full prend 5-10 min × N itérations. Si une itération est gardée, la
    # ré-exécution finale du pipeline full pourra confirmer.
    if args.llm_iterations > 0:
        run([py, "-m", "src.llm_loop",
             "--iterations", str(args.llm_iterations),
             "--mode", "quick"])
        # Si des sub-params ont été acceptés, refait dataset + NN + CMA-ES en
        # mode `args.mode` pour validation avec le NN full-précision.
        run([py, "-m", "src.dataset_generator", "--mode", args.mode])
        run([py, "-m", "src.neural_predictor"])
        cmaes_cmd2 = [py, "-m", "src.genetic_optimizer"]
        if args.mode == "quick":
            cmaes_cmd2.append("--quick")
        run(cmaes_cmd2)

    # Phase 3bis : qui gagne ? (P(victoire) tous candidats × scénarios × shocks)
    if not args.skip_winner:
        run([py, "-m", "src.winner_analysis"])

    # Phase 3ter : recherche extrême (peut-on faire gagner Villepin sans biais ?)
    if not args.skip_extreme:
        n_restarts = "10" if args.mode == "full" else "5"
        run([py, "-m", "src.extreme_search", "--n-restarts", n_restarts])
        # Phase 3quater : chemin minimal vers la victoire (perturbation depuis baseline)
        run([py, "-m", "src.path_to_victory",
             "--extreme-restarts", n_restarts,
             "--targets", "0.05", "0.10", "0.25", "0.50", "0.75"])

    # Phase 5 : rapport
    run([py, "-m", "src.reporter"])

    print("\n=== Pipeline terminé ===")
    print("  outputs/final_report.md")
    print("  outputs/plots/")
    print("  pour le dashboard : streamlit run src/dashboard.py")


if __name__ == "__main__":
    main()
