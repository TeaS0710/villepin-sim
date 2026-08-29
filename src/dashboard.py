"""Dashboard Streamlit pour interroger le modèle interactivement.

Lancement :
    streamlit run src/dashboard.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.historical_context import VILLEPIN_EQUIVALENT
from src.parameters import EXOGENOUS, INTERNAL, TIER1_PARAMS
from src.physical_model import (
    compute_villepin_mass,
    first_round_scores,
    load_config,
    second_round_probability,
    simulate_monte_carlo,
)

st.set_page_config(page_title="Villepin 2027 : exploration", layout="wide")


@st.cache_data
def _load_cfg():
    p = Path("config.fitted.yaml")
    if not p.exists():
        p = Path("config.yaml")
    return load_config(p)


@st.cache_data
def _load_history():
    return pd.read_csv("data/historical_elections.csv")


cfg = _load_cfg()
hist = _load_history()

st.title("Villepin 2027 : exploration interactive")
st.caption("⚠️ Exercice de modélisation exploratoire : pas une prédiction.")

with st.sidebar:
    st.header("Paramètres Tier 1")
    st.markdown("**Exogènes** (la campagne ne les contrôle pas)")
    params: dict[str, float] = {}
    for n in TIER1_PARAMS:
        if n in EXOGENOUS:
            params[n] = st.slider(n, 0.0, 1.0, 0.5, 0.05, key=f"sld_{n}")
    st.markdown("---")
    st.markdown("**Internes** (controllables)")
    for n in TIER1_PARAMS:
        if n in INTERNAL:
            params[n] = st.slider(n, 0.0, 1.0, 0.5, 0.05, key=f"sld_{n}")
    st.markdown("---")
    n_mc = st.slider("Monte Carlo (n samples)", 10, 500, 100, 10)
    noise = st.slider("Bruit σ", 0.0, 0.3, 0.15, 0.01)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Scores prédits (1er tour)")
    scores = first_round_scores(params, cfg)
    df_scores = pd.DataFrame({
        "candidat": list(scores.keys()),
        "score_1T (%)": [round(v, 2) for v in scores.values()],
    }).sort_values("score_1T (%)", ascending=False).reset_index(drop=True)
    st.dataframe(df_scores, use_container_width=True, hide_index=True)
    st.bar_chart(df_scores.set_index("candidat")["score_1T (%)"], height=240)

with col2:
    st.subheader("Indicateurs")
    mass = compute_villepin_mass(params, cfg)
    qualified, opponent, p2T = second_round_probability(scores, params, cfg)
    st.metric("Masse Villepin", f"{mass:.3f}")
    st.metric("Score 1er tour Villepin", f"{scores['villepin']:.2f} %")
    st.metric("Qualifié 2T ?", "✔ oui" if qualified else "✘ non")
    if qualified:
        st.metric(f"Adversaire 2T", opponent or "?")
        st.metric("P(victoire 2T)", f"{p2T*100:.2f} %")

    st.markdown("---")
    st.markdown("**Monte Carlo (bruit sur params)**")
    mc = simulate_monte_carlo(params, cfg, n_samples=n_mc, noise_std=noise)
    st.write({
        "score_1T_mean": round(mc["score_1T_mean"], 2),
        "score_1T_p5": round(mc["score_1T_p5"], 2),
        "score_1T_p95": round(mc["score_1T_p95"], 2),
        "p_qualif": round(mc["p_qualif"], 3),
        "p_victory": round(mc["p_victory"], 3),
    })

st.markdown("---")
st.subheader("Élections historiques de référence")
df_hist = hist.copy()
df_hist["villepin_equiv"] = df_hist.apply(
    lambda r: "★" if r["candidate"] == VILLEPIN_EQUIVALENT.get(r["year"]) else "",
    axis=1,
)
st.dataframe(
    df_hist[["year", "candidate", "party", "pct_exprimes", "villepin_equiv"]],
    use_container_width=True,
    hide_index=True,
)
