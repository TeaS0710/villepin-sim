"""Scraping reproductible des sondages 1er tour T-2 mois pour chaque
élection présidentielle française 2002-2022, depuis Wikipédia.

Objectif : alimenter une validation walk-forward stricte qui n'utilise QUE
des informations disponibles AVANT l'élection (pas les résultats finaux).

Sortie : data/historical_polls_T2.csv avec colonnes
    year, candidate_norm, mean_pct, n_polls, window_start, window_end

La fenêtre est [T-90j, T-30j] (centrée sur T-2 mois) afin d'avoir un nombre
suffisant de sondages tout en restant ex-ante du résultat final.

Reproductibilité :
- Cache HTML dans data/raw/ avec SHA-256.
- Re-lance sans --force lit le cache (résultats identiques run à run).
"""
from __future__ import annotations
import argparse
import hashlib
import re
import unicodedata
from datetime import date, timedelta
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

WIKI_BASE = "https://fr.wikipedia.org/wiki/"
# Plusieurs patterns d'URL selon les années (Wikipédia n'est pas uniforme).
POLLS_URL_PATTERNS = [
    "Liste_de_sondages_sur_l%27élection_présidentielle_française_de_{year}",
    "Sondages_sur_l%27élection_présidentielle_française_de_{year}",
]

# Date du 1er tour de chaque élection (servant à calculer la fenêtre T-2 mois).
ELECTION_DATES = {
    2002: date(2002, 4, 21),
    2007: date(2007, 4, 22),
    2012: date(2012, 4, 22),
    2017: date(2017, 4, 23),
    2022: date(2022, 4, 10),
}

# Candidats-clés par année pour identifier la "bonne" table parmi les ~50
# tables HTML d'une page Wikipédia.
KEY_CANDIDATES = {
    2002: ["Chirac", "Jospin", "Le Pen", "Bayrou", "Madelin"],
    2007: ["Sarkozy", "Royal", "Bayrou", "Le Pen"],
    2012: ["Hollande", "Sarkozy", "Bayrou", "Le Pen", "Mélenchon"],
    2017: ["Macron", "Le Pen", "Fillon", "Mélenchon", "Hamon"],
    2022: ["Macron", "Le Pen", "Mélenchon", "Pécresse", "Zemmour"],
}

USER_AGENT = "DDV_ML/0.1 (villepin sim project; vergneadrien65@gmail.com)"
TIMEOUT = 30

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"


def _normalize_str(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("'", "").replace("-", " ").lower().strip()


def _strip_accents(s: str) -> str:
    """Pour les noms de fichiers cache."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("'", "_").replace(" ", "_")


def fetch_page(year: int, force: bool = False) -> str:
    """Télécharge la page sondages Wikipédia pour `year`, ou la relit du cache."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"wiki_polls_{year}.html"
    sha_path = RAW_DIR / f"wiki_polls_{year}.sha256"

    if cache_path.exists() and not force:
        return cache_path.read_text(encoding="utf-8")

    last_err = None
    for pattern in POLLS_URL_PATTERNS:
        url = WIKI_BASE + pattern.format(year=year)
        try:
            r = requests.get(url, headers={"User-Agent": USER_AGENT},
                             timeout=TIMEOUT)
            if r.status_code == 200:
                cache_path.write_text(r.text, encoding="utf-8")
                sha = hashlib.sha256(r.text.encode()).hexdigest()
                sha_path.write_text(f"{sha}  {cache_path.name}\n")
                print(f"  fetched {year}  ({len(r.text)//1024} kB)")
                return r.text
            last_err = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            last_err = str(e)
    raise RuntimeError(f"Impossible de télécharger les sondages {year} : {last_err}")


def _flatten_columns(cols) -> list[str]:
    """Aplatit un MultiIndex (souvent retourné par read_html) en str simples,
    en gardant le niveau le plus informatif (le nom du candidat si présent)."""
    out = []
    for c in cols:
        if isinstance(c, tuple):
            parts = [str(p) for p in c if not str(p).startswith("Unnamed")]
            out.append(parts[-1] if parts else "")
        else:
            out.append(str(c))
    return out


def _parse_date_fr(s: str, year: int) -> date | None:
    """Parse une cellule date FR comme '12 février 2017', '12 fév. 2017',
    '12 fév.-15 fév. 2017', '2-4 février 2017' etc.

    Retourne la date de FIN du sondage (la plus récente).
    """
    if not s or pd.isna(s):
        return None
    s = str(s).strip()
    months = {
        "janv": 1, "fev": 2, "fév": 2, "mars": 3, "avr": 4, "mai": 5, "juin": 6,
        "juil": 7, "aout": 8, "août": 8, "sept": 9, "oct": 10, "nov": 11, "dec": 12, "déc": 12,
    }
    # Cherche le DERNIER jour mentionné suivi du mois et de l'année
    pat = re.compile(
        r"(\d{1,2})\s*(janv|fev|fév|mars|avr|mai|juin|juil|aout|août|sept|oct|nov|dec|déc)[a-zé.]*\s*(\d{4})?",
        re.IGNORECASE,
    )
    found = pat.findall(s)
    if not found:
        return None
    last = found[-1]
    day = int(last[0])
    mon_key = last[1].lower().rstrip(".")
    # normalise les abrévations possibles
    for k in months:
        if mon_key.startswith(k):
            month = months[k]
            break
    else:
        return None
    y = int(last[2]) if last[2] else year
    try:
        return date(y, month, day)
    except ValueError:
        return None


def _parse_pct(s) -> float | None:
    """'24,01 %' / '24.01' / '24' -> 24.01"""
    if s is None or pd.isna(s):
        return None
    txt = str(s).replace("\xa0", "").replace("%", "").strip()
    txt = txt.replace(",", ".")
    # extraire le premier nombre (parfois '24,5 (1er)')
    m = re.search(r"-?\d+(?:\.\d+)?", txt)
    if not m:
        return None
    try:
        v = float(m.group())
        if 0 <= v <= 100:
            return v
    except ValueError:
        pass
    return None


def _is_candidate_column(col_name: str, key_candidates: list[str]) -> bool:
    """Une colonne représente-t-elle un candidat clé ?

    Rejette les colonnes vides ou techniques (Sondeur, Date, etc.) pour éviter
    les faux positifs (chaîne vide qui matche tous les noms).
    """
    n = _normalize_str(col_name)
    if not n or len(n) < 3:
        return False
    if n in {"sondeur", "date", "echantillon", "abstention", "indecis (echantillon)",
             "abstention, blanc ou nul", "region", "circonscription", "pays",
             "profession", "diplome", "age", "sexe", "religion",
             "autres candidats", "non exprime", "indecis"}:
        return False
    return any(_normalize_str(k) in n for k in key_candidates if k)


def find_polls_table(html: str, year: int) -> pd.DataFrame:
    """Identifie la grande table des sondages 1er tour et la nettoie.

    Heuristique : on cherche une table avec MAX de candidats-clés distincts
    parmi ses colonnes (signature 1er tour, qui contient 5+ candidats), une
    colonne date, et une colonne sondeur. On rejette explicitement les tables
    qui n'ont que 2 candidats (sondages 2nd tour).
    """
    tables = pd.read_html(StringIO(html))
    best = None
    best_score = -1
    for t in tables:
        cols = _flatten_columns(t.columns)
        n_cand_cols = sum(_is_candidate_column(c, KEY_CANDIDATES[year])
                          for c in cols)
        # Une table 2T n'a que 2 candidats : on la rejette d'emblée pour les
        # tables "petites" (< 30 colonnes/lignes), où la confusion est plus
        # probable.
        if n_cand_cols < 3:
            continue
        has_date = any("date" in _normalize_str(c) for c in cols)
        has_pollster = any(_normalize_str(c).startswith("sond") for c in cols)
        # Le nb de candidats distincts est le critère principal.
        score = n_cand_cols * 20 + (5 if has_date else 0) + (3 if has_pollster else 0)
        score += min(len(t), 100) / 20
        if score > best_score:
            best_score = score
            best = t
    if best is None:
        raise RuntimeError(f"Aucune table de sondages 1T trouvée pour {year}")
    df = best.copy()
    df.columns = _flatten_columns(df.columns)
    return df


def extract_polls_T2(year: int, df: pd.DataFrame) -> pd.DataFrame:
    """Filtre les sondages dans la fenêtre [T-90j, T-30j] et calcule la
    moyenne par candidat sur cette fenêtre.

    Retourne un DataFrame [candidate_norm, candidate, mean_pct, n_polls,
    window_start, window_end].
    """
    el_date = ELECTION_DATES[year]
    window_start = el_date - timedelta(days=90)
    window_end = el_date - timedelta(days=30)

    # Trouve la colonne date
    date_col = next((c for c in df.columns
                     if "date" in _normalize_str(c)), None)
    if date_col is None:
        raise RuntimeError(f"Pas de colonne date {year}")
    dates = df[date_col].apply(lambda s: _parse_date_fr(s, year))
    mask = dates.apply(lambda d: (d is not None
                                  and window_start <= d <= window_end))
    sub = df[mask].copy()
    if sub.empty:
        # Élargir si trop strict (parfois peu de sondages T-2 mois pour 2002)
        window_start = el_date - timedelta(days=120)
        window_end = el_date - timedelta(days=20)
        mask = dates.apply(lambda d: (d is not None
                                      and window_start <= d <= window_end))
        sub = df[mask].copy()

    # Identifier les colonnes candidat-clés
    candidate_cols = [c for c in sub.columns
                      if _is_candidate_column(c, KEY_CANDIDATES[year])
                      or _is_candidate_column(c, _candidate_list_for_year(year))]

    rows = []
    seen_cols = set()
    for c in candidate_cols:
        if c in seen_cols:
            continue
        seen_cols.add(c)
        col_data = sub[c]
        if isinstance(col_data, pd.DataFrame):
            # Plusieurs colonnes ont le même nom (fréquent sur Wikipédia) :
            # on garde la 1re série non-vide.
            col_data = col_data.iloc[:, 0]
        vals = col_data.apply(_parse_pct).dropna()
        if len(vals) < 2:
            continue
        rows.append({
            "year": year,
            "candidate": c,
            "candidate_norm": _normalize_str(c),
            "mean_pct": float(vals.mean()),
            "median_pct": float(vals.median()),
            "std_pct": float(vals.std()),
            "n_polls": int(len(vals)),
            "window_start": window_start.isoformat(),
            "window_end": window_end.isoformat(),
        })
    return pd.DataFrame(rows)


def _candidate_list_for_year(year: int) -> list[str]:
    """Liste élargie pour ne pas rater les candidats secondaires."""
    from .historical_context import ARCHETYPE_MAPPING
    out = []
    for arch, cands in ARCHETYPE_MAPPING[year].items():
        out.extend(cands)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="re-télécharge les pages (ignore le cache)")
    ap.add_argument("--out", default="data/historical_polls_T2.csv")
    args = ap.parse_args()

    all_rows = []
    for year in [2002, 2007, 2012, 2017, 2022]:
        print(f"== {year} ==")
        html = fetch_page(year, force=args.force)
        tbl = find_polls_table(html, year)
        print(f"  table : {tbl.shape[0]} lignes × {tbl.shape[1]} colonnes")
        polls = extract_polls_T2(year, tbl)
        print(f"  {len(polls)} candidats avec sondages T-2 mois")
        all_rows.append(polls)

    out_df = pd.concat(all_rows, ignore_index=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nDone. {len(out_df)} lignes écrites dans {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
