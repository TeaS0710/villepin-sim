"""Scraping reproductible des résultats du 1er tour des présidentielles
françaises 2002-2022 depuis Wikipédia.

Reproductibilité :
- Chaque page est sauvegardée en HTML dans data/raw/, avec son SHA-256.
- Re-lance sans `--force` lit le cache local (résultats identiques run à run).
- `--force` re-télécharge.

Sortie : data/historical_elections.csv avec colonnes
    year, candidate, party, votes, pct_exprimes
"""
from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

WIKI_BASE = "https://fr.wikipedia.org/wiki/"
PAGES = {
    2002: "Élection_présidentielle_française_de_2002",
    2007: "Élection_présidentielle_française_de_2007",
    2012: "Élection_présidentielle_française_de_2012",
    2017: "Élection_présidentielle_française_de_2017",
    2022: "Élection_présidentielle_française_de_2022",
}

# Candidats "phares" qu'on utilise pour identifier la table de résultats
# de chaque élection (présence simultanée nécessaire).
TABLE_MARKERS = {
    2002: ("Chirac", "Le Pen", "Jospin"),
    2007: ("Sarkozy", "Royal", "Bayrou"),
    2012: ("Hollande", "Sarkozy", "Le Pen"),
    2017: ("Macron", "Le Pen", "Fillon"),
    2022: ("Macron", "Le Pen", "Mélenchon"),
}

USER_AGENT = "DDV_ML/0.1 (villepin sim project; vergneadrien65@gmail.com)"
TIMEOUT = 30


def _normalize_pct(text: str) -> float | None:
    """'24,01\xa0%' -> 24.01 ; '24.01' -> 24.01 ; '' -> None"""
    if not text:
        return None
    cleaned = (
        text.replace("\xa0", "")
        .replace(" ", "")
        .replace("%", "")
        .replace(",", ".")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_votes(text: str) -> int | None:
    cleaned = text.replace("\xa0", "").replace(" ", "").replace(",", "").strip()
    try:
        return int(cleaned)
    except ValueError:
        return None


def _clean_candidate_name(text: str) -> str:
    """Retire annotations '[1]', âge entre parenthèses, etc."""
    s = re.sub(r"\[[^\]]*\]", "", text)
    s = re.sub(r"\(\d+\s*ans?\)", "", s)
    s = s.replace("\xa0", " ")
    return s.strip()


def _slug(s: str) -> str:
    """Slug ASCII pour fichiers."""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_ = "".join(c for c in nfkd if not unicodedata.combining(c))
    return re.sub(r"[^A-Za-z0-9._-]+", "_", ascii_)


def fetch_page(year: int, raw_dir: Path, force: bool = False) -> str:
    raw_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(PAGES[year])
    html_path = raw_dir / f"wiki_pres_{year}_{slug}.html"
    if html_path.exists() and not force:
        return html_path.read_text(encoding="utf-8")
    url = WIKI_BASE + PAGES[year]
    r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)
    r.raise_for_status()
    html_path.write_text(r.text, encoding="utf-8")
    digest = hashlib.sha256(r.text.encode("utf-8")).hexdigest()[:12]
    (raw_dir / f"wiki_pres_{year}.sha256").write_text(digest)
    return r.text


def _find_results_table(soup: BeautifulSoup, year: int) -> BeautifulSoup | None:
    """Trouve la table 'Résultats' identifiée par la co-présence des candidats
    phares et d'un en-tête 'Premier tour'."""
    markers = TABLE_MARKERS[year]
    for table in soup.find_all("table", class_="wikitable"):
        text = table.get_text(" ", strip=True)
        if all(m in text for m in markers) and "Premier tour" in text:
            return table
    return None


def parse_first_round(html: str, year: int) -> pd.DataFrame:
    """Parse la table des résultats officiels, retourne DataFrame du 1er tour."""
    soup = BeautifulSoup(html, "html.parser")
    table = _find_results_table(soup, year)
    if table is None:
        raise RuntimeError(f"Pas de table de résultats trouvée pour {year}")

    rows = table.find_all("tr")
    # Identifier les indices de colonnes 'Voix' et '%' du 1er tour.
    # Structure typique : header row 0 = 'Candidats | Partis | Premier tour | Second tour'
    # header row 1 = 'Voix | % | Voix | %'
    # data rows ensuite.
    # On lit toutes les cellules d'une ligne data et on prend les 2 dernières
    # colonnes numériques AVANT le 2T, càd colonnes 3 et 4 (idx 3,4) avec
    # éventuellement un offset si la couleur/parti est en col supplémentaire.
    data = []
    for tr in rows:
        cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
        if len(cells) < 5:
            continue
        # On veut une ligne data : la 1re cellule contient un nom de candidat.
        candidate_raw = cells[1] if cells[0] == "" else cells[0]
        if not candidate_raw or candidate_raw in {"Voix", "%", "Candidats", "Premier tour", "Second tour"}:
            continue
        # Cherche dans la ligne deux nombres consécutifs : voix puis pct
        # (le pct contient une virgule).
        votes, pct, party = None, None, None
        # Indices : cellule 1 = candidat (parfois 0), cellule 2 = parti,
        # cellule 3 = voix 1T, cellule 4 = % 1T (avec possible décalage couleur).
        # Stratégie robuste : itérer et prendre la première séquence (int gros, float<100).
        for i in range(len(cells) - 1):
            v = _normalize_votes(cells[i])
            p = _normalize_pct(cells[i + 1])
            if v is not None and v > 5000 and p is not None and 0 < p < 100:
                votes, pct = v, p
                # Le parti = cellule juste avant la voix, généralement non-vide non-numérique
                party_idx = i - 1
                while party_idx >= 0 and not cells[party_idx]:
                    party_idx -= 1
                if party_idx >= 0:
                    party = cells[party_idx]
                break
        if votes is None or pct is None:
            continue
        cand_clean = _clean_candidate_name(candidate_raw)
        if not cand_clean or any(x in cand_clean.lower() for x in ["votes exprimé", "blancs", "nuls", "abstention", "inscrits", "exprim", "participation"]):
            continue
        data.append(
            {
                "year": year,
                "candidate": cand_clean,
                "party": party,
                "votes": votes,
                "pct_exprimes": pct,
            }
        )

    if not data:
        raise RuntimeError(f"Aucune ligne parsée pour {year}")
    df = pd.DataFrame(data)
    # Tri décroissant + dédup (cas où la ligne 'total' aurait fuité)
    df = df.sort_values("pct_exprimes", ascending=False).reset_index(drop=True)
    df = df[df["pct_exprimes"] <= 50]  # exclut accidentellement la ligne total
    df = df.drop_duplicates(subset=["candidate"], keep="first").reset_index(drop=True)
    return df


def build_dataset(raw_dir: Path, force: bool = False) -> pd.DataFrame:
    frames = []
    for year in PAGES:
        html = fetch_page(year, raw_dir, force=force)
        df = parse_first_round(html, year)
        print(f"[{year}] {len(df)} candidats parsés. "
              f"Total %: {df['pct_exprimes'].sum():.2f}. "
              f"Top 3: {', '.join(df['candidate'].head(3).tolist())}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-télécharge les pages")
    ap.add_argument("--out", default="data/historical_elections.csv")
    ap.add_argument("--raw", default="data/raw")
    args = ap.parse_args()
    df = build_dataset(Path(args.raw), force=args.force)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n✓ {len(df)} lignes -> {out_path}")


if __name__ == "__main__":
    main()
