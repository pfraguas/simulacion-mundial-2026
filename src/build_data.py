"""
build_data.py — Prepara los datos para la simulacion.

1. Calcula ratings Elo para todas las selecciones recorriendo el historico
   completo de partidos internacionales (motor en elo.py). Auto-contenido y
   reproducible: no depende de mapeos de codigos de pais externos.
2. Deriva los 12 grupos del Mundial 2026 desde los fixtures reales (en un grupo
   de 4 los 4 equipos juegan entre si -> componentes conexas del grafo).

Salidas:
    data/elo_ratings.csv   (team, elo, n_matches, last_date)
    data/groups.json       ({"A": [t1,t2,t3,t4], ...})
"""
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

from elo import load_played, replay

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS_CSV = DATA / "results.csv"


def derive_groups(wc_fixtures: pd.DataFrame) -> dict:
    """Reconstruye los grupos via union-find sobre los enfrentamientos."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r in wc_fixtures.itertuples(index=False):
        parent[find(r.home_team)] = find(r.away_team)

    comps = defaultdict(list)
    for team in parent:
        comps[find(team)].append(team)

    groups = {}
    for i, members in enumerate(sorted(comps.values(), key=lambda m: sorted(m))):
        groups[chr(ord("A") + i)] = sorted(members)
    return groups


def main():
    played = load_played(RESULTS_CSV)
    elo, n_matches, last_date, _ = replay(played)

    df = pd.read_csv(RESULTS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    wc = df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-01-01")]
    groups = derive_groups(wc)
    wc_teams = sorted({t for g in groups.values() for t in g})

    rows = [
        {
            "team": t,
            "elo": round(elo[t], 1),
            "n_matches": n_matches[t],
            "last_date": last_date[t].date().isoformat() if t in last_date else "",
        }
        for t in wc_teams
    ]
    elo_df = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
    elo_df.to_csv(DATA / "elo_ratings.csv", index=False)

    with open(DATA / "groups.json", "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)

    print(f"Partidos historicos usados: {len(played):,}")
    print(f"Selecciones Mundial 2026: {len(wc_teams)}  |  Grupos: {len(groups)}\n")
    print("Top 10 por Elo:")
    print(elo_df.head(10).to_string(index=False))
    print("\nGrupos:")
    for g, members in groups.items():
        print(f"  {g}: {', '.join(members)}")


if __name__ == "__main__":
    main()
