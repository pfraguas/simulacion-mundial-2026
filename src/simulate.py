"""
simulate.py — Loop Monte Carlo del Mundial 2026.

Simula el torneo completo N veces y agrega, por seleccion, la frecuencia con que
alcanza cada ronda. Las probabilidades = frecuencia / N.

Uso:
    python src/simulate.py [N] [--seed S]

Salida:
    outputs/probabilidades.csv
    (y un resumen por consola)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from goals import ScoreSampler, load_calib
from tournament import simulate_tournament

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"

# Rondas en orden; alcanzar una implica haber alcanzado todas las anteriores.
STAGES = ["R32", "Octavos", "Cuartos", "Semis", "Final", "Campeon"]
_STAGE_IDX = {s: i for i, s in enumerate(STAGES)}


def load_team_defs():
    elo = pd.read_csv(DATA / "elo_ratings.csv").set_index("team")["elo"].to_dict()
    with open(DATA / "groups.json", encoding="utf-8") as f:
        groups = json.load(f)
    defs = []
    for g, members in groups.items():
        for name in members:
            defs.append((name, float(elo[name]), g))
    return defs, groups


def run(n_sims, seed=0):
    calib = load_calib()
    team_defs, groups = load_team_defs()
    names = [d[0] for d in team_defs]
    rng = np.random.default_rng(seed)
    sampler = ScoreSampler(calib["rho"], rng=rng)

    # counts[name][stage] = veces que alcanzo (al menos) esa ronda
    counts = {name: np.zeros(len(STAGES), dtype=np.int64) for name in names}

    t0 = time.time()
    for s in range(n_sims):
        reached = simulate_tournament(team_defs, sampler, calib, rng)
        for name, stage in reached.items():
            if stage == "group":
                continue
            idx = _STAGE_IDX[stage]
            counts[name][: idx + 1] += 1  # acumula esta ronda y las previas
        if (s + 1) % max(1, n_sims // 10) == 0:
            el = time.time() - t0
            print(f"  {s + 1:>7,}/{n_sims:,}  ({el:.1f}s)", flush=True)

    rows = []
    for name, elo, group in team_defs:
        c = counts[name]
        row = {"team": name, "group": group, "elo": elo}
        for i, stage in enumerate(STAGES):
            row[stage] = c[i] / n_sims
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Campeon", ascending=False).reset_index(drop=True)

    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "probabilidades.csv", index=False)
    return df, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=50_000, help="numero de simulaciones")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"Simulando {args.n:,} mundiales (seed={args.seed})...")
    df, elapsed = run(args.n, args.seed)
    print(f"\nListo en {elapsed:.1f}s. -> outputs/probabilidades.csv\n")

    pct = df.copy()
    for s in STAGES:
        pct[s] = (pct[s] * 100).round(1)
    print("Top 15 candidatos al titulo (%):")
    cols = ["team", "group", "Octavos", "Cuartos", "Semis", "Final", "Campeon"]
    print(pct[cols].head(15).to_string(index=False))
    print(f"\nSuma P(Campeon): {df['Campeon'].sum():.3f}  (debe ser ~1.0)")


if __name__ == "__main__":
    main()
