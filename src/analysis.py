"""
analysis.py — Conclusiones condicionales a partir del Monte Carlo.

Reutiliza el mismo motor de simulacion pero registra el resultado de fase de
grupos de cada equipo (puntos, posicion, si avanzo) para responder preguntas como:

  - P(avanzar a la eliminatoria | puntos sumados en el grupo)
  - P(avanzar | quedar 3ro, segun puntos)  -> el "corte" de los mejores terceros
  - Cuantos puntos hacen falta para clasificar con cierta confianza

Uso:
    python src/analysis.py [N] [--team "Mexico"]

Salidas:
    outputs/avance_por_puntos.csv
    outputs/avance_por_puntos.png
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from goals import ScoreSampler, load_calib
from simulate import load_team_defs
from tournament import simulate_tournament

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"

MAXP = 9  # puntos posibles en 3 partidos: 0..9 (8 es imposible)


def run(n_sims, seed=0, focus_team=None):
    calib = load_calib()
    team_defs, _ = load_team_defs()
    rng = np.random.default_rng(seed)
    sampler = ScoreSampler(calib["rho"], rng=rng)

    # [puntos] -> [total, avanzaron], global y por posicion 3ra; y para 1 equipo.
    tot = np.zeros(MAXP + 1)
    adv = np.zeros(MAXP + 1)
    tot3 = np.zeros(MAXP + 1)
    adv3 = np.zeros(MAXP + 1)
    tot_team = np.zeros(MAXP + 1)
    adv_team = np.zeros(MAXP + 1)

    def recorder(name, group, place, pts, gd, gf, advanced):
        tot[pts] += 1
        adv[pts] += advanced
        if place == 3:
            tot3[pts] += 1
            adv3[pts] += advanced
        if focus_team and name == focus_team:
            tot_team[pts] += 1
            adv_team[pts] += advanced

    for _ in range(n_sims):
        simulate_tournament(team_defs, sampler, calib, rng, on_group_result=recorder)

    def rate(a, t):
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(t > 0, a / t, np.nan)

    rows = []
    for p in range(MAXP + 1):
        if tot[p] == 0:
            continue
        rows.append({
            "puntos": p,
            "p_avanzar": rate(adv, tot)[p],
            "freq_global": tot[p] / tot.sum(),
            "p_avanzar_si_3ro": rate(adv3, tot3)[p],
            "casos_3ro": int(tot3[p]),
        })
    df = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "avance_por_puntos.csv", index=False)

    team_df = None
    if focus_team:
        team_rows = [
            {"puntos": p, "p_avanzar": rate(adv_team, tot_team)[p],
             "freq": tot_team[p] / max(tot_team.sum(), 1)}
            for p in range(MAXP + 1) if tot_team[p] > 0
        ]
        team_df = pd.DataFrame(team_rows)

    return df, team_df


def plot(df, focus_team, team_df):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["puntos"], df["p_avanzar"] * 100, "o-", label="Cualquier equipo", color="#2c7fb8")
    m = df["p_avanzar_si_3ro"].notna()
    ax.plot(df["puntos"][m], df["p_avanzar_si_3ro"][m] * 100, "s--",
            label="Si terminás 3ro", color="#d95f0e")
    if team_df is not None:
        ax.plot(team_df["puntos"], team_df["p_avanzar"] * 100, "^-",
                label=focus_team, color="#31a354")
    ax.set_xlabel("Puntos en la fase de grupos")
    ax.set_ylabel("Probabilidad de avanzar (%)")
    ax.set_title("Mundial 2026 — P(avanzar a eliminatoria | puntos)")
    ax.set_xticks(range(0, 10))
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "avance_por_puntos.png", dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--team", type=str, default=None, help="equipo a destacar (opcional)")
    args = ap.parse_args()

    print(f"Analizando {args.n:,} simulaciones...")
    df, team_df = run(args.n, args.seed, args.team)
    plot(df, args.team, team_df)

    show = df.copy()
    show["p_avanzar"] = (show["p_avanzar"] * 100).round(1)
    show["p_avanzar_si_3ro"] = (show["p_avanzar_si_3ro"] * 100).round(1)
    show["freq_global"] = (show["freq_global"] * 100).round(1)
    show.columns = ["Puntos", "P(avanzar) %", "Freq %", "P(avanzar|3ro) %", "casos_3ro"]
    print("\nP(avanzar a la eliminatoria) según puntos de grupo:")
    print(show[["Puntos", "P(avanzar) %", "Freq %", "P(avanzar|3ro) %"]].to_string(index=False))
    if team_df is not None:
        t = team_df.copy()
        t["p_avanzar"] = (t["p_avanzar"] * 100).round(1)
        t["freq"] = (t["freq"] * 100).round(1)
        print(f"\n{args.team} — P(avanzar) por puntos:")
        print(t.to_string(index=False))
    print(f"\n-> outputs/avance_por_puntos.csv  y  .png")


if __name__ == "__main__":
    main()
