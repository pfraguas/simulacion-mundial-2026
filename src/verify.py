"""
verify.py — Chequeos de sanidad del modelo y la simulacion.

1. Tasa de goles y de empates que produce el modelo Dixon-Coles sobre todos los
   enfrentamientos posibles del Mundial, vs. el objetivo de calibracion.
2. Convergencia Monte Carlo: probabilidades de campeon con dos semillas distintas.
"""
import itertools
from pathlib import Path

import numpy as np

from goals import expected_goals, load_calib, outcome_probs, score_matrix
from simulate import load_team_defs, run

ROOT = Path(__file__).resolve().parent.parent


def goal_and_draw_rates():
    calib = load_calib()
    team_defs, _ = load_team_defs()
    elos = {n: e for n, e, _ in team_defs}
    tot_goals = tot_draw = w = 0.0
    k = np.arange(score_matrix(1, 1, calib["rho"]).shape[0])
    for a, b in itertools.combinations(elos, 2):
        lam_h, lam_a = expected_goals(elos[a], elos[b], True, calib, calib["home_adv"])
        m = score_matrix(lam_h, lam_a, calib["rho"])
        exp_goals = (m.sum(1) @ k) + (m.sum(0) @ k)
        _, p_draw, _ = outcome_probs(lam_h, lam_a, calib["rho"])
        tot_goals += exp_goals
        tot_draw += p_draw
        w += 1
    return tot_goals / w, tot_draw / w


def main():
    calib = load_calib()
    print("== Chequeo 1: tasa de goles / empates del modelo ==")
    g, d = goal_and_draw_rates()
    print(f"  goles/partido (modelo, todos los cruces): {g:.3f}  | base_total calib: {calib['base_total']:.3f}")
    print(f"  prob. de empate (modelo):                 {d:.3f}  | empates historicos ~0.23")
    print("  (esperable un poco menos de goles entre rivales parejos del Mundial)")

    print("\n== Chequeo 2: convergencia Monte Carlo (campeon) ==")
    n = 20000
    df0, _ = run(n, seed=0)
    df1, _ = run(n, seed=123)
    a = df0.set_index("team")["Campeon"]
    b = df1.set_index("team")["Campeon"]
    diff = (a - b).abs()
    print(f"  N={n:,} por semilla. Max |dif| P(campeon): {diff.max() * 100:.2f} pp "
          f"(en {diff.idxmax()})")
    print(f"  Suma P(campeon) seed0={a.sum():.3f}  seed123={b.sum():.3f}")
    top = a.nlargest(5)
    print("  Top 5 (seed0 vs seed123):")
    for t in top.index:
        print(f"    {t:<12} {a[t] * 100:5.1f}%  vs  {b[t] * 100:5.1f}%")


if __name__ == "__main__":
    main()
