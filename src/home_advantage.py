"""
home_advantage.py — Cuantifica el valor de la ventaja de localia.

Corre el Monte Carlo dos veces sobre los mismos equipos:
  A) CON localia: los anfitriones (EE.UU., Mexico, Canada) suman +home_adv Elo
     cuando juegan de local (valor calibrado, default 100).
  B) SIN localia: home_adv = 0  -> todos juegan en cancha neutral.

Compara, para cada anfitrion, la probabilidad de alcanzar cada ronda, y muestra
el efecto neto. Tambien permite barrer varios valores de home_adv.

Uso:
    python src/home_advantage.py [N] [--sweep]
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd

from goals import ScoreSampler, load_calib
from simulate import STAGES, load_team_defs
from tournament import HOSTS, simulate_tournament

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
_STAGE_IDX = {s: i for i, s in enumerate(STAGES)}


def run(n_sims, home_adv, team_defs, calib, seed=0):
    """Devuelve DataFrame de P(alcanzar ronda) por equipo, con home_adv dado."""
    cal = copy.deepcopy(calib)
    cal["home_adv"] = float(home_adv)
    rng = np.random.default_rng(seed)
    sampler = ScoreSampler(cal["rho"], rng=rng)
    names = [d[0] for d in team_defs]
    counts = {n: np.zeros(len(STAGES), dtype=np.int64) for n in names}

    for _ in range(n_sims):
        reached = simulate_tournament(team_defs, sampler, cal, rng)
        for name, stage in reached.items():
            if stage == "group":
                continue
            counts[name][: _STAGE_IDX[stage] + 1] += 1

    rows = [{"team": n, **{s: counts[n][i] / n_sims for i, s in enumerate(STAGES)}}
            for n in names]
    return pd.DataFrame(rows).set_index("team")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", action="store_true", help="barrer home_adv 0..125")
    args = ap.parse_args()

    calib = load_calib()
    team_defs, _ = load_team_defs()
    base_adv = calib["home_adv"]
    hosts = sorted(HOSTS)

    print(f"Anfitriones: {', '.join(hosts)}")
    print(f"home_adv calibrado: {base_adv:.0f} pts Elo  |  N={args.n:,} por escenario\n")

    # Misma semilla en ambos escenarios -> comparacion pareada (menos ruido).
    con = run(args.n, base_adv, team_defs, calib, seed=args.seed)
    sin = run(args.n, 0.0, team_defs, calib, seed=args.seed)

    print("Efecto de la localia (probabilidades %, CON vs SIN, y diferencia):")
    print(f"{'Equipo':<16}{'Ronda':<10}{'CON':>7}{'SIN':>7}{'Δ':>8}")
    for t in hosts:
        for stage in ["R32", "Octavos", "Cuartos", "Semis", "Campeon"]:
            c, s = con.loc[t, stage] * 100, sin.loc[t, stage] * 100
            print(f"{t:<16}{stage:<10}{c:>7.1f}{s:>7.1f}{c - s:>+8.1f}")
        print()

    # Resumen: cuanto "regala" la localia en P(campeon) a los anfitriones.
    delta_champ = (con["Campeon"] - sin["Campeon"]) * 100
    print("Cambio en P(campeon) por la localia:")
    for t in hosts:
        print(f"  {t:<16}{sin.loc[t,'Campeon']*100:5.1f}%  ->  "
              f"{con.loc[t,'Campeon']*100:5.1f}%   ({delta_champ[t]:+.1f} pp)")
    suma_anfitriones = con.loc[hosts, "Campeon"].sum() * 100
    print(f"\n  Suma P(campeon) anfitriones: {sin.loc[hosts,'Campeon'].sum()*100:.1f}% "
          f"(sin)  ->  {suma_anfitriones:.1f}% (con)")

    out = pd.DataFrame({
        "P_campeon_sin": sin["Campeon"], "P_campeon_con": con["Campeon"],
        "delta_pp": (con["Campeon"] - sin["Campeon"]) * 100,
    })
    OUT.mkdir(exist_ok=True)
    out.loc[hosts].to_csv(OUT / "ventaja_localia.csv")
    print("\n-> outputs/ventaja_localia.csv")

    if args.sweep:
        print("\nBarrido de home_adv (P(campeon) % por anfitrion):")
        print(f"{'home_adv':<10}" + "".join(f"{t[:10]:>12}" for t in hosts))
        for adv in [0, 25, 50, 75, 100, 125]:
            d = run(args.n // 2, adv, team_defs, calib, seed=args.seed)
            print(f"{adv:<10}" + "".join(f"{d.loc[t,'Campeon']*100:>12.1f}" for t in hosts))


if __name__ == "__main__":
    main()
