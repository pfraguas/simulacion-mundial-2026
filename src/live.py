"""
live.py — Simulacion *in-tournament*: re-estima las probabilidades fijando los
resultados de grupo ya jugados y simulando solo lo que falta.

Lee los marcadores reales del Mundial 2026 desde data/results.csv (los partidos
jugados tienen score; los pendientes vienen como NA) y corre el Monte Carlo
condicionado a esos resultados. Asi las probabilidades se actualizan a medida que
avanza el torneo, sin tocar el modelo.

Uso:
    python src/live.py [N] [--seed S]

Salida:
    outputs/probabilidades_live.csv   (+ comparacion vs. linea de base pre-torneo)

Nota: por ahora solo se fijan resultados de FASE DE GRUPOS (los unicos disponibles
y bien definidos). El bracket de eliminatoria es sembrado por fuerza, no posicional,
asi que fijar resultados de eliminatoria queda como refinamiento futuro.
"""
import argparse
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import build_data
from goals import ScoreSampler, load_calib
from simulate import STAGES, load_team_defs
from tournament import simulate_tournament

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
_STAGE_IDX = {s: i for i, s in enumerate(STAGES)}

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def update_data(timeout=30):
    """Descarga el results.csv mas reciente y reconstruye Elo + grupos.

    Si la descarga falla (sin red, etc.) avisa y sigue con la copia local.
    Devuelve True si actualizo, False si uso la copia existente.
    """
    dest = DATA / "results.csv"
    try:
        before = pd.read_csv(dest)
        n_before = before.dropna(subset=["home_score", "away_score"]).shape[0]
    except FileNotFoundError:
        n_before = None

    repo = "/".join(RESULTS_URL.split("/")[3:5])  # martj42/international_results
    print(f"Descargando resultados de {repo} ...")
    try:
        req = urllib.request.Request(RESULTS_URL, headers={"User-Agent": "mundial-sim"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        # Validar que es un CSV plausible antes de pisar el archivo bueno.
        if not data or b"home_team" not in data[:200]:
            raise ValueError("respuesta inesperada (no parece results.csv)")
        with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        n_after = pd.read_csv(tmp_path).dropna(subset=["home_score", "away_score"]).shape[0]
        shutil.move(tmp_path, dest)
    except Exception as e:  # red caida, timeout, etc. -> seguir con lo local
        print(f"  ⚠ No se pudo actualizar ({e}). Uso la copia local.")
        return False

    print(f"  OK. Partidos con resultado: {n_before} -> {n_after}")
    print("Reconstruyendo Elo y grupos...")
    build_data.build(verbose=False)
    return True


def load_known_group_results(team_defs):
    """Lee results.csv y devuelve los partidos de grupo ya jugados.

    Un partido cuenta como 'de grupo' si ambos equipos estan en el mismo grupo
    (robusto: no depende de etiquetas de ronda). Devuelve:
      known: {frozenset({home,away}): (home, away, hs, as_)}
      played_rows: DataFrame de los partidos fijados (para mostrar).
    """
    group_of = {name: g for name, _, g in team_defs}
    df = pd.read_csv(DATA / "results.csv")
    df["date"] = pd.to_datetime(df["date"])
    wc = df[(df.tournament == "FIFA World Cup") & (df.date >= "2026-01-01")].copy()
    played = wc.dropna(subset=["home_score", "away_score"]).copy()

    known = {}
    rows = []
    for r in played.itertuples(index=False):
        h, a = r.home_team, r.away_team
        if group_of.get(h) and group_of.get(h) == group_of.get(a):
            hs, as_ = int(r.home_score), int(r.away_score)
            known[frozenset((h, a))] = (h, a, hs, as_)
            rows.append({"group": group_of[h], "date": r.date.date(),
                         "home": h, "hs": hs, "as": as_, "away": a})
    played_rows = pd.DataFrame(rows).sort_values(["group", "date"]) if rows else pd.DataFrame()
    return known, played_rows


def run_live(n_sims, known, team_defs, calib, seed=0):
    rng = np.random.default_rng(seed)
    sampler = ScoreSampler(calib["rho"], rng=rng)
    names = [d[0] for d in team_defs]
    counts = {n: np.zeros(len(STAGES), dtype=np.int64) for n in names}
    t0 = time.time()
    for _ in range(n_sims):
        reached = simulate_tournament(team_defs, sampler, calib, rng,
                                      known_results=known)
        for name, stage in reached.items():
            if stage == "group":
                continue
            counts[name][: _STAGE_IDX[stage] + 1] += 1
    group_of = {n: g for n, _, g in team_defs}
    rows = [{"team": n, "group": group_of[n],
             **{s: counts[n][i] / n_sims for i, s in enumerate(STAGES)}}
            for n in names]
    df = pd.DataFrame(rows).sort_values("Campeon", ascending=False).reset_index(drop=True)
    return df, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n", nargs="?", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-update", action="store_true",
                    help="no descargar resultados; usar la copia local")
    args = ap.parse_args()

    if not args.no_update:
        update_data()
        print()

    calib = load_calib()
    team_defs, _ = load_team_defs()
    known, played = load_known_group_results(team_defs)

    total_group_matches = 72
    print(f"Partidos de grupo jugados (fijados): {len(known)} / {total_group_matches}")
    if not played.empty:
        print("\nResultados ya cargados:")
        for r in played.itertuples(index=False):
            print(f"  [{r.group}] {r.home} {r.hs}-{r._4} {r.away}  ({r.date})")

    print(f"\nSimulando {args.n:,} mundiales condicionados...")
    df, elapsed = run_live(args.n, known, team_defs, calib, args.seed)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "probabilidades_live.csv", index=False)
    print(f"Listo en {elapsed:.1f}s -> outputs/probabilidades_live.csv\n")

    # Comparacion con la linea de base pre-torneo, si existe.
    base_path = OUT / "probabilidades.csv"
    pct = df.copy()
    for s in STAGES:
        pct[s] = pct[s] * 100
    cols = ["team", "group", "Octavos", "Cuartos", "Semis", "Campeon"]

    if base_path.exists():
        base = pd.read_csv(base_path).set_index("team")["Campeon"] * 100
        pct = pct.set_index("team")
        pct["Campeon_pre"] = base
        pct["delta_pp"] = pct["Campeon"] - pct["Campeon_pre"]
        pct = pct.reset_index()
        view = pct.sort_values("Campeon", ascending=False).head(15)
        print("Top 15 — P(campeon) actualizada vs pre-torneo (Δ en pp):")
        out = view[["team", "group", "Octavos", "Cuartos", "Semis",
                    "Campeon_pre", "Campeon", "delta_pp"]].copy()
        for c in ["Octavos", "Cuartos", "Semis", "Campeon_pre", "Campeon", "delta_pp"]:
            out[c] = out[c].round(1)
        out = out.rename(columns={"Campeon_pre": "Camp_pre", "delta_pp": "Δ"})
        print(out.to_string(index=False))
        print("\nMayores movimientos en P(campeon) (pp):")
        mov = pct[["team", "delta_pp"]].copy()
        print("  Suben: ", ", ".join(f"{r.team} {r.delta_pp:+.2f}"
              for r in mov.nlargest(5, "delta_pp").itertuples(index=False)))
        print("  Bajan: ", ", ".join(f"{r.team} {r.delta_pp:+.2f}"
              for r in mov.nsmallest(5, "delta_pp").itertuples(index=False)))
    else:
        view = pct.sort_values("Campeon", ascending=False).head(15)
        for s in ["Octavos", "Cuartos", "Semis", "Campeon"]:
            view[s] = view[s].round(1)
        print("Top 15 — P(campeon) (corré simulate.py para la comparacion pre-torneo):")
        print(view[cols].to_string(index=False))

    # Donde el modo live mas se nota: P(avanzar) de los grupos ya en juego.
    if base_path.exists():
        groups_jugando = sorted(played["group"].unique()) if not played.empty else []
        base_r32 = pd.read_csv(base_path).set_index("team")["R32"] * 100
        now_r32 = df.set_index("team")["R32"] * 100
        print("\nP(avanzar de grupo) — grupos con partidos jugados (ahora vs pre):")
        for g in groups_jugando:
            sub = df[df.group == g].sort_values("R32", ascending=False)
            print(f"  Grupo {g}:")
            for r in sub.itertuples(index=False):
                n, pre = r.team, base_r32.get(r.team, float("nan"))
                cur = now_r32.get(r.team, float("nan"))
                print(f"    {n:<24}{pre:5.0f}% -> {cur:5.0f}%  ({cur - pre:+.0f})")

    print(f"\nSuma P(campeon): {df['Campeon'].sum():.3f}")


if __name__ == "__main__":
    main()
