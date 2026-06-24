"""
elo.py — Motor de ratings Elo (World Football Elo Ratings).

Modulo compartido por build_data.py (calcular Elo final de las selecciones) y
calibrate.py (recolectar snapshots pre-partido de dr -> goles).

La diferencia de Elo `dr = (elo_local + ventaja_localia) - elo_visitante`
predice el resultado esperado via la curva logistica estandar.
"""
from collections import defaultdict

INIT_ELO = 1500.0
HOME_ADV = 100.0  # puntos Elo de ventaja de localia (0 en cancha neutral)

# Importancia del partido -> K base (valores estandar de eloratings.net).
K_BY_TOURNAMENT = {
    "FIFA World Cup": 60.0,
    "Copa America": 50.0,
    "UEFA Euro": 50.0,
    "African Cup of Nations": 50.0,
    "AFC Asian Cup": 50.0,
    "Gold Cup": 50.0,
    "Confederations Cup": 40.0,
    "UEFA Nations League": 40.0,
    "FIFA World Cup qualification": 40.0,
    "UEFA Euro qualification": 40.0,
    "Copa America qualification": 40.0,
    "African Cup of Nations qualification": 40.0,
    "AFC Asian Cup qualification": 40.0,
    "CONCACAF Nations League": 40.0,
    "Friendly": 20.0,
}
K_DEFAULT = 30.0


def goal_diff_multiplier(gd: int) -> float:
    """Multiplicador G por diferencia de goles (formula eloratings)."""
    gd = abs(gd)
    if gd <= 1:
        return 1.0
    if gd == 2:
        return 1.5
    return (11.0 + gd) / 8.0


def expected_score(dr: float) -> float:
    """We del local dado dr (diferencia de Elo, localia ya incluida)."""
    return 1.0 / (10.0 ** (-dr / 400.0) + 1.0)


def replay(played, collect_since=None):
    """
    Recorre los partidos (DataFrame ordenado por fecha) actualizando Elo.

    Devuelve (elo, n_matches, last_date, snapshots).
    snapshots: lista de dicts (dr, home_score, away_score, date) para partidos
    con fecha >= collect_since (None = no recolectar). Sirve para calibrar.
    """
    elo = defaultdict(lambda: INIT_ELO)
    n_matches = defaultdict(int)
    last_date = {}
    snapshots = []

    for r in played.itertuples(index=False):
        h, a = r.home_team, r.away_team
        hs, as_ = r.home_score, r.away_score
        ha = 0.0 if bool(r.neutral) else HOME_ADV
        dr = (elo[h] + ha) - elo[a]

        if collect_since is not None and r.date >= collect_since:
            snapshots.append(
                {"dr": dr, "home_score": hs, "away_score": as_, "date": r.date}
            )

        we = expected_score(dr)
        w = 1.0 if hs > as_ else (0.0 if hs < as_ else 0.5)
        k = K_BY_TOURNAMENT.get(r.tournament, K_DEFAULT) * goal_diff_multiplier(hs - as_)
        delta = k * (w - we)
        elo[h] += delta
        elo[a] -= delta

        for t in (h, a):
            n_matches[t] += 1
            last_date[t] = r.date

    return elo, n_matches, last_date, snapshots


def load_played(results_csv):
    """Carga results.csv y devuelve solo partidos jugados, ordenados por fecha."""
    import pandas as pd

    df = pd.read_csv(results_csv)
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(str).str.upper().eq("TRUE")
    played = df.dropna(subset=["home_score", "away_score"]).copy()
    played["home_score"] = played["home_score"].astype(int)
    played["away_score"] = played["away_score"].astype(int)
    return played.sort_values("date").reset_index(drop=True)
