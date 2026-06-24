"""
calibrate.py — Ajusta los parametros del modelo de goles sobre el historico.

Recolecta snapshots pre-partido (dr, marcador) recorriendo el Elo, y estima:
  - slope:      pendiente de  diferencia_de_goles ~ dr  (sin intercepto, porque
                dr ya incluye la localia: a dr=0 la diferencia esperada es 0).
  - base_total: promedio de goles totales por partido.
  - rho:        parametro Dixon-Coles que maximiza la verosimilitud de los
                marcadores observados (busqueda en grilla).

Se ponderan los partidos por recencia (half-life configurable) para que la
forma reciente pese mas que la antigua.

Salida: data/calib.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from elo import load_played, replay
from goals import MIN_LAMBDA, _dc_tau
from scipy.stats import poisson

DATA = Path(__file__).resolve().parent.parent / "data"
RESULTS_CSV = DATA / "results.csv"

COLLECT_SINCE = pd.Timestamp("2006-01-01")  # ventana de calibracion
HALF_LIFE_DAYS = 365.0 * 4  # la forma de hace 4 anios pesa la mitad


def main():
    played = load_played(RESULTS_CSV)
    _, _, _, snaps = replay(played, collect_since=COLLECT_SINCE)
    snaps = [s for s in snaps if s["date"] >= COLLECT_SINCE]
    df = pd.DataFrame(snaps)

    # Pesos por recencia.
    age_days = (df["date"].max() - df["date"]).dt.days.to_numpy()
    w = 0.5 ** (age_days / HALF_LIFE_DAYS)

    dr = df["dr"].to_numpy()
    hs = df["home_score"].to_numpy()
    as_ = df["away_score"].to_numpy()

    # slope: regresion ponderada de (hs - as_) ~ dr, sin intercepto.
    diff = hs - as_
    slope = float(np.sum(w * dr * diff) / np.sum(w * dr * dr))

    # base_total: promedio ponderado de goles totales.
    base_total = float(np.sum(w * (hs + as_)) / np.sum(w))

    # rho: busqueda en grilla maximizando la log-verosimilitud Dixon-Coles.
    supremacy = slope * dr
    lam_h = np.clip((base_total + supremacy) / 2.0, MIN_LAMBDA, None)
    lam_a = np.clip((base_total - supremacy) / 2.0, MIN_LAMBDA, None)

    base_ll = poisson.logpmf(hs, lam_h) + poisson.logpmf(as_, lam_a)
    low = (hs <= 1) & (as_ <= 1)  # tau != 1 solo en celdas bajas

    def weighted_ll(rho):
        tau = np.ones_like(lam_h)
        for i in np.where(low)[0]:
            tau[i] = _dc_tau(int(hs[i]), int(as_[i]), lam_h[i], lam_a[i], rho)
        if np.any(tau <= 0):
            return -np.inf
        return float(np.sum(w * (base_ll + np.log(tau))))

    grid = np.arange(-0.30, 0.30 + 1e-9, 0.01)
    rho = float(max(grid, key=weighted_ll))

    calib = {
        "slope": round(slope, 6),
        "base_total": round(base_total, 4),
        "rho": round(rho, 3),
        "home_adv": 100.0,
        "n_matches": int(len(df)),
        "window_since": COLLECT_SINCE.date().isoformat(),
    }
    with open(DATA / "calib.json", "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2)

    # Diagnostico: comparar tasas observadas vs modeladas.
    obs_total = float(np.sum(w * (hs + as_)) / np.sum(w))
    obs_draw = float(np.sum(w * (hs == as_)) / np.sum(w))
    print("Calibracion:")
    for k, v in calib.items():
        print(f"  {k}: {v}")
    print(f"\n  goles/partido observados (pond.): {obs_total:.3f}")
    print(f"  empates observados (pond.):       {obs_draw:.3f}")


if __name__ == "__main__":
    main()
