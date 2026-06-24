"""
goals.py — De fuerza Elo a goles, y muestreo de marcadores Dixon-Coles.

Enfoque "supremacia + total" (estandar en casas de apuestas):
    supremacia = slope * dr           (diferencia de goles esperada)
    total      = base_total           (goles totales esperados del partido)
    lambda_local     = (total + supremacia) / 2
    lambda_visitante = (total - supremacia) / 2

Luego el marcador se modela con un Poisson bivariado con la correccion
Dixon-Coles tau(x, y) sobre las 4 celdas de marcador bajo (0-0,1-0,0-1,1-1),
que corrige la subestimacion de empates bajos del Poisson independiente.

Los parametros (slope, base_total, rho) se ajustan en calibrate.py y se
guardan en data/calib.json.
"""
import json
from math import factorial
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
CALIB_PATH = DATA / "calib.json"

MAX_GOALS = 12  # tope de goles por equipo para la matriz de probabilidad
MIN_LAMBDA = 0.15  # piso para lambda (evita lambdas <= 0 en goleadas)

_K = np.arange(MAX_GOALS + 1)
_FACT = np.array([factorial(k) for k in range(MAX_GOALS + 1)], dtype=float)


def _poisson_pmf(lam):
    """pmf de Poisson para k=0..MAX_GOALS (rapido, sin scipy)."""
    return np.exp(-lam) * lam ** _K / _FACT


def load_calib() -> dict:
    with open(CALIB_PATH, encoding="utf-8") as f:
        return json.load(f)


def expected_goals(elo_h, elo_a, neutral, calib, home_adv=100.0):
    """Devuelve (lambda_local, lambda_visitante) a partir de los Elo."""
    ha = 0.0 if neutral else home_adv
    dr = (elo_h + ha) - elo_a
    supremacy = calib["slope"] * dr
    total = calib["base_total"]
    lam_h = max((total + supremacy) / 2.0, MIN_LAMBDA)
    lam_a = max((total - supremacy) / 2.0, MIN_LAMBDA)
    return lam_h, lam_a


def _dc_tau(x, y, lam_h, lam_a, rho):
    """Correccion Dixon-Coles en las celdas (0,0),(0,1),(1,0),(1,1)."""
    if x == 0 and y == 0:
        return 1.0 - lam_h * lam_a * rho
    if x == 0 and y == 1:
        return 1.0 + lam_h * rho
    if x == 1 and y == 0:
        return 1.0 + lam_a * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def score_matrix(lam_h, lam_a, rho):
    """Matriz (MAX_GOALS+1)x(MAX_GOALS+1) de probabilidades de marcador."""
    m = np.outer(_poisson_pmf(lam_h), _poisson_pmf(lam_a))
    # Aplicar tau a las 4 celdas bajas.
    m[0, 0] *= 1.0 - lam_h * lam_a * rho
    m[0, 1] *= 1.0 + lam_h * rho
    m[1, 0] *= 1.0 + lam_a * rho
    m[1, 1] *= 1.0 - rho
    m /= m.sum()
    return m


def outcome_probs(lam_h, lam_a, rho):
    """Probabilidades (P_local, P_empate, P_visitante) de un partido."""
    m = score_matrix(lam_h, lam_a, rho)
    p_home = np.tril(m, -1).sum()  # x > y
    p_draw = np.trace(m)
    p_away = np.triu(m, 1).sum()  # x < y
    return p_home, p_draw, p_away


class ScoreSampler:
    """Muestreador de marcadores via CDF aplanada + searchsorted.

    Cachea la CDF por (lambda_local, lambda_visitante) redondeados, asi miles de
    partidos con la misma fuerza relativa reusan la matriz (gran ahorro).
    """

    def __init__(self, rho, rng=None, cache_round=2):
        self.rho = rho
        self.rng = rng or np.random.default_rng()
        self.cache_round = cache_round
        self._cache = {}

    def _cdf(self, lam_h, lam_a):
        key = (round(lam_h, self.cache_round), round(lam_a, self.cache_round))
        cdf = self._cache.get(key)
        if cdf is None:
            cdf = np.cumsum(score_matrix(*key, self.rho).ravel())
            self._cache[key] = cdf
        return cdf

    def sample(self, lam_h, lam_a):
        """Devuelve (home_goals, away_goals) para un solo partido."""
        cdf = self._cdf(lam_h, lam_a)
        k = int(np.searchsorted(cdf, self.rng.random()))
        return divmod(k, MAX_GOALS + 1)
