"""
tournament.py — Simula un Mundial 2026 completo (formato 48 equipos).

Fase de grupos: 12 grupos de 4, todos contra todos (3 pts victoria, 1 empate).
Clasifican: 1ro y 2do de cada grupo (24) + los 8 mejores 3ros (32 total).
Eliminatoria: bracket de 32 -> 16 -> 8 -> 4 -> final.

Desempates en grupo (aprox. reglas FIFA): puntos, diferencia de gol, goles a
favor, y luego sorteo (aleatorio). El head-to-head exacto se omite en v1.

Bracket de eliminacion (v1, aproximacion documentada): se siembra a los 32
clasificados por desempenio (ganadores de grupo > segundos > terceros; dentro de
cada nivel por pts/dg/gf) y se arma un cuadro sembrado estandar de 32 que
mantiene separados a los mejores y evita revanchas del mismo grupo en la primera
ronda. El mapeo posicional oficial FIFA de los 8 terceros queda como refinamiento.
"""
import numpy as np

from goals import expected_goals

# Orden de siembra de un bracket de 32 (posiciones que solo cruzan 1 vs 2 en la
# final). seed_order[i] = semilla (1-indexada) que va en la ranura i.
_SEED_ORDER_32 = [
    1, 32, 16, 17, 8, 25, 9, 24, 4, 29, 13, 20, 5, 28, 12, 21,
    2, 31, 15, 18, 7, 26, 10, 23, 3, 30, 14, 19, 6, 27, 11, 22,
]

ROUND_NAMES = {32: "R32", 16: "Octavos", 8: "Cuartos", 4: "Semis", 2: "Final"}


class TeamState:
    """Estado de un equipo dentro de una simulacion de torneo."""

    __slots__ = ("name", "elo", "group", "pts", "gf", "ga", "rnd_seed")

    def __init__(self, name, elo, group):
        self.name = name
        self.elo = elo
        self.group = group
        self.pts = self.gf = self.ga = 0
        self.rnd_seed = 0

    @property
    def gd(self):
        return self.gf - self.ga


def _play(t_h, t_a, sampler, calib, neutral, knockout, rng):
    """Juega un partido, actualiza gf/ga de ambos y devuelve el ganador (KO)."""
    lam_h, lam_a = expected_goals(t_h.elo, t_a.elo, neutral, calib, calib["home_adv"])
    hs, as_ = sampler.sample(lam_h, lam_a)
    t_h.gf += hs
    t_h.ga += as_
    t_a.gf += as_
    t_a.ga += hs
    if not knockout:
        if hs > as_:
            t_h.pts += 3
        elif hs < as_:
            t_a.pts += 3
        else:
            t_h.pts += 1
            t_a.pts += 1
        return None
    if hs > as_:
        return t_h
    if as_ > hs:
        return t_a
    # Empate -> penales (leve sesgo por Elo).
    p_home = 1.0 / (10.0 ** (-(t_h.elo - t_a.elo) / 400.0) + 1.0)
    return t_h if rng.random() < p_home else t_a


# Sedes con ventaja de localia (anfitriones). El resto se juega en cancha neutral.
HOSTS = {"United States", "Mexico", "Canada"}


def _is_neutral(t_h, t_a):
    return t_h.name not in HOSTS


def _apply_known(by_name, home, away, hs, as_):
    """Aplica un marcador real ya jugado a la tabla del grupo."""
    th, ta = by_name[home], by_name[away]
    th.gf += hs
    th.ga += as_
    ta.gf += as_
    ta.ga += hs
    if hs > as_:
        th.pts += 3
    elif hs < as_:
        ta.pts += 3
    else:
        th.pts += 1
        ta.pts += 1


def simulate_group(teams, sampler, calib, rng, known=None):
    """Round-robin de un grupo de 4; devuelve la lista ordenada por posicion.

    known: dict {frozenset({home,away}): (home, away, hs, as_)} con los partidos
    ya jugados. Esos no se simulan: se usa el marcador real.
    """
    by_name = {t.name: t for t in teams}
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            h, a = teams[i], teams[j]
            key = frozenset((h.name, a.name))
            if known and key in known:
                _apply_known(by_name, *known[key])
            else:
                _play(h, a, sampler, calib, _is_neutral(h, a), knockout=False, rng=rng)
    # Orden: pts, dg, gf, desempate aleatorio.
    rank_key = [(t.pts, t.gd, t.gf, rng.random()) for t in teams]
    order = sorted(range(len(teams)), key=lambda k: rank_key[k], reverse=True)
    return [teams[k] for k in order]


def _seed_score(t, place):
    """Puntaje de siembra: nivel (1ro/2do/3ro) primero, luego pts/dg/gf."""
    tier = {1: 2_000_000, 2: 1_000_000, 3: 0}[place]
    return tier + t.pts * 10_000 + t.gd * 100 + t.gf


def simulate_tournament(team_defs, sampler, calib, rng, on_group_result=None,
                        known_results=None):
    """
    team_defs: lista de (name, elo, group).
    on_group_result: callback opcional (kwargs: name, group, place, pts, gd, gf,
        advanced) invocado una vez por equipo tras la fase de grupos. Sirve para
        analisis condicional (p. ej. P(avanzar | puntos)). None = sin overhead.
    known_results: dict {frozenset({home,away}): (home, away, hs, as_)} con
        partidos de grupo ya jugados; se fijan en vez de simularse (modo live).
    Devuelve dict: para cada equipo la ronda mas lejana alcanzada
    ('group','R32','Octavos','Cuartos','Semis','Final','Campeon').
    """
    # --- Fase de grupos ---
    by_group = {}
    for name, elo, group in team_defs:
        by_group.setdefault(group, []).append(TeamState(name, elo, group))

    reached = {t[0]: "group" for t in team_defs}
    standings = {}  # grupo -> [1ro, 2do, 3ro, 4to]
    thirds = []
    for g, teams in by_group.items():
        ordered = simulate_group(teams, sampler, calib, rng, known=known_results)
        standings[g] = ordered
        reached[ordered[0].name] = reached[ordered[1].name] = "R32"
        thirds.append(ordered[2])

    # 8 mejores terceros.
    thirds.sort(key=lambda t: (t.pts, t.gd, t.gf, rng.random()), reverse=True)
    best_thirds = thirds[:8]
    for t in best_thirds:
        reached[t.name] = "R32"

    # Hook opcional: registrar resultados de fase de grupos (para analisis).
    if on_group_result is not None:
        advanced = {t.name for t in best_thirds}
        for g, ordered in standings.items():
            advanced.add(ordered[0].name)
            advanced.add(ordered[1].name)
        for g, ordered in standings.items():
            for place, t in enumerate(ordered, start=1):
                on_group_result(
                    name=t.name, group=g, place=place, pts=t.pts,
                    gd=t.gd, gf=t.gf, advanced=t.name in advanced,
                )

    # --- Siembra de los 32 clasificados ---
    qualified = []
    for g, ordered in standings.items():
        qualified.append((ordered[0], 1))
        qualified.append((ordered[1], 2))
    for t in best_thirds:
        qualified.append((t, 3))

    qualified.sort(key=lambda tp: _seed_score(tp[0], tp[1]), reverse=True)
    for seed, (t, _place) in enumerate(qualified, start=1):
        t.rnd_seed = seed
    seeded = {t.rnd_seed: t for t, _ in qualified}

    # Cuadro inicial (32) en orden de ranura. Evitar revancha del mismo grupo en R32.
    bracket = [seeded[s] for s in _SEED_ORDER_32]
    for i in range(0, 32, 2):
        if bracket[i].group == bracket[i + 1].group:
            # intercambiar con el rival de la llave siguiente
            j = i + 3 if i + 3 < 32 else i - 1
            bracket[i + 1], bracket[j] = bracket[j], bracket[i + 1]

    # --- Eliminatoria ---
    champion = None
    while len(bracket) > 1:
        round_size = len(bracket)
        winners = []
        for i in range(0, round_size, 2):
            h, a = bracket[i], bracket[i + 1]
            # localia solo si el anfitrion juega; bracket no tiene "local" fijo,
            # asi que se considera neutral salvo que uno sea anfitrion.
            neutral = not (h.name in HOSTS or a.name in HOSTS)
            # si el anfitrion es 'a', ponerlo de local para la ventaja
            if a.name in HOSTS and h.name not in HOSTS:
                h, a = a, h
            w = _play(h, a, sampler, calib, neutral, knockout=True, rng=rng)
            winners.append(w)
        next_size = round_size // 2
        label = ROUND_NAMES.get(next_size)
        if label:  # marcar a los que avanzan a la proxima ronda
            for w in winners:
                reached[w.name] = label
        bracket = winners
        if next_size == 1:
            champion = bracket[0]

    reached[champion.name] = "Campeon"
    return reached
