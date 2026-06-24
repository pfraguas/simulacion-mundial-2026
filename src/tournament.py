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


# Alargue = 30 min = 1/3 de un partido de 90 -> se escalan las lambdas.
EXTRA_TIME_FRAC = 30.0 / 90.0


def _play_knockout(t_h, t_a, sampler, calib, neutral, rng):
    """Resuelve un partido de eliminatoria y devuelve el ganador.

    90' (Dixon-Coles) -> si empata, alargue de 30' (1/3 de partido) -> si sigue
    empatado, penales 50/50 (sin sesgo: la tanda es practicamente azar).
    """
    lam_h, lam_a = expected_goals(t_h.elo, t_a.elo, neutral, calib, calib["home_adv"])
    hs, as_ = sampler.sample(lam_h, lam_a)
    if hs > as_:
        return t_h
    if as_ > hs:
        return t_a
    # Empate en 90' -> alargue: 1/3 de la tasa de goles del partido.
    eh, ea = sampler.sample(lam_h * EXTRA_TIME_FRAC, lam_a * EXTRA_TIME_FRAC)
    if eh > ea:
        return t_h
    if ea > eh:
        return t_a
    # Sigue empatado -> penales 50/50.
    return t_h if rng.random() < 0.5 else t_a


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


def _h2h_stats(names, matches):
    """Mini-tabla (pts, dg, gf) contando solo los partidos ENTRE 'names'."""
    s = set(names)
    stats = {n: [0, 0, 0] for n in names}  # [pts, dg, gf]
    for home, away, hs, as_ in matches:
        if home in s and away in s:
            stats[home][1] += hs - as_
            stats[home][2] += hs
            stats[away][1] += as_ - hs
            stats[away][2] += as_
            if hs > as_:
                stats[home][0] += 3
            elif as_ > hs:
                stats[away][0] += 3
            else:
                stats[home][0] += 1
                stats[away][0] += 1
    return stats


def rank_group(teams, matches, rng):
    """Ordena un grupo con los desempates oficiales FIFA 2026.

    1) puntos  2-4) head-to-head entre empatados (pts, dg, gf)
    5) dg general  6) gf general  7) sorteo (proxy de fair-play / ranking / lots).

    El head-to-head se calcula dentro de cada bloque de equipos igualados en
    puntos. La re-aplicacion recursiva de FIFA cuando un subconjunto se separa se
    simplifica (caso de borde poco frecuente).
    """
    by_pts = {}
    for t in teams:
        by_pts.setdefault(t.pts, []).append(t.name)

    h2h = {}
    for pts, names in by_pts.items():
        if len(names) > 1:  # solo hay empate que romper si son 2+ en ese puntaje
            stats = _h2h_stats(names, matches)
            h2h.update(stats)
        else:
            h2h[names[0]] = [0, 0, 0]

    rnd = {t.name: rng.random() for t in teams}

    def key(t):
        hp, hgd, hgf = h2h[t.name]
        return (t.pts, hp, hgd, hgf, t.gd, t.gf, rnd[t.name])

    return sorted(teams, key=key, reverse=True)


def simulate_group(teams, sampler, calib, rng, known=None):
    """Round-robin de un grupo de 4; devuelve la lista ordenada por posicion.

    known: dict {frozenset({home,away}): (home, away, hs, as_)} con los partidos
    ya jugados. Esos no se simulan: se usa el marcador real.
    """
    by_name = {t.name: t for t in teams}
    matches = []  # (home, away, hs, as_) -> necesario para el head-to-head
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            h, a = teams[i], teams[j]
            key = frozenset((h.name, a.name))
            if known and key in known:
                home, away, hs, as_ = known[key]
            else:
                neutral = _is_neutral(h, a)
                lam_h, lam_a = expected_goals(h.elo, a.elo, neutral, calib, calib["home_adv"])
                hs, as_ = sampler.sample(lam_h, lam_a)
                home, away = h.name, a.name
            _apply_known(by_name, home, away, hs, as_)
            matches.append((home, away, hs, as_))
    return rank_group(teams, matches, rng)


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
            w = _play_knockout(h, a, sampler, calib, neutral, rng)
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
