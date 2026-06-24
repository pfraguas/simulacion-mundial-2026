# Simulación Monte Carlo — Mundial 2026

Estima las probabilidades de cada selección (avanzar de fase, llegar a cada ronda,
ser campeón) simulando el torneo completo decenas de miles de veces.

## Modelo

```
Elo (fuerza del equipo)  ->  goles esperados λ  ->  marcador Dixon-Coles  ->  Monte Carlo
```

1. **Elo (`elo.py`)** — se calcula un rating Elo para cada selección recorriendo el
   histórico completo de partidos internacionales (1872–2026) con el método *World
   Football Elo Ratings* (K por importancia del torneo, multiplicador por diferencia
   de goles, ventaja de localía). Es auto-contenido: no depende de un proveedor externo.
2. **Elo → goles (`goals.py`)** — enfoque *supremacía + total*: la diferencia de Elo
   `dr` da la diferencia de goles esperada (`supremacy = slope·dr`) y un total base de
   goles; de ahí salen `λ_local` y `λ_visitante`.
3. **Marcador Dixon-Coles** — Poisson bivariado con la corrección τ sobre las celdas de
   marcador bajo (0-0, 1-0, 0-1, 1-1), que corrige la subestimación de empates bajos del
   Poisson independiente. Parámetros (`slope`, `base_total`, `rho`) calibrados por máxima
   verosimilitud sobre el histórico (`calibrate.py`).
4. **Monte Carlo (`simulate.py`)** — simula 104 partidos (72 de grupos + 32 de
   eliminatoria) N veces y agrega frecuencias → probabilidades.

### ¿Por qué Elo y no xG?

El **xG** es excelente para fútbol de clubes (datos de tiros, muestras grandes) pero
**poco confiable para selecciones**: juegan pocos partidos al año, los planteles rotan y
la muestra es chica y ruidosa. Los modelos serios del Mundial (FiveThirtyEight SPI, PELE
de Nate Silver) usan un rating de fuerza tipo Elo alimentando un modelo Poisson/Dixon-Coles.
El xG queda como posible señal secundaria a futuro (ver *Limitaciones*).

## Datos

- `data/results.csv` — histórico de partidos internacionales
  ([martj42/international_results](https://github.com/martj42/international_results)).
  Incluye los fixtures del Mundial 2026.
- `data/elo_ratings.csv` — Elo de las 48 selecciones (generado por `build_data.py`).
- `data/groups.json` — los 12 grupos, **derivados automáticamente de los fixtures
  reales** (en un grupo de 4 los 4 equipos juegan entre sí → componentes conexas).
- `data/calib.json` — parámetros calibrados del modelo de goles.

## Uso

```bash
pip install -r requirements.txt

python src/build_data.py      # genera elo_ratings.csv y groups.json
python src/calibrate.py       # genera calib.json
python src/simulate.py 50000  # corre la simulación -> outputs/probabilidades.csv
python src/report.py          # gráfico outputs/campeon.png + tabla por grupos
python src/verify.py          # chequeos de sanidad y convergencia
```

`src/simulate.py [N] [--seed S]` — N simulaciones (default 50.000). 50k corren en ~12 s.

### Análisis adicionales

```bash
python src/analysis.py 50000 --team "Mexico"   # P(avanzar | puntos de grupo)
python src/home_advantage.py 50000 --sweep     # valor de la localía (con vs sin)
python src/live.py 50000                        # simulación in-tournament
```

- **`analysis.py`** — probabilidad de avanzar según los puntos sumados en grupos.
- **`home_advantage.py`** — corre el torneo con y sin localía para aislar su efecto.
- **`live.py`** — modo *in-tournament*: fija los resultados de grupo ya jugados
  (los lee de `results.csv`) y simula sólo lo que falta, actualizando las
  probabilidades a medida que avanza el Mundial. Compara contra la línea de base
  pre-torneo (`outputs/probabilidades.csv`). Sólo fija resultados de **fase de
  grupos**; el conteo de eliminatoria queda como refinamiento.
  - **Auto-actualiza los datos**: al correrlo descarga el `results.csv` más
    reciente de martj42 y reconstruye Elo/grupos antes de simular. Si no hay red,
    avisa y sigue con la copia local. Usá `--no-update` para forzar la copia local
    (offline o reproducibilidad).

## Verificación

- Goles/partido del modelo ≈ histórico (~2.74); tasa de empates ≈ 0.23 (confirma que
  Dixon-Coles actúa).
- `Σ P(campeón) = 1.000`; los favoritos por Elo encabezan.
- Convergencia: con semillas distintas a 20k, la diferencia máxima en P(campeón) < 0.6 pp.

## Supuestos y limitaciones (v1)

- **Bracket de eliminatoria**: se siembra a los 32 clasificados por desempeño (ganadores
  de grupo > segundos > terceros) en un cuadro sembrado estándar de 32 que separa a los
  mejores y evita revanchas del mismo grupo en la primera ronda. El **mapeo posicional
  oficial FIFA** de los 8 mejores terceros queda como refinamiento futuro.
- **Desempates de grupo**: puntos → diferencia de gol → goles a favor → sorteo. No se
  aplica el head-to-head exacto.
- **Localía**: ventaja Elo (+100) sólo para los anfitriones (EE.UU., México, Canadá);
  el resto se juega en cancha neutral.
- **Supremacía lineal**: a diferencias de Elo muy grandes el λ del débil se satura en un
  piso; razonable pero simplificado.

## Extensiones futuras

- xG como señal secundaria de ajuste ataque/defensa donde haya datos confiables.
- Mapeo oficial exacto del bracket de los 8 mejores terceros.
- Modelo bayesiano jerárquico (ataque/defensa por equipo, ponderación temporal).
- Actualización *in-tournament*: fijar partidos jugados y re-simular sólo lo restante.
