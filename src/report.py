"""
report.py — Visualiza los resultados de la simulacion.

Lee outputs/probabilidades.csv y produce:
  - outputs/campeon.png        grafico de barras P(Campeon) top 20
  - resumen por consola: probabilidad de avanzar de grupo por grupo.

Uso:
    python src/report.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
CSV = OUT / "probabilidades.csv"


def main():
    df = pd.read_csv(CSV)

    # --- Grafico de barras: top 20 candidatos al titulo ---
    top = df.nlargest(20, "Campeon").iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(top["team"], top["Campeon"] * 100, color="#2c7fb8")
    for y, v in enumerate(top["Campeon"] * 100):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Probabilidad de ser campeon (%)")
    ax.set_title("Mundial 2026 — Monte Carlo (Elo + Dixon-Coles)")
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(OUT / "campeon.png", dpi=130)
    print(f"Grafico -> {OUT / 'campeon.png'}")

    # --- Probabilidad de avanzar de fase de grupos, por grupo ---
    print("\nProbabilidad de avanzar de la fase de grupos (R32), por grupo:")
    for g, sub in df.groupby("group"):
        sub = sub.sort_values("R32", ascending=False)
        line = "  ".join(f"{r.team} {r.R32 * 100:4.0f}%" for r in sub.itertuples())
        print(f"  Grupo {g}: {line}")

    # --- Tabla resumen top 10 ---
    print("\nTop 10 (probabilidades %):")
    pct = df.nlargest(10, "Campeon").copy()
    for s in ["Octavos", "Cuartos", "Semis", "Final", "Campeon"]:
        pct[s] = (pct[s] * 100).round(1)
    print(pct[["team", "group", "Octavos", "Cuartos", "Semis", "Final", "Campeon"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
