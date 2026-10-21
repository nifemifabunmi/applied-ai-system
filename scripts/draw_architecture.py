"""Render diagrams/architecture.png (and .svg) with matplotlib.  Usage: python -m scripts.draw_architecture"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
INK, MUTED, BG = "#16213e", "#566077", "#ffffff"
COLORS = {"input": "#e8ecf6", "core": "#d9f0ec", "ai": "#ece6fb", "guard": "#fbe9d0", "out": "#ffffff", "evid": "#eef1f5"}


def box(ax, x, y, w, h, title, body, kind):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=COLORS[kind], ec=INK, lw=1.3))
    ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=11.5, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.72, body, ha="center", va="top", fontsize=8.8, color=MUTED, linespacing=1.35)
    return (x, y, w, h)


def arrow(ax, a, b, text=None, rad=0.0, color=INK, style="-|>", dy=0.0, dx=0.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((a[0] + b[0]) / 2 + dx, (a[1] + b[1]) / 2 + dy, text, ha="center", va="center", fontsize=8.2,
                color=color, bbox=dict(fc=BG, ec="none", pad=1.2))


def main():
    fig, ax = plt.subplots(figsize=(17.3, 10.4))
    ax.set_xlim(0, 17.3)
    ax.set_ylim(0, 10.4)
    ax.axis("off")
    ax.text(0.2, 10.15, "VibeFinder architecture (v5)", fontsize=17, fontweight="bold", color=INK, va="center")
    ax.text(0.2, 9.72, "Rules and similarity pick the songs; AI text, if enabled, is only shown when it matches the score.",
            fontsize=10.5, color=MUTED, va="center")

    # inputs
    taste = box(ax, 0.3, 7.7, 4.3, 1.55, "Your taste", "genre, mood, energy, acoustic preference,\noptional sound targets", "input")
    liked = box(ax, 5.0, 7.7, 4.3, 1.55, "Songs you already like (optional)", "never recommended back;\nshape a taste vector from their sound", "input")
    cat = box(ax, 9.7, 7.7, 5.9, 1.55, "Catalog", "50-song demo, or 1,985 songs from Spotify audio features\n(mood derived from energy and valence)", "input")

    # core pipeline
    y = 5.15
    retr = box(ax, 0.3, y, 3.4, 1.9, "1. Retrieval", "genre AND mood, then OR,\nthen whole catalog\n(skipped when songs are liked)", "core")
    score = box(ax, 4.0, y, 4.1, 1.9, "2. Scoring", "genre, mood, energy, acoustic,\nsound targets, similarity to liked songs\n= points per factor", "core")
    div = box(ax, 8.4, y, 3.1, 1.9, "3. Diversity ranking", "artist-repeat penalty\nduring selection,\ntop k songs", "core")
    expl = box(ax, 11.8, y, 3.8, 1.9, "4. Rule-based explanation", "plain-language reasons\nand similar songs", "core")

    # AI path
    llm = box(ax, 8.4, 2.75, 3.1, 1.75, "AI writer (optional)", "Claude writes 1 to 2 sentences\nfrom this song's facts only", "ai")
    guard = box(ax, 11.8, 2.75, 3.8, 1.75, "Guardrail", "checks genre/mood words, numbers,\ntitles, artists and claims\nagainst the score breakdown", "guard")

    # outputs
    out = box(ax, 8.4, 0.85, 7.2, 1.3, "Result cards", "score bar by factor, explanation: AI text if it passes, otherwise the rule-based text.\nShown in the Streamlit app (with AI) and the static page (rules only).", "out")

    # evidence band
    items = [
        ("Baselines and ablation", "random, energy, genre,\nkNN; remove one part at a time"),
        ("Weight tuning", "held-out validation,\nsensitivity sweeps"),
        ("Synthetic listeners", "genre taste, audio taste;\n5 MUSIC-model archetypes"),
        ("Real listeners", "opt-in taste files\nfrom the app"),
        ("Guardrail suite", "labelled good and bad\nexplanations"),
    ]
    ex, ey, ew, eh = 0.3, 0.3, 7.5, 3.75
    ax.add_patch(FancyBboxPatch((ex, ey), ew, eh, boxstyle="round,pad=0.02,rounding_size=0.12", fc=COLORS["evid"], ec=INK, lw=1.3, ls="--"))
    ax.text(ex + 0.25, ey + eh - 0.3, "Evidence (experiments/ and reports/)", fontsize=11.5, fontweight="bold", color=INK, va="center")
    for i, (t, b) in enumerate(items):
        yy = ey + eh - 0.95 - i * 0.62
        ax.text(ex + 0.3, yy, t, fontsize=9.6, fontweight="bold", color=INK, va="center")
        ax.text(ex + 2.55, yy, b.replace("\n", " "), fontsize=8.6, color=MUTED, va="center")

    # arrows
    arrow(ax, (2.4, 7.7), (2.0, 7.05))
    arrow(ax, (7.1, 7.7), (6.4, 7.05))
    arrow(ax, (12.6, 7.7), (6.9, 7.05), rad=-0.05)
    arrow(ax, (3.7, 6.1), (4.0, 6.1))
    arrow(ax, (8.1, 6.1), (8.4, 6.1))
    arrow(ax, (11.5, 6.1), (11.8, 6.1))
    # picks -> AI writer (facts), writer -> guardrail, guardrail -> cards
    arrow(ax, (9.95, 5.15), (9.95, 4.5))
    ax.text(10.1, 4.82, "facts of each pick", fontsize=8.2, color=MUTED, va="center")
    arrow(ax, (11.5, 3.62), (11.8, 3.62))
    arrow(ax, (13.7, 2.75), (13.7, 2.15), color="#1f7a5a")
    ax.text(13.85, 2.45, "passes: show AI text", fontsize=8.2, color="#1f7a5a", va="center")
    # rule-based text is the fallback, and what the static page always shows
    arrow(ax, (15.6, 6.1), (16.6, 6.1), style="-")
    arrow(ax, (16.6, 6.1), (16.6, 1.5), style="-", color="#a65e00")
    arrow(ax, (16.6, 1.5), (15.6, 1.5), color="#a65e00")
    ax.text(16.75, 3.8, "fails, or AI off:\nrule-based text", fontsize=8.2, color="#a65e00", va="center", ha="left", rotation=0)

    for ext in ("png", "svg"):
        fig.savefig(ROOT / f"diagrams/architecture.{ext}", dpi=130, bbox_inches="tight", facecolor=BG)
    print("wrote diagrams/architecture.png and .svg")


if __name__ == "__main__":
    main()
