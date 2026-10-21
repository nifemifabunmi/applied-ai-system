"""
v2 experiment: are the hand-picked weights any good, and how fragile are they?

  1. Tuning: random search over the four weights that matter for a basic
     profile (genre, mood, energy, acoustic), validated by leave-one-listener-out.
  2. Sensitivity: sweep each weight alone, and jointly perturb all of them.
  3. Output: reports/v2_results.md and reports/v2_sensitivity.png

Usage:  python -m experiments.run_v2
"""
import random
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.run_v1 import K, load_listeners, patched, score
from src import recommender as rec
from src.metrics import bootstrap_ci
from src.recommender import load_songs, recommend_songs
from src.tuning import (TUNED_KEYS, best_config_index, distance_from_default,
                        leave_one_listener_out, sample_configs)

ROOT = Path(__file__).resolve().parent.parent
N_CONFIGS = 800
SEED = 7
SWEEP_FACTORS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
PERTURB_TRIALS = 300
PERTURB_SPREAD = 0.3   # each weight multiplied by a random factor in [0.7, 1.3]


def ndcg_for(weights, listener):
    with patched(weights=weights):
        ids = [s["id"] for s, _, _ in recommend_songs(listener["profile"], listener["catalog"], k=K)]
    return score(ids, listener["relevant"])["ndcg"]


def mean_ndcg(weights, listeners):
    return statistics.mean(ndcg_for(weights, l) for l in listeners)


def safe_range(points, baseline, tolerance=0.03):
    """Contiguous span of multipliers around 1x whose score stays within `tolerance` of the baseline."""
    ok = [abs(v - baseline) <= tolerance or v > baseline for _, v in points]
    i = [f for f, _ in points].index(1.0)
    lo = hi = i
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    while hi < len(points) - 1 and ok[hi + 1]:
        hi += 1
    return f"{points[lo][0]:g}x to {points[hi][0]:g}x"


def pct(x):
    return f"{x:+.2f}"


def run():
    songs = load_songs(str(ROOT / "data/songs.csv"))
    listeners = load_listeners(songs)
    default = {k: rec.WEIGHTS[k] for k in TUNED_KEYS}
    rng = random.Random(SEED)

    # ---- 1. tuning ---------------------------------------------------------
    configs = sample_configs(default, N_CONFIGS, rng)
    matrix = [[ndcg_for(c, l) for l in listeners] for c in configs]
    all_idx = list(range(len(listeners)))
    best_all = best_config_index(matrix, configs, default, all_idx)
    in_sample = {
        "default": statistics.mean(matrix[0]),
        "tuned": statistics.mean(matrix[best_all]),
        "config": configs[best_all],
    }
    folds = leave_one_listener_out(matrix, configs, default)
    diffs = [f["tuned"] - f["default"] for f in folds]
    ci = bootstrap_ci(diffs)
    cv = {
        "default": statistics.mean(f["default"] for f in folds),
        "tuned": statistics.mean(f["tuned"] for f in folds),
        "diff_ci": ci,
        "folds": folds,
    }
    ties_at_top = sum(1 for c in range(len(configs))
                      if abs(statistics.mean(matrix[c]) - in_sample["tuned"]) < 1e-9)

    # ---- 2. sensitivity ------------------------------------------------------
    sweeps = {}
    for key in TUNED_KEYS:
        sweeps[key] = [(f, mean_ndcg({key: default[key] * f}, listeners)) for f in SWEEP_FACTORS]
    prng = random.Random(SEED + 1)
    perturbed = [
        mean_ndcg({k: default[k] * prng.uniform(1 - PERTURB_SPREAD, 1 + PERTURB_SPREAD) for k in TUNED_KEYS},
                  listeners)
        for _ in range(PERTURB_TRIALS)
    ]
    baseline = statistics.mean(matrix[0])
    return {"listeners": listeners, "default": default, "configs": configs, "matrix": matrix,
            "in_sample": in_sample, "cv": cv, "sweeps": sweeps, "perturbed": perturbed,
            "baseline": baseline, "ties_at_top": ties_at_top}


def plot(res, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [3, 2]})
    colors = {"genre": "#1f9a8a", "mood": "#e0a030", "energy": "#5b5fef", "acoustic": "#d9567b"}
    ax = axes[0]
    for key, pts in res["sweeps"].items():
        ax.plot([f for f, _ in pts], [v for _, v in pts], marker="o", ms=4, label=key, color=colors[key])
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.set_xlabel("weight as a multiple of its current value")
    ax.set_ylabel("mean NDCG@5 (7 listeners)")
    ax.set_title("One weight at a time")
    ax.legend(frameon=False)
    ax2 = axes[1]
    ax2.hist(res["perturbed"], bins=12, color="#8a94a8")
    ax2.axvline(res["baseline"], color="#16213e", lw=2, label="current weights")
    ax2.set_xlabel("mean NDCG@5")
    ax2.set_title(f"All weights jittered by ±{int(PERTURB_SPREAD * 100)}%")
    ax2.legend(frameon=False)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_report(res):
    L = res["listeners"]
    d = res["default"]
    L_out = []
    w = L_out.append
    w(f"# v2 results: weight tuning and sensitivity (k={K}, {len(L)} listeners)\n")
    w("Generated by `python -m experiments.run_v2`. Metric: mean NDCG@5 across listeners.\n")

    w("## 1. Does tuning the weights help?\n")
    w(f"Random search over {N_CONFIGS} weight settings (each of genre, mood, energy and acoustic scaled "
      "between 0.25x and 4x). Ties go to the setting closest to the current weights.\n")
    ins = res["in_sample"]
    w("| | Current weights | Best found |")
    w("|---|---|---|")
    for k in TUNED_KEYS:
        w(f"| {k} | {d[k]:.2f} | {ins['config'][k]:.2f} |")
    w(f"| **NDCG@5 on all 7 listeners (optimistic)** | {ins['default']:.3f} | {ins['tuned']:.3f} |\n")
    n_ties = res["ties_at_top"]
    if n_ties == 1:
        w(f"Only 1 of {N_CONFIGS} settings reaches that best score: a narrow peak, which is a warning sign for overfitting.\n")
    else:
        w(f"{n_ties} of {N_CONFIGS} settings tie for that best score, so the data cannot single out one set of weights.\n")

    cv = res["cv"]
    lo, hi = cv["diff_ci"]
    w("### Honest check: leave one listener out\n")
    w("For each listener, tune on the other six, then score on the one left out.\n")
    w("| Held-out listener | Current weights | Tuned on the other six | Difference |")
    w("|---|---|---|---|")
    for f in cv["folds"]:
        w(f"| {L[f['held_out']]['name']} | {f['default']:.2f} | {f['tuned']:.2f} | {pct(f['tuned'] - f['default'])} |")
    w(f"| **Mean** | **{cv['default']:.3f}** | **{cv['tuned']:.3f}** | **{pct(cv['tuned'] - cv['default'])}** "
      f"(95% interval {lo:+.2f} to {hi:+.2f}) |\n")
    gap = ins["tuned"] - cv["tuned"]
    w(f"Tuning looks better on the data it was trained on ({ins['tuned']:.3f}) than on unseen listeners "
      f"({cv['tuned']:.3f}): a gap of {gap:.2f} that is overfitting, not real improvement.\n")

    w("## 2. How sensitive is the system to each weight?\n")
    w("![Sensitivity](v2_sensitivity.png)\n")
    header = "| Weight (current value) | " + " | ".join(f"{f:g}x" for f in SWEEP_FACTORS) + " | Range | Within 0.03 of current |"
    w(header)
    w("|---|" + "---|" * (len(SWEEP_FACTORS) + 2))
    for key, pts in res["sweeps"].items():
        vals = [v for _, v in pts]
        w(f"| {key} ({d[key]:.2f}) | " + " | ".join(f"{v:.2f}" for v in vals)
          + f" | {max(vals) - min(vals):.2f} | {safe_range(pts, res['baseline'])} |")
    pert = res["perturbed"]
    w(f"\nJittering all four weights at once by up to ±{int(PERTURB_SPREAD * 100)}% "
      f"({PERTURB_TRIALS} trials): mean NDCG@5 ranges from {min(pert):.2f} to {max(pert):.2f} "
      f"(median {statistics.median(pert):.2f}) against {res['baseline']:.2f} for the current weights.\n")
    return "\n".join(L_out)


def main():
    res = run()
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    plot(res, out / "v2_sensitivity.png")
    report = build_report(res)
    (out / "v2_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
