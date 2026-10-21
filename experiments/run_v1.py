"""
v1 experiment: does the recommender beat simple baselines, and which of its
parts actually matter?

Protocol (per hand-labelled listener, see data/eval_listeners.json):
  1. Derive the listener's taste profile from their 3 seed songs.
  2. Remove the seed songs from the catalog (the listener already knows them).
  3. Ask each method for the top k songs.
  4. Score against the listener's held-out liked songs.

Usage:  python -m experiments.run_v1
"""
import json
import random
import statistics
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from src import recommender as rec
from src.baselines import (centroid_knn, energy_only, genre_only, random_baseline,
                           rule_match_count)
from src.metrics import bootstrap_ci, hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.recommender import load_songs, recommend_songs

K = 5
RANDOM_TRIALS = 500
ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- data ------
def profile_from_seeds(seeds):
    """Taste profile a user would plausibly state: modal genre/mood, mean energy."""
    def mode(values):  # ties go to the earliest seed
        counts = Counter(values)
        best = max(counts.values())
        return next(v for v in values if counts[v] == best)

    return {
        "genre": mode([s["genre"] for s in seeds]),
        "mood": mode([s["mood"] for s in seeds]),
        "energy": round(sum(s["energy"] for s in seeds) / len(seeds), 2),
        "likes_acoustic": sum(s["acousticness"] for s in seeds) / len(seeds) > 0.5,
    }


def load_listeners(songs):
    by_id = {s["id"]: s for s in songs}
    data = json.loads((ROOT / "data/eval_listeners.json").read_text(encoding="utf-8"))
    listeners = []
    for item in data["listeners"]:
        seeds = [by_id[i] for i in item["seeds"]]
        listeners.append({
            "name": item["name"],
            "seeds": seeds,
            "profile": profile_from_seeds(seeds),
            "relevant": sorted(set(item["liked"]) - set(item["seeds"])),
            "catalog": [s for s in songs if s["id"] not in item["seeds"]],
        })
    return listeners


# ------------------------------------------------------------- ablation -----
@contextmanager
def patched(weights=None, artist_penalty=None, no_retrieval=False):
    """Temporarily change the recommender's settings (restored on exit)."""
    saved_w = dict(rec.WEIGHTS)
    saved_p = rec.ARTIST_REPEAT_PENALTY
    saved_r = rec.retrieve_candidates_with_tier
    try:
        rec.WEIGHTS.update(weights or {})
        if artist_penalty is not None:
            rec.ARTIST_REPEAT_PENALTY = artist_penalty
        if no_retrieval:
            rec.retrieve_candidates_with_tier = lambda prefs, songs, k: (list(songs), "all")
        yield
    finally:
        rec.WEIGHTS.clear()
        rec.WEIGHTS.update(saved_w)
        rec.ARTIST_REPEAT_PENALTY = saved_p
        rec.retrieve_candidates_with_tier = saved_r


ABLATIONS = {
    "Full system": {},
    "Without genre": {"weights": {"genre": 0.0}},
    "Without mood": {"weights": {"mood": 0.0}},
    "Without energy": {"weights": {"energy": 0.0}},
    "Without acoustic preference": {"weights": {"acoustic": 0.0, "acoustic_target": 0.0}},
    "Without diversity penalty": {"artist_penalty": 0.0},
    "Without retrieval filter": {"no_retrieval": True},
}


def system_ids(listener, **patch):
    with patched(**patch):
        return [s["id"] for s, _, _ in recommend_songs(listener["profile"], listener["catalog"], k=K)]


# ------------------------------------------------------------ evaluation ----
def score(ids, relevant):
    return {
        "precision": precision_at_k(ids, relevant, K),
        "recall": recall_at_k(ids, relevant, K),
        "hit": hit_rate_at_k(ids, relevant, K),
        "ndcg": ndcg_at_k(ids, relevant, K),
    }


def evaluate(method, listeners):
    """method(listener) -> ids, or (for random) a function returning averaged scores."""
    return [score(method(l), l["relevant"]) for l in listeners]


def random_scores(listeners):
    out = []
    for l in listeners:
        rng = random.Random(l["name"])
        runs = [score(random_baseline(l["catalog"], K, rng), l["relevant"]) for _ in range(RANDOM_TRIALS)]
        out.append({m: statistics.mean(r[m] for r in runs) for m in runs[0]})
    return out


def summarize(per_listener):
    summary = {m: statistics.mean(r[m] for r in per_listener) for m in per_listener[0]}
    lo, hi = bootstrap_ci([r["ndcg"] for r in per_listener])
    summary["ndcg_ci"] = (lo, hi)
    return summary


def run():
    songs = load_songs(str(ROOT / "data/songs.csv"))
    listeners = load_listeners(songs)

    methods = {
        "Random (avg of 500 draws)": None,
        "Energy only": lambda l: energy_only(l["profile"], l["catalog"], K),
        "Genre only": lambda l: genre_only(l["profile"], l["catalog"], K),
        "Genre + mood rules, no scoring": lambda l: rule_match_count(l["profile"], l["catalog"], K),
        "Audio-feature nearest neighbours": lambda l: centroid_knn(l["seeds"], l["catalog"], K),
        "This system (full)": lambda l: system_ids(l),
    }
    results = {}
    for name, fn in methods.items():
        results[name] = random_scores(listeners) if fn is None else evaluate(fn, listeners)

    ablation = {name: evaluate(lambda l, p=patch: system_ids(l, **p), listeners)
                for name, patch in ABLATIONS.items()}
    return listeners, results, ablation


# ---------------------------------------------------------------- report ----
def fmt_ci(ci):
    return f"{ci[0]:.2f} to {ci[1]:.2f}"


def build_report(listeners, results, ablation):
    L = []
    L.append(f"# v1 results: baselines and ablation (k={K}, {len(listeners)} listeners)\n")
    L.append("Generated by `python -m experiments.run_v1`. Relevance = songs a hand-labelled listener "
             "would enjoy that were *not* used to build their profile. With only "
             f"{len(listeners)} listeners these numbers are indicative, not statistically conclusive; "
             "the 95% interval is a bootstrap over listeners.\n")

    L.append("## Listeners\n")
    L.append("| Listener | Derived profile | Held-out relevant songs |")
    L.append("|---|---|---|")
    for l in listeners:
        p = l["profile"]
        L.append(f"| {l['name']} | {p['genre']} / {p['mood']} / energy {p['energy']} / "
                 f"{'likes' if p['likes_acoustic'] else 'avoids'} acoustic | {len(l['relevant'])} |")

    L.append("\n## Baseline comparison\n")
    L.append("| Method | Precision@5 | Recall@5 | Hit rate@5 | NDCG@5 | NDCG@5 95% interval |")
    L.append("|---|---|---|---|---|---|")
    for name, per in results.items():
        s = summarize(per)
        L.append(f"| {name} | {s['precision']:.2f} | {s['recall']:.2f} | {s['hit']:.2f} | "
                 f"{s['ndcg']:.2f} | {fmt_ci(s['ndcg_ci'])} |")

    L.append("\n### Per-listener NDCG@5\n")
    names = list(results)
    L.append("| Listener | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    for i, l in enumerate(listeners):
        L.append(f"| {l['name']} | " + " | ".join(f"{results[n][i]['ndcg']:.2f}" for n in names) + " |")

    L.append("\n## Ablation: what happens when one part of the system is removed?\n")
    full = summarize(ablation["Full system"])
    L.append("| Variant | Precision@5 | NDCG@5 | Change in NDCG vs full |")
    L.append("|---|---|---|---|")
    for name, per in ablation.items():
        s = summarize(per)
        delta = s["ndcg"] - full["ndcg"]
        L.append(f"| {name} | {s['precision']:.2f} | {s['ndcg']:.2f} | "
                 f"{'' if name == 'Full system' else f'{delta:+.2f}'} |")
    return "\n".join(L) + "\n"


def main():
    listeners, results, ablation = run()
    report = build_report(listeners, results, ablation)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/v1_results.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
