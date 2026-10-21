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
import argparse
import random
import statistics
from contextlib import contextmanager
from pathlib import Path

from src import recommender as rec
from src.baselines import (centroid_knn, energy_only, genre_only, random_baseline,
                           rule_match_count)
from experiments.datasets import DATASETS, load_dataset, load_listeners, profile_from_seeds  # noqa: F401 (re-exported)
from src.metrics import (cluster_bootstrap_ci, hit_rate_at_k, ndcg_at_k, precision_at_k,
                         recall_at_k)
from src.recommender import recommend_songs

K = 5
RANDOM_TRIALS = 500
ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------- ablation -----
@contextmanager
def patched(weights=None, artist_penalty=None, no_retrieval=False, filter_when_liked=None):
    """Temporarily change the recommender's settings (restored on exit)."""
    saved_w = dict(rec.WEIGHTS)
    saved_p = rec.ARTIST_REPEAT_PENALTY
    saved_r = rec.retrieve_candidates_with_tier
    saved_f = rec.FILTER_WHEN_LIKED
    try:
        if filter_when_liked is not None:
            rec.FILTER_WHEN_LIKED = filter_when_liked
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
        rec.FILTER_WHEN_LIKED = saved_f


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


def summarize(per_listener, groups):
    summary = {m: statistics.mean(r[m] for r in per_listener) for m in per_listener[0]}
    summary["ndcg_ci"] = cluster_bootstrap_ci([r["ndcg"] for r in per_listener], groups)
    return summary


def system_hybrid_ids(listener):
    """The shipped default when the user also lists liked songs (the listener's seeds)."""
    prefs = dict(listener["profile"])
    prefs["liked_ids"] = [s["id"] for s in listener["seeds"]]
    return [s["id"] for s, _, _ in recommend_songs(prefs, listener["songs"], k=K)]


def run(dataset="small"):
    songs, listeners = load_dataset(dataset)
    return run_listeners(listeners)


def run_listeners(listeners, with_ablation=True):

    methods = {
        "Random (avg of 500 draws)": None,
        "Energy only": lambda l: energy_only(l["profile"], l["catalog"], K),
        "Genre only": lambda l: genre_only(l["profile"], l["catalog"], K),
        "Genre + mood rules, no scoring": lambda l: rule_match_count(l["profile"], l["catalog"], K),
        "Audio-feature nearest neighbours": lambda l: centroid_knn(l["seeds"], l["catalog"], K),
        "This system (full)": lambda l: system_ids(l),
        "This system + liked songs (default blend)": lambda l: system_hybrid_ids(l),
    }
    results = {}
    for name, fn in methods.items():
        results[name] = random_scores(listeners) if fn is None else evaluate(fn, listeners)

    ablation = None
    if with_ablation:
        ablation = {name: evaluate(lambda l, p=patch: system_ids(l, **p), listeners)
                    for name, patch in ABLATIONS.items()}
    return listeners, results, ablation


# ---------------------------------------------------------------- report ----
def fmt_ci(ci, n_groups=10):
    if n_groups < 5:
        return "n/a (fewer than 5 listeners)"
    return f"{ci[0]:.2f} to {ci[1]:.2f}"


def group_means(per_listener, groups, metric="ndcg"):
    """Mean of `metric` per group, in order of first appearance."""
    by = {}
    for r, g in zip(per_listener, groups):
        by.setdefault(g, []).append(r[metric])
    return {g: statistics.mean(v) for g, v in by.items()}


def build_report(listeners, results, ablation, dataset="small"):
    groups = [l["group"] for l in listeners]
    n_groups = len(set(groups))
    n_songs = len(listeners[0]["catalog"]) + len(listeners[0]["seeds"])
    L = []
    L.append(f"# v1 results ({dataset} catalog): baselines and ablation (k={K}, {n_songs} songs, "
             f"{len(listeners)} listeners in {n_groups} groups)\n")
    if dataset == "small":
        L.append("Generated by `python -m experiments.run_v1`. Relevance = songs a hand-labelled listener "
                 "would enjoy that were *not* used to build their profile. With only "
                 f"{len(listeners)} listeners these numbers are indicative, not statistically conclusive; "
                 "the 95% interval is a bootstrap over listeners.\n")
    elif dataset == "real":
        L.append("Generated by `python -m experiments.run_v5_real`. **Real listeners**: each person contributed songs "
                 "they like; a listener here is 3 random seed songs from their list, and the rest of their list is "
                 "the held-out answer. Lists are short and people can only pick songs they recognise, so NDCG "
                 "values are small and the sample is tiny. Intervals appear only with 5 or more people.\n")
    elif dataset == "large":
        L.append("Generated by `python -m experiments.run_v1 --dataset large`. **Genre-taste protocol** "
                 "(synthetic): each archetype likes every song in its genres; a listener is 3 random seed songs "
                 "from them; relevant = the archetype's other songs. Because taste is *defined by genre*, any "
                 "method that returns the seeds' genre is favoured, so read these numbers as a sanity check, "
                 "not as a ranking of methods. Mood here is derived from energy and valence. The 95% interval "
                 "resamples whole archetypes.\n")
    else:
        L.append("Generated by `python -m experiments.run_v1 --dataset large_audio`. **Audio-taste protocol** "
                 "(synthetic): same seeds, but a listener's taste is the 100 songs closest to the seeds in a "
                 "richer audio space (speechiness, instrumentalness, liveness and loudness, which the "
                 "recommender never sees, plus its own features). Taste is not defined by genre, so "
                 "similarity-based methods are favoured instead. Mood is derived from energy and valence. "
                 "The 95% interval resamples whole archetypes.\n")

    if len(listeners) <= 12:
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
        s = summarize(per, groups)
        L.append(f"| {name} | {s['precision']:.2f} | {s['recall']:.2f} | {s['hit']:.2f} | "
                 f"{s['ndcg']:.2f} | {fmt_ci(s['ndcg_ci'], n_groups)} |")

    names = list(results)
    L.append("\n### NDCG@5 by " + ("listener" if len(listeners) <= 12 else "archetype (mean of its draws)") + "\n")
    L.append("| " + ("Listener" if len(listeners) <= 12 else "Archetype") + " | " + " | ".join(names) + " |")
    L.append("|---|" + "---|" * len(names))
    per_group = {n: group_means(results[n], groups) for n in names}
    for g in dict.fromkeys(groups):
        L.append(f"| {g} | " + " | ".join(f"{per_group[n][g]:.2f}" for n in names) + " |")

    if ablation is None:
        return "\n".join(L) + "\n"
    L.append("\n## Ablation: what happens when one part of the system is removed?\n")
    full = summarize(ablation["Full system"], groups)
    L.append("| Variant | Precision@5 | NDCG@5 | Change in NDCG vs full |")
    L.append("|---|---|---|---|")
    for name, per in ablation.items():
        s = summarize(per, groups)
        delta = s["ndcg"] - full["ndcg"]
        L.append(f"| {name} | {s['precision']:.2f} | {s['ndcg']:.2f} | "
                 f"{'' if name == 'Full system' else f'{delta:+.2f}'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="small")
    args = ap.parse_args()
    listeners, results, ablation = run(args.dataset)
    report = build_report(listeners, results, ablation, args.dataset)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / f"reports/v1_results{DATASETS[args.dataset]['suffix']}.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
