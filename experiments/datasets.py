"""Datasets for the experiments: the small hand-built catalog and the large derived one."""
import json
import random
from collections import Counter
from pathlib import Path

from src.recommender import load_songs

ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "small": {"csv": "data/songs.csv", "suffix": "", "protocol": "hand"},
    # Same catalog, two opposite ways of defining what a listener likes (see docs in README):
    "large": {"csv": "data/catalog_large.csv", "suffix": "_large", "protocol": "genre"},
    "large_audio": {"csv": "data/catalog_large.csv", "suffix": "_large_audio", "protocol": "audio"},
}
AUDIO_NEIGHBOURS = 100   # size of a listener's "true taste" neighbourhood (5% of the catalog)
VISIBLE = ("energy", "valence", "danceability", "acousticness")
HIDDEN = ("gt_speechiness", "gt_instrumentalness", "gt_liveness", "gt_loudness")


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


def _listener(name, group, seeds, relevant_ids, songs):
    seed_ids = {s["id"] for s in seeds}
    return {
        "songs": songs,
        "name": name,
        "group": group,
        "seeds": seeds,
        "profile": profile_from_seeds(seeds),
        "relevant": sorted(set(relevant_ids) - seed_ids),
        "catalog": [s for s in songs if s["id"] not in seed_ids],
    }


def load_listeners(songs):
    """The 7 hand-labelled listeners for the small catalog."""
    by_id = {s["id"]: s for s in songs}
    data = json.loads((ROOT / "data/eval_listeners.json").read_text(encoding="utf-8"))
    return [
        _listener(item["name"], item["name"], [by_id[i] for i in item["seeds"]], item["liked"], songs)
        for item in data["listeners"]
    ]


def archetype_listeners(songs):
    """Synthetic listeners for the large catalog: several random seed draws per archetype."""
    cfg = json.loads((ROOT / "data/eval_archetypes.json").read_text(encoding="utf-8"))
    rng = random.Random(cfg["random_seed"])
    listeners = []
    for arch in cfg["archetypes"]:
        liked = [s for s in songs if s["genre"] in arch["genres"]]
        for draw in range(cfg["draws_per_archetype"]):
            seeds = rng.sample(liked, cfg["seeds_per_listener"])
            listeners.append(_listener(f"{arch['name']} #{draw + 1}", arch["name"], seeds,
                                       [s["id"] for s in liked], songs))
    return listeners


def _rich_vectors(csv_path, songs):
    """
    Every song as a point in a RICHER audio space than the recommender sees:
    its visible features plus speechiness, instrumentalness, liveness and loudness
    (read from the hidden gt_ columns). Each dimension is scaled to 0..1.
    """
    import csv
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        hidden = {int(r["id"]): r for r in csv.DictReader(f)}
    raw = {}
    for s in songs:
        h = hidden[s["id"]]
        loud = min(0.0, max(-30.0, float(h["gt_loudness"])))
        raw[s["id"]] = ([s[f] for f in VISIBLE] + [s["tempo_bpm"] / 200.0, float(h["gt_speechiness"]),
                         float(h["gt_instrumentalness"]), float(h["gt_liveness"]), (loud + 30.0) / 30.0])
    dims = list(zip(*raw.values()))
    lo, hi = [min(d) for d in dims], [max(d) for d in dims]
    return {i: [(v - a) / (b - a) if b > a else 0.0 for v, a, b in zip(vec, lo, hi)] for i, vec in raw.items()}


def audio_taste_listeners(songs, csv_path):
    """
    Same seeds as the genre protocol, but a listener's taste is the AUDIO_NEIGHBOURS songs
    closest to their seeds in the richer audio space (so taste is not defined by genre).
    """
    import math
    rich = _rich_vectors(csv_path, songs)
    out = []
    for l in archetype_listeners(songs):
        centroid = [sum(c) / len(c) for c in zip(*[rich[s["id"]] for s in l["seeds"]])]
        ranked = sorted((s for s in l["catalog"]), key=lambda s: math.dist(centroid, rich[s["id"]]))
        out.append({**l, "relevant": sorted(s["id"] for s in ranked[:AUDIO_NEIGHBOURS])})
    return out


def load_dataset(name):
    """Return (songs, listeners) for 'small', 'large' (genre taste) or 'large_audio' (audio taste)."""
    cfg = DATASETS[name]
    path = str(ROOT / cfg["csv"])
    songs = load_songs(path)
    if cfg["protocol"] == "hand":
        return songs, load_listeners(songs)
    if cfg["protocol"] == "genre":
        return songs, archetype_listeners(songs)
    return songs, audio_taste_listeners(songs, path)


def real_listener_draws(songs, taste_files, draws=20, seeds_per_listener=3, seed=5):
    """
    Turn real taste files into evaluation listeners: several random draws of 3 seed songs
    per person; the person's other liked songs are the held-out relevant songs.
    """
    rng = random.Random(seed)
    by_id = {s["id"]: s for s in songs}
    listeners = []
    for data in taste_files:
        liked = [by_id[i] for i in data["liked_ids"]]
        for draw in range(draws):
            seeds = rng.sample(liked, seeds_per_listener)
            listeners.append(_listener(f"{data['nickname']} #{draw + 1}", data["nickname"], seeds,
                                       data["liked_ids"], songs))
    return listeners
