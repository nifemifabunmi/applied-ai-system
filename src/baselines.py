"""
Simple reference recommenders to compare the real system against.
Every function returns a list of up to k song ids, best first.
"""
import math
import random
from typing import Dict, List

AUDIO_FEATURES = ("energy", "valence", "danceability", "acousticness")


def random_baseline(catalog: List[Dict], k: int, rng: random.Random) -> List[int]:
    return [s["id"] for s in rng.sample(catalog, min(k, len(catalog)))]


def energy_only(profile: Dict, catalog: List[Dict], k: int) -> List[int]:
    target = profile.get("energy", 0.5)
    ranked = sorted(catalog, key=lambda s: abs(s["energy"] - target))
    return [s["id"] for s in ranked[:k]]


def genre_only(profile: Dict, catalog: List[Dict], k: int) -> List[int]:
    """Same-genre songs first; any shortfall is filled by energy closeness."""
    target = profile.get("energy", 0.5)
    ranked = sorted(
        catalog,
        key=lambda s: (s["genre"] != profile.get("genre"), abs(s["energy"] - target)),
    )
    return [s["id"] for s in ranked[:k]]


def rule_match_count(profile: Dict, catalog: List[Dict], k: int) -> List[int]:
    """Count exact genre + mood matches; ties keep catalog order (no energy, no features)."""
    def matches(s):
        return (s["genre"] == profile.get("genre")) + (s["mood"] == profile.get("mood"))
    ranked = sorted(catalog, key=lambda s: -matches(s))  # sorted() is stable
    return [s["id"] for s in ranked[:k]]


def _vector(song: Dict) -> List[float]:
    return [song[f] for f in AUDIO_FEATURES] + [song["tempo_bpm"] / 200.0]


def centroid_knn(seeds: List[Dict], catalog: List[Dict], k: int) -> List[int]:
    """
    Nearest songs (Euclidean) to the average of the listener's seed songs in
    audio-feature space (energy, valence, danceability, acousticness, tempo).
    Ignores genre and mood labels entirely.
    """
    vectors = [_vector(s) for s in seeds]
    centroid = [sum(col) / len(col) for col in zip(*vectors)]

    def dist(s):
        return math.dist(centroid, _vector(s))

    return [s["id"] for s in sorted(catalog, key=dist)[:k]]
