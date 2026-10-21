"""Ranking-quality metrics with binary relevance, plus a bootstrap CI helper."""
import math
import random
from typing import Iterable, List, Sequence, Tuple


def precision_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float:
    rel = set(relevant)
    top = list(recs)[:k]
    return sum(r in rel for r in top) / k if k else 0.0


def recall_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return sum(r in rel for r in list(recs)[:k]) / len(rel)


def hit_rate_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float:
    rel = set(relevant)
    return 1.0 if any(r in rel for r in list(recs)[:k]) else 0.0


def ndcg_at_k(recs: Sequence[int], relevant: Iterable[int], k: int) -> float:
    """Normalised discounted cumulative gain: rewards relevant songs placed higher."""
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, r in enumerate(list(recs)[:k]) if r in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / ideal if ideal else 0.0


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap CI of the mean (resampling listeners)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values) for _ in range(n_boot)
    )
    lo = means[int(n_boot * alpha / 2)]
    hi = means[int(n_boot * (1 - alpha / 2)) - 1]
    return lo, hi


def cluster_bootstrap_ci(values: Sequence[float], groups: Sequence[str], n_boot: int = 2000,
                         seed: int = 0, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Bootstrap CI of the mean where whole groups (e.g. listeners drawn from the same
    archetype) are resampled together. Each group counts once, so repeated random
    draws from one archetype cannot make the interval look tighter than it is.
    """
    by_group = {}
    for v, g in zip(values, groups):
        by_group.setdefault(g, []).append(v)
    means = [sum(vs) / len(vs) for vs in by_group.values()]
    return bootstrap_ci(means, n_boot=n_boot, seed=seed, alpha=alpha)
