"""
Pure helpers for weight tuning (no recommender imports, so they are easy to test).

A "config" is a dict of weights. `matrix[c][l]` is the score (e.g. NDCG@5) of
config `c` for listener `l`. Tuning picks the config with the best average score
on the *training* listeners; honest evaluation then scores it on a listener that
was left out.
"""
import math
import random
from typing import Dict, List, Sequence

TUNED_KEYS = ("genre", "mood", "energy", "acoustic")


def sample_configs(default: Dict[str, float], n: int, rng: random.Random,
                   keys: Sequence[str] = TUNED_KEYS, octaves: float = 2.0) -> List[Dict[str, float]]:
    """
    Config 0 is the default. The rest scale each tuned weight by a random factor
    between 2**-octaves and 2**octaves (log-uniform), so 0.25x to 4x by default.
    """
    configs = [{k: default[k] for k in keys}]
    for _ in range(n - 1):
        configs.append({k: default[k] * 2 ** rng.uniform(-octaves, octaves) for k in keys})
    return configs


def distance_from_default(config: Dict[str, float], default: Dict[str, float]) -> float:
    """Total number of octaves (doublings/halvings) the config is away from the default."""
    return sum(abs(math.log2(config[k] / default[k])) for k in config)


def best_config_index(matrix: List[List[float]], configs: List[Dict[str, float]],
                      default: Dict[str, float], train: Sequence[int]) -> int:
    """
    Index of the config with the highest mean score over `train` listeners.
    Many configs tie (rankings change in steps), so ties go to the config
    closest to the default: change the weights only when it clearly helps.
    """
    def key(c):
        mean = sum(matrix[c][l] for l in train) / len(train)
        return (round(mean, 9), -distance_from_default(configs[c], default))

    return max(range(len(configs)), key=key)


def leave_one_listener_out(matrix: List[List[float]], configs: List[Dict[str, float]],
                           default: Dict[str, float]) -> List[Dict]:
    """
    For each listener: tune on the others, then score the chosen config on the
    held-out listener. Also records the default's score for a paired comparison.
    """
    n_listeners = len(matrix[0])
    rows = []
    for held in range(n_listeners):
        train = [l for l in range(n_listeners) if l != held]
        c = best_config_index(matrix, configs, default, train)
        rows.append({
            "held_out": held,
            "config_index": c,
            "config": configs[c],
            "tuned": matrix[c][held],
            "default": matrix[0][held],
        })
    return rows
