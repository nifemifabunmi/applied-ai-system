"""
Pure helpers for weight tuning (no recommender imports, so they are easy to test).

A "config" is a dict of weights. `matrix[c][l]` is the score (e.g. NDCG@5) of
config `c` for listener `l`. Tuning picks the config with the best average score
on the *training* listeners; honest evaluation then scores it on a listener that
was left out.
"""
import math
import random
from typing import Dict, List, Optional, Sequence

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


def _folds(n_listeners: int, groups: Optional[Sequence[str]]):
    """Yield (held_out_indices, train_indices). A group is held out as a whole."""
    labels = list(groups) if groups is not None else list(range(n_listeners))
    for label in dict.fromkeys(labels):
        held = [i for i, g in enumerate(labels) if g == label]
        yield held, [i for i in range(n_listeners) if labels[i] != label]


def leave_one_listener_out(matrix: List[List[float]], configs: List[Dict[str, float]],
                           default: Dict[str, float], groups: Optional[Sequence[str]] = None) -> List[Dict]:
    """
    For each listener (or each whole group, when `groups` is given): tune on the
    rest, then score the chosen config on the held-out ones (their mean). Also
    records the default's score for a paired comparison.
    """
    rows = []
    for held, train in _folds(len(matrix[0]), groups):
        c = best_config_index(matrix, configs, default, train)
        rows.append({
            "held_out": held[0] if len(held) == 1 else held,
            "config_index": c,
            "config": configs[c],
            "tuned": sum(matrix[c][l] for l in held) / len(held),
            "default": sum(matrix[0][l] for l in held) / len(held),
        })
    return rows


def best_index_simplest(matrix: List[List[float]], complexity: Sequence[float],
                        train: Sequence[int]) -> int:
    """
    Index of the config with the best mean score over `train` listeners; ties go
    to the least complex config, so a new component is only adopted when it clearly helps.
    """
    def key(c):
        return (round(sum(matrix[c][l] for l in train) / len(train), 9), -complexity[c])

    return max(range(len(matrix)), key=key)


def leave_one_out_simplest(matrix: List[List[float]], complexity: Sequence[float],
                           baseline_index: int = 0, groups: Optional[Sequence[str]] = None) -> List[Dict]:
    """Leave-one-listener (or whole-group) out selection among configs; compares with `baseline_index`."""
    rows = []
    for held, train in _folds(len(matrix[0]), groups):
        c = best_index_simplest(matrix, complexity, train)
        rows.append({"held_out": held[0] if len(held) == 1 else held, "config_index": c,
                     "chosen": sum(matrix[c][l] for l in held) / len(held),
                     "baseline": sum(matrix[baseline_index][l] for l in held) / len(held)})
    return rows
