import random

from src.tuning import (TUNED_KEYS, best_config_index, best_index_simplest, distance_from_default,
                        leave_one_listener_out, leave_one_out_simplest, sample_configs)

DEFAULT = {"genre": 1.0, "mood": 2.0, "energy": 2.0, "acoustic": 0.5}


def test_sample_configs_starts_with_default_and_stays_in_bounds():
    configs = sample_configs(DEFAULT, 200, random.Random(0))
    assert len(configs) == 200 and configs[0] == DEFAULT
    for c in configs:
        for k in TUNED_KEYS:
            assert DEFAULT[k] / 4.0001 <= c[k] <= DEFAULT[k] * 4.0001


def test_sampling_is_reproducible_for_a_seed():
    assert sample_configs(DEFAULT, 20, random.Random(3)) == sample_configs(DEFAULT, 20, random.Random(3))


def test_distance_from_default_counts_octaves():
    assert distance_from_default(DEFAULT, DEFAULT) == 0
    doubled = {**DEFAULT, "genre": 2.0, "mood": 1.0}   # +1 octave, -1 octave
    assert abs(distance_from_default(doubled, DEFAULT) - 2.0) < 1e-12


def test_best_config_prefers_higher_score_then_closer_to_default():
    configs = [DEFAULT, {**DEFAULT, "genre": 2.0}, {**DEFAULT, "genre": 4.0}]
    #            default, tied-better, tied-better-but-farther
    matrix = [[0.2, 0.2], [0.5, 0.5], [0.5, 0.5]]
    assert best_config_index(matrix, configs, DEFAULT, [0, 1]) == 1
    flat = [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]
    assert best_config_index(flat, configs, DEFAULT, [0, 1]) == 0   # all tie -> keep default


def test_leave_one_out_never_trains_on_the_held_out_listener():
    configs = [DEFAULT, {**DEFAULT, "genre": 2.0}]
    # config 1 is great for listener 0 only. When listener 0 is held out it must NOT be chosen.
    matrix = [[0.0, 0.3, 0.3], [1.0, 0.0, 0.0]]
    rows = leave_one_listener_out(matrix, configs, DEFAULT)
    assert rows[0]["config_index"] == 0 and rows[0]["tuned"] == 0.0
    assert rows[1]["default"] == 0.3


def test_simplest_wins_ties_and_better_scores_win_otherwise():
    matrix = [[0.5, 0.5], [0.5, 0.5], [0.9, 0.9]]
    assert best_index_simplest(matrix, [0, 1, 2], [0, 1]) == 2
    tied = [[0.5, 0.5], [0.5, 0.5]]
    assert best_index_simplest(tied, [0, 1], [0, 1]) == 0


def test_leave_one_out_simplest_ignores_the_held_out_listener():
    matrix = [[0.0, 0.3, 0.3], [1.0, 0.0, 0.0]]     # config 1 only helps listener 0
    rows = leave_one_out_simplest(matrix, [0, 1])
    assert rows[0]["config_index"] == 0 and rows[0]["chosen"] == 0.0
