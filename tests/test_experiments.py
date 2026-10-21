import random

from experiments.run_v1 import K, load_listeners, patched, profile_from_seeds, system_ids
from src import recommender as rec
from src.baselines import centroid_knn, energy_only, genre_only, random_baseline, rule_match_count
from src.metrics import bootstrap_ci, hit_rate_at_k, ndcg_at_k, precision_at_k, recall_at_k
from src.recommender import load_songs

SONGS = load_songs("data/songs.csv")


# ---- metrics: hand-computed values -----------------------------------------
def test_precision_recall_hit_known_values():
    recs, relevant = [1, 2, 3, 4, 5], [2, 5, 9, 10]
    assert precision_at_k(recs, relevant, 5) == 0.4
    assert recall_at_k(recs, relevant, 5) == 0.5
    assert hit_rate_at_k(recs, relevant, 5) == 1.0
    assert hit_rate_at_k([1, 3], relevant, 2) == 0.0


def test_ndcg_rewards_higher_placement_and_is_bounded():
    relevant = [7]
    top = ndcg_at_k([7, 1, 2], relevant, 3)
    low = ndcg_at_k([1, 2, 7], relevant, 3)
    assert top == 1.0 and 0.0 < low < top
    assert ndcg_at_k([1, 2, 3], relevant, 3) == 0.0
    assert ndcg_at_k([1], [], 3) == 0.0


def test_bootstrap_ci_contains_the_mean_and_is_ordered():
    values = [0.2, 0.4, 0.6, 0.8]
    lo, hi = bootstrap_ci(values)
    assert lo <= sum(values) / len(values) <= hi


# ---- baselines ---------------------------------------------------------------
def test_every_baseline_returns_k_unique_known_ids():
    listener = load_listeners(SONGS)[0]
    outputs = [
        random_baseline(listener["catalog"], K, random.Random(0)),
        energy_only(listener["profile"], listener["catalog"], K),
        genre_only(listener["profile"], listener["catalog"], K),
        rule_match_count(listener["profile"], listener["catalog"], K),
        centroid_knn(listener["seeds"], listener["catalog"], K),
    ]
    valid = {s["id"] for s in listener["catalog"]}
    for ids in outputs:
        assert len(ids) == K and len(set(ids)) == K and set(ids) <= valid


def test_genre_only_puts_same_genre_first():
    listener = load_listeners(SONGS)[0]  # study session, lofi
    ids = genre_only(listener["profile"], listener["catalog"], 2)  # only 2 lofi songs remain once seeds are removed
    genres = {s["id"]: s["genre"] for s in SONGS}
    assert all(genres[i] == "lofi" for i in ids)


# ---- evaluation set integrity --------------------------------------------------
def test_listeners_are_well_formed_and_seeds_never_leak():
    for l in load_listeners(SONGS):
        seed_ids = {s["id"] for s in l["seeds"]}
        assert len(seed_ids) == 3
        assert seed_ids.isdisjoint(set(l["relevant"]))          # held-out only
        assert seed_ids.isdisjoint({s["id"] for s in l["catalog"]})
        assert len(l["relevant"]) >= 3
        assert set(l["relevant"]) <= {s["id"] for s in l["catalog"]}


def test_system_never_recommends_a_seed_song():
    for l in load_listeners(SONGS):
        ids = system_ids(l)
        assert len(ids) == K
        assert not ({s["id"] for s in l["seeds"]} & set(ids))


def test_profile_from_seeds_uses_modal_genre_and_mean_energy():
    seeds = [s for s in SONGS if s["id"] in (2, 4, 6)]  # lofi, lofi, ambient
    p = profile_from_seeds(seeds)
    assert p["genre"] == "lofi" and p["mood"] == "chill" and p["likes_acoustic"] is True
    assert abs(p["energy"] - round((0.42 + 0.35 + 0.28) / 3, 2)) < 1e-9


# ---- ablation helper restores global state ---------------------------------------
def test_patched_restores_weights_penalty_and_retrieval():
    before = (dict(rec.WEIGHTS), rec.ARTIST_REPEAT_PENALTY, rec.retrieve_candidates_with_tier)
    with patched(weights={"genre": 0.0}, artist_penalty=0.0, no_retrieval=True):
        assert rec.WEIGHTS["genre"] == 0.0 and rec.ARTIST_REPEAT_PENALTY == 0.0
    assert (dict(rec.WEIGHTS), rec.ARTIST_REPEAT_PENALTY, rec.retrieve_candidates_with_tier) == before
