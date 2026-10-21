from experiments.datasets import (AUDIO_NEIGHBOURS, archetype_listeners, audio_taste_listeners,
                                  load_dataset)
from src import recommender as rec
from src.metrics import bootstrap_ci, cluster_bootstrap_ci
from src.recommender import load_songs, recommend_detailed
from src.tuning import leave_one_out_simplest
from scripts.build_catalog import derive_mood

LARGE_CSV = "data/catalog_large.csv"
LARGE = load_songs(LARGE_CSV)
SMALL = load_songs("data/songs.csv")
MOODS = {"happy", "energetic", "intense", "upbeat", "relaxed", "moody", "peaceful", "chill", "sad"}


# ---- catalog -------------------------------------------------------------------
def test_derive_mood_covers_the_energy_valence_grid():
    assert derive_mood(0.9, 0.9) == "happy"
    assert derive_mood(0.9, 0.5) == "energetic"
    assert derive_mood(0.9, 0.1) == "intense"
    assert derive_mood(0.5, 0.9) == "upbeat"
    assert derive_mood(0.5, 0.5) == "relaxed"
    assert derive_mood(0.5, 0.1) == "moody"
    assert derive_mood(0.2, 0.9) == "peaceful"
    assert derive_mood(0.2, 0.5) == "chill"
    assert derive_mood(0.2, 0.1) == "sad"
    assert derive_mood(0.4, 0.33) == "relaxed"      # boundaries belong to the higher bin


def test_large_catalog_is_clean_and_balanced():
    assert len(LARGE) >= 1500
    assert len({s["id"] for s in LARGE}) == len(LARGE)
    assert len({(s["title"], s["artist"]) for s in LARGE}) == len(LARGE)
    assert {s["mood"] for s in LARGE} <= MOODS
    genres = {}
    for s in LARGE:
        genres[s["genre"]] = genres.get(s["genre"], 0) + 1
        for key in ("energy", "valence", "danceability", "acousticness"):
            assert 0.0 <= s[key] <= 1.0
        assert s["tempo_bpm"] > 0
    assert len(genres) >= 30 and min(genres.values()) >= 30


def test_hidden_ground_truth_columns_never_reach_the_recommender():
    assert not any(k.startswith("gt_") for k in LARGE[0])
    assert "popularity" not in LARGE[0]          # would change scoring if present


# ---- evaluation protocols -----------------------------------------------------------
def test_archetype_listeners_are_well_formed_and_reproducible():
    import json
    cfg = json.load(open("data/eval_archetypes.json", encoding="utf-8"))
    a, b = archetype_listeners(LARGE), archetype_listeners(LARGE)
    assert len(a) == len(cfg["archetypes"]) * cfg["draws_per_archetype"]
    assert len({l["group"] for l in a}) == len(cfg["archetypes"])
    assert [[s["id"] for s in l["seeds"]] for l in a] == [[s["id"] for s in l["seeds"]] for l in b]
    for l in a:
        seed_ids = {s["id"] for s in l["seeds"]}
        assert len(seed_ids) == 3
        assert seed_ids.isdisjoint(l["relevant"])
        assert seed_ids.isdisjoint({s["id"] for s in l["catalog"]})
        assert set(l["relevant"]) <= {s["id"] for s in l["catalog"]}


def test_genre_protocol_relevance_is_exactly_the_archetype_genres_minus_seeds():
    import json
    cfg = json.load(open("data/eval_archetypes.json", encoding="utf-8"))
    genres_of = {a["name"]: set(a["genres"]) for a in cfg["archetypes"]}
    for l in archetype_listeners(LARGE):
        expected = {s["id"] for s in LARGE if s["genre"] in genres_of[l["group"]]} - {s["id"] for s in l["seeds"]}
        assert set(l["relevant"]) == expected
        assert all(s["genre"] in genres_of[l["group"]] for s in l["seeds"])


def test_audio_protocol_uses_a_fixed_size_neighbourhood_that_differs_from_genre():
    audio = audio_taste_listeners(LARGE, LARGE_CSV)
    genre = archetype_listeners(LARGE)
    assert len(audio) == len(genre) and len(audio) >= 50
    for la, lg in zip(audio, genre):
        assert len(la["relevant"]) == AUDIO_NEIGHBOURS
        assert {s["id"] for s in la["seeds"]}.isdisjoint(la["relevant"])
        assert la["relevant"] != lg["relevant"]
        assert [s["id"] for s in la["seeds"]] == [s["id"] for s in lg["seeds"]]   # same seeds


def test_load_dataset_names():
    for name in ("small", "large", "large_audio"):
        songs, listeners = load_dataset(name)
        assert songs and listeners and all("group" in l for l in listeners)


# ---- statistics ---------------------------------------------------------------------
def test_cluster_bootstrap_is_not_tighter_than_treating_draws_as_independent():
    values = [0.0] * 8 + [1.0] * 8
    groups = ["A"] * 8 + ["B"] * 8                  # 16 draws but only 2 independent groups
    lo_c, hi_c = cluster_bootstrap_ci(values, groups)
    lo_p, hi_p = bootstrap_ci(values)
    assert (hi_c - lo_c) > (hi_p - lo_p)


def test_group_holdout_never_trains_on_any_member_of_the_group():
    # listeners 0 and 1 share a group; config 1 only helps them.
    matrix = [[0.0, 0.0, 0.3, 0.3], [1.0, 1.0, 0.0, 0.0]]
    rows = leave_one_out_simplest(matrix, [0, 1], groups=["G", "G", "H", "H"])
    assert rows[0]["config_index"] == 0 and rows[0]["chosen"] == 0.0 and len(rows) == 2


# ---- liked songs bypass the genre/mood filter -------------------------------------------
def test_liked_songs_search_the_whole_catalog_by_default():
    detail = recommend_detailed({"liked_ids": [2, 4], "genre": "lofi", "mood": "chill", "energy": 0.4}, SMALL, k=5)
    assert detail["tier"] == "liked" and detail["pool_size"] == len(SMALL) - 2


def test_filter_when_liked_switch_restores_the_old_behaviour():
    saved = rec.FILTER_WHEN_LIKED
    rec.FILTER_WHEN_LIKED = True
    try:
        detail = recommend_detailed({"liked_ids": [2, 4], "genre": "lofi", "mood": "chill", "energy": 0.4}, SMALL, k=5)
    finally:
        rec.FILTER_WHEN_LIKED = saved
    assert detail["tier"] in {"both", "either", "fallback"}


def test_every_catalog_genre_belongs_to_exactly_one_researched_factor():
    import json
    cfg = json.load(open("data/eval_archetypes.json", encoding="utf-8"))
    mapped = [m["genre"] for m in cfg["mapping"]]
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == {s["genre"] for s in LARGE}
    assert {m["factor"] for m in cfg["mapping"]} == {a["name"] for a in cfg["archetypes"]}
    assert all(m["basis"] for m in cfg["mapping"]) and len(cfg["sources"]) >= 2
    # archetype genre lists are generated from the mapping, so they must agree
    for a in cfg["archetypes"]:
        assert set(a["genres"]) == {m["genre"] for m in cfg["mapping"] if m["factor"] == a["name"]}
