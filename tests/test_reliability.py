import pytest
from src import recommender
from src.recommender import load_songs, recommend_songs, score_song

SONGS = load_songs("data/songs.csv")

PROFILES = [
    {"genre": "pop", "mood": "happy", "energy": 0.8},
    {"genre": "rock", "mood": "intense", "energy": 0.95},   # rare combo: used to return only 2
    {"genre": "lofi", "mood": "chill", "energy": 0.4},
    {"genre": "jazz", "mood": "relaxed", "energy": 0.5},
    {"genre": "nonexistent", "mood": "none", "energy": 0.1},  # forces fallback
    {},                                                      # no preferences at all
]


def song(id, artist, genre="pop", mood="happy", energy=0.8):
    return {"id": id, "title": f"S{id}", "artist": artist, "genre": genre, "mood": mood,
            "energy": energy, "tempo_bpm": 120, "valence": 0.5, "danceability": 0.5,
            "acousticness": 0.2}


# ---- reliability: always k results, no duplicates, sorted, explained -------
@pytest.mark.parametrize("prefs", PROFILES)
@pytest.mark.parametrize("k", [1, 3, 5, 10])
def test_always_returns_exactly_k_unique_sorted_results(prefs, k):
    recs = recommend_songs(prefs, SONGS, k=k)
    assert len(recs) == k
    ids = [s["id"] for s, _, _ in recs]
    assert len(set(ids)) == k
    scores = [score for _, score, _ in recs]
    assert scores == sorted(scores, reverse=True)
    assert all(expl.strip() for _, _, expl in recs)


def test_edge_cases_for_k_and_catalog_size():
    assert recommend_songs({"genre": "pop"}, SONGS, k=0) == []
    assert recommend_songs({"genre": "pop"}, [], k=5) == []
    assert len(recommend_songs({"genre": "pop"}, SONGS[:3], k=10)) == 3


def test_out_of_range_energy_is_clamped_not_crashing():
    recs = recommend_songs({"genre": "pop", "energy": 7.0}, SONGS, k=3)
    assert len(recs) == 3


# ---- quality: the right things rank first ----------------------------------
def test_exact_match_ranks_first():
    top, _, explanation = recommend_songs({"genre": "pop", "mood": "happy", "energy": 0.8}, SONGS, k=1)[0]
    assert top["genre"] == "pop" and top["mood"] == "happy"
    assert "This song" in explanation and "pop" in explanation


def test_genre_matches_dominate_top_results_when_available():
    recs = recommend_songs({"genre": "lofi", "mood": "chill", "energy": 0.4}, SONGS, k=3)
    assert all(s["genre"] == "lofi" for s, _, _ in recs)


def test_retrieval_relaxes_until_k_candidates():
    pool = recommender.retrieve_candidates({"genre": "rock", "mood": "intense"}, SONGS, k=5)
    assert len(pool) >= 5


# ---- scoring behaviour ------------------------------------------------------
def test_energy_score_decreases_smoothly_with_distance():
    prefs = {"energy": 0.5}
    scores = [score_song(song(1, "A", energy=0.5 + d / 100), prefs) for d in range(0, 60, 1)]
    assert scores == sorted(scores, reverse=True)
    # no cliff: neighbouring energy values never differ by more than a small step
    assert max(a - b for a, b in zip(scores, scores[1:])) < 0.1


def test_optional_numeric_targets_change_ranking():
    near = song(1, "A"); near["valence"] = 0.9
    far = song(2, "B"); far["valence"] = 0.1
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.9}
    assert score_song(near, prefs) > score_song(far, prefs)


# ---- diversity actually changes the selected songs -------------------------
def test_diversity_penalty_can_swap_in_a_new_artist():
    # Three near-identical songs by one artist, one slightly worse song by another.
    songs = [song(1, "Same", energy=0.80), song(2, "Same", energy=0.79),
             song(3, "Same", energy=0.78), song(4, "Other", energy=0.70)]
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    picked = {s["artist"] for s, _, _ in recommend_songs(prefs, songs, k=3)}
    assert "Other" in picked


def test_artist_diversity_in_real_catalog():
    recs = recommend_songs({"genre": "lofi", "mood": "chill", "energy": 0.4}, SONGS, k=5)
    artists = [s["artist"] for s, _, _ in recs]
    assert len(set(artists)) >= 4


# ---- data files -------------------------------------------------------------
@pytest.mark.parametrize("path", ["data/songs.csv", "data/songs_new.csv"])
def test_data_files_load_and_have_valid_ranges(path):
    songs = load_songs(path)
    assert songs
    assert len({s["id"] for s in songs}) == len(songs)
    for s in songs:
        for key in ("energy", "valence", "danceability", "acousticness"):
            assert 0.0 <= s[key] <= 1.0, (s["title"], key)


def test_breakdown_sums_to_score_and_names_every_component():
    from src.recommender import score_breakdown
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "valence": 0.8}
    for s in SONGS[:10]:
        parts = score_breakdown(s, prefs)
        assert abs(sum(parts.values()) - score_song(s, prefs)) < 1e-12
        assert {"genre", "mood", "energy", "acoustic"} <= set(parts)


def test_detailed_results_are_consistent_with_recommend_songs():
    from src.recommender import recommend_detailed
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    detailed = recommend_detailed(prefs, SONGS, k=5)
    plain = recommend_songs(prefs, SONGS, k=5)
    assert [r["song"]["id"] for r in detailed["results"]] == [s["id"] for s, _, _ in plain]
    assert detailed["pool_size"] >= 5 and detailed["tier"] in {"both", "either", "fallback", "all"}
    for r in detailed["results"]:
        assert abs(sum(r["breakdown"].values()) - r["raw_score"]) < 1e-12
        assert abs(r["raw_score"] - r["penalty"] - r["score"]) < 1e-12
        assert r["penalty"] >= 0


def test_retrieval_tier_reports_how_far_the_filter_relaxed():
    from src.recommender import retrieve_candidates_with_tier
    assert retrieve_candidates_with_tier({}, SONGS, 5)[1] == "all"
    assert retrieve_candidates_with_tier({"genre": "nonexistent", "mood": "none"}, SONGS, 5)[1] == "fallback"


# ---- v3: similarity to liked songs ---------------------------------------------
def _mode(mode):
    """Context manager to switch the similarity mode and restore it."""
    from contextlib import contextmanager

    @contextmanager
    def cm():
        saved = recommender.SIMILARITY_MODE
        recommender.SIMILARITY_MODE = mode
        try:
            yield
        finally:
            recommender.SIMILARITY_MODE = saved
    return cm()


def _sound_song(id, artist, energy, valence=0.5, dance=0.5, acoustic=0.5, tempo=100):
    s = song(id, artist, energy=energy)
    s.update(valence=valence, danceability=dance, acousticness=acoustic, tempo_bpm=tempo)
    return s


def test_similarity_is_one_for_identical_and_zero_beyond_radius():
    a = _sound_song(1, "A", 0.2)
    far = _sound_song(2, "B", 0.2, valence=0.0, dance=0.0, acoustic=1.0, tempo=400)
    taste = recommender.taste_vector([a])
    assert abs(recommender.similarity_to_taste(a, taste) - 1.0) < 1e-12
    assert recommender.similarity_to_taste(far, taste) == 0.0


def test_taste_vector_is_the_average_sound():
    a, b = _sound_song(1, "A", 0.2), _sound_song(2, "B", 0.8)
    assert abs(recommender.taste_vector([a, b])[0] - 0.5) < 1e-12


def test_liked_songs_are_never_recommended_back_and_boost_similar_songs():
    liked = [s for s in SONGS if s["id"] in (2, 4)]  # two lofi songs
    detail = recommender.recommend_detailed({"liked_ids": [2, 4], "energy": 0.4}, SONGS, k=5)
    ids = [r["song"]["id"] for r in detail["results"]]
    assert not ({2, 4} & set(ids)) and len(ids) == 5
    assert all(r["breakdown"]["similarity"] >= 0 for r in detail["results"])
    assert detail["results"][0]["breakdown"]["similarity"] > 0
    assert any("sounds like" in r["explanation"] for r in detail["results"])


def test_no_liked_songs_means_no_similarity_and_unchanged_results():
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    base = [s["id"] for s, _, _ in recommend_songs(prefs, SONGS, k=5)]
    empty = [s["id"] for s, _, _ in recommend_songs({**prefs, "liked_ids": []}, SONGS, k=5)]
    unknown = [s["id"] for s, _, _ in recommend_songs({**prefs, "liked_ids": [99999]}, SONGS, k=5)]
    assert base == empty == unknown
    assert recommender.recommend_detailed(prefs, SONGS, 3)["results"][0]["breakdown"]["similarity"] == 0.0


def test_nearest_mode_handles_split_taste_better_than_centroid():
    quiet, loud = _sound_song(1, "A", 0.1, acoustic=0.9), _sound_song(2, "B", 0.9, acoustic=0.1)
    near_quiet = _sound_song(3, "C", 0.12, acoustic=0.88)
    prefs = {"taste_vector": recommender.taste_vector([quiet, loud]), "liked_songs": [quiet, loud]}
    with _mode("centroid"):
        centroid = recommender.taste_similarity(near_quiet, prefs)
    with _mode("nearest"):
        nearest = recommender.taste_similarity(near_quiet, prefs)
    assert nearest > centroid
