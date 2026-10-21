from src.recommender import Song, UserProfile, Recommender, recommend_songs


def make_song(id, artist="Test Artist", genre="pop", mood="happy", energy=0.8, acousticness=0.2):
    return Song(id=id, title=f"Song {id}", artist=artist, genre=genre, mood=mood,
                energy=energy, tempo_bpm=120, valence=0.8, danceability=0.8,
                acousticness=acousticness)


def make_small_recommender() -> Recommender:
    return Recommender([
        make_song(1, genre="pop", mood="happy", energy=0.8, acousticness=0.2),
        make_song(2, genre="lofi", mood="chill", energy=0.4, acousticness=0.9),
    ])


POP_USER = UserProfile(favorite_genre="pop", favorite_mood="happy", target_energy=0.8, likes_acoustic=False)


def test_recommend_returns_songs_sorted_by_score():
    results = make_small_recommender().recommend(POP_USER, k=2)
    assert len(results) == 2
    assert results[0].genre == "pop" and results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    rec = make_small_recommender()
    explanation = rec.explain_recommendation(POP_USER, rec.songs[0])
    assert isinstance(explanation, str) and explanation.strip()
    assert "pop" in explanation


def test_class_and_function_share_one_scoring_path():
    """The OO wrapper must agree with the functional pipeline used by main.py."""
    rec = Recommender([make_song(i, artist=f"A{i}", energy=0.1 * i) for i in range(1, 9)])
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    from_class = [s.id for s in rec.recommend(POP_USER, k=5)]
    from_func = [s["id"] for s, _, _ in recommend_songs(prefs, rec._dicts, k=5)]
    assert from_class == from_func
