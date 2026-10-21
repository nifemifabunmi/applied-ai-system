from src.main import PROFILES, show
from src.recommender import load_songs


def test_cli_shows_profile_and_recommendations(capsys):
    songs = load_songs("data/songs.csv")

    show(PROFILES[0], songs, k=1)

    output = capsys.readouterr().out
    assert "User profile: {'genre': 'pop', 'mood': 'happy', 'energy': 0.8}" in output
    assert "Top recommendations:" in output
    assert "- Sunrise City by Neon Echo" in output
    assert "Score: 4.56" in output
    assert "Reasons: This song" in output
