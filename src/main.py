"""Command line runner for the Music Recommender."""
from .recommender import load_songs, recommend_songs

PROFILES = [
    {"name": "Starter: pop / happy", "genre": "pop", "mood": "happy", "energy": 0.8},
    {
        "name": "High-Energy Pop",
        "genre": "pop", "mood": "happy", "energy": 0.9,
        "valence": 0.85, "danceability": 0.8, "tempo_bpm": 125, "acousticness": 0.2,
    },
    {
        "name": "Chill Lofi",
        "genre": "lofi", "mood": "chill", "energy": 0.4,
        "valence": 0.6, "danceability": 0.6, "tempo_bpm": 75, "acousticness": 0.7,
    },
    {
        "name": "Deep Intense Rock",
        "genre": "rock", "mood": "intense", "energy": 0.95,
        "valence": 0.3, "danceability": 0.65, "tempo_bpm": 150, "acousticness": 0.1,
    },
]


def show(profile: dict, songs: list, k: int = 5) -> None:
    prefs = {key: v for key, v in profile.items() if key != "name"}
    print(f"\n=== {profile['name']} ===\n{prefs}\n")
    for song, score, explanation in recommend_songs(prefs, songs, k=k):
        print(f"🎵 {song['title']} by {song['artist']}")
        print(f"   Score: {score:.2f}")
        print(f"   Reasons: {explanation}")
        print("-" * 40)


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"Loaded songs: {len(songs)}")
    for profile in PROFILES:
        show(profile, songs)


if __name__ == "__main__":
    main()
