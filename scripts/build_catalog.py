"""
Build data/catalog_large.csv from a Spotify-tracks style CSV
(columns: track_id, artists, track_name, popularity, danceability, energy,
valence, acousticness, tempo, track_genre, ...).

The raw file is too big to commit and is not needed at runtime; only the
derived catalog is kept.

Usage:  python -m scripts.build_catalog path/to/dataset.csv
"""
import csv
import random
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 11
PER_GENRE = 50
MAX_PER_ARTIST = 3
MIN_POPULARITY = 30   # drops obscure and placeholder tracks

# Real music genres only: the dataset also has mood-like tags (happy, sad, study...)
# and non-music categories (comedy, kids...), which are left out on purpose.
GENRES = [
    "pop", "indie-pop", "k-pop", "synth-pop", "rock", "alt-rock", "hard-rock", "punk-rock", "indie",
    "heavy-metal", "hip-hop", "r-n-b", "soul", "funk", "disco", "house", "techno", "trance", "edm",
    "drum-and-bass", "reggae", "reggaeton", "salsa", "samba", "afrobeat", "latin", "ska", "country",
    "bluegrass", "folk", "singer-songwriter", "jazz", "blues", "gospel", "classical", "piano",
    "ambient", "new-age", "trip-hop", "dancehall",
]


def derive_mood(energy: float, valence: float) -> str:
    """
    Mood from two audio features (a 3x3 energy x valence grid). This is a
    derived label, NOT a human judgement, so mood correlates with energy and
    valence by construction.
    """
    e = "low" if energy < 0.4 else "mid" if energy < 0.7 else "high"
    v = "low" if valence < 0.33 else "mid" if valence < 0.66 else "high"
    return {
        ("high", "high"): "happy", ("high", "mid"): "energetic", ("high", "low"): "intense",
        ("mid", "high"): "upbeat", ("mid", "mid"): "relaxed", ("mid", "low"): "moody",
        ("low", "high"): "peaceful", ("low", "mid"): "chill", ("low", "low"): "sad",
    }[(e, v)]


def build(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path, index_col=0)
    df = df[df.track_genre.isin(GENRES)]
    df = df.dropna(subset=["artists", "track_name"])
    df = df[(df.popularity >= MIN_POPULARITY) & (df.tempo > 0)]
    # A track can be listed under several genres: keep one, chosen at random.
    df = df.sample(frac=1, random_state=SEED)
    df = df.drop_duplicates("track_id").drop_duplicates(["track_name", "artists"])

    rng = random.Random(SEED)
    artist_counts, rows = {}, []
    for genre in GENRES:
        pool = df[df.track_genre == genre].to_dict("records")
        rng.shuffle(pool)
        taken = 0
        for r in pool:
            artist = r["artists"].split(";")[0].strip()
            if artist_counts.get(artist, 0) >= MAX_PER_ARTIST:
                continue
            artist_counts[artist] = artist_counts.get(artist, 0) + 1
            rows.append({
                "title": r["track_name"].strip(), "artist": r["artists"].replace(";", ", "),
                "genre": genre, "mood": derive_mood(r["energy"], r["valence"]),
                "energy": round(r["energy"], 3), "tempo_bpm": round(r["tempo"], 1),
                "valence": round(r["valence"], 3), "danceability": round(r["danceability"], 3),
                "acousticness": round(r["acousticness"], 3), "track_id": r["track_id"],
                # Hidden columns: never used by the recommender, only by the audio-taste
                # evaluation protocol (experiments/datasets.py).
                "gt_speechiness": round(r["speechiness"], 4), "gt_instrumentalness": round(r["instrumentalness"], 4),
                "gt_liveness": round(r["liveness"], 4), "gt_loudness": round(r["loudness"], 2),
            })
            taken += 1
            if taken == PER_GENRE:
                break
    out = pd.DataFrame(rows)
    out.insert(0, "id", range(1, len(out) + 1))
    return out


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    out = build(sys.argv[1])
    path = ROOT / "data/catalog_large.csv"
    out.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, encoding="utf-8")
    print(f"Wrote {path} ({len(out)} songs, {out.genre.nunique()} genres, {out.artist.nunique()} artists)")
    print(out.mood.value_counts().to_string())


if __name__ == "__main__":
    main()
