"""
Evaluation harness: runs the recommender on fixed user profiles and reports
quality metrics. Exits non-zero if any guardrail check fails.

Usage:  python evaluation.py
"""
import sys

from src.recommender import load_songs, recommend_songs

K = 5

TEST_USERS = [
    {"genre": "pop", "mood": "happy", "energy": 0.8},
    {"genre": "rock", "mood": "intense", "energy": 0.95},
    {"genre": "lofi", "mood": "chill", "energy": 0.4},
    {"genre": "jazz", "mood": "relaxed", "energy": 0.5},
    {"genre": "electronic", "mood": "energetic", "energy": 0.9},
    {"genre": "blues", "mood": "sad", "energy": 0.9},          # conflicting preferences
    {"genre": "nonexistent", "mood": "none", "energy": 0.1},   # nothing matches
]


def genre_match(recs, genre):
    return sum(s["genre"] == genre for s, _, _ in recs) / len(recs) if recs else 0.0


def mood_match(recs, mood):
    return sum(s["mood"] == mood for s, _, _ in recs) / len(recs) if recs else 0.0


def diversity(recs):
    artists = [s["artist"] for s, _, _ in recs]
    return len(set(artists)) / len(artists) if artists else 0.0


def run_evaluation() -> bool:
    songs = load_songs("data/songs.csv")
    print(f"EVALUATION SUMMARY  (catalog: {len(songs)} songs, k={K})\n" + "=" * 50)

    ok = True
    seen_ids = set()
    rows = []
    for i, user in enumerate(TEST_USERS, 1):
        recs = recommend_songs(user, songs, k=K)
        seen_ids.update(s["id"] for s, _, _ in recs)
        g, m, d = genre_match(recs, user["genre"]), mood_match(recs, user["mood"]), diversity(recs)
        rows.append((g, m, d))

        # Guardrails: k results, no duplicates, explanations present.
        checks = {
            "returned k results": len(recs) == K,
            "no duplicate songs": len({s["id"] for s, _, _ in recs}) == len(recs),
            "all explained": all(e.strip() for _, _, e in recs),
        }
        ok &= all(checks.values())

        print(f"User {i}: {user}")
        print(f"  Genre match: {g:.2f} | Mood match: {m:.2f} | Artist diversity: {d:.2f}")
        for song, score, _ in recs:
            print(f"    - {song['title']} by {song['artist']} [{song['genre']}/{song['mood']}] ({score:.2f})")
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            print(f"  GUARDRAIL FAILED: {', '.join(failed)}")
        print()

    n = len(rows)
    print("AGGREGATE")
    print(f"  Mean genre match:      {sum(r[0] for r in rows) / n:.2f}")
    print(f"  Mean mood match:       {sum(r[1] for r in rows) / n:.2f}")
    print(f"  Mean artist diversity: {sum(r[2] for r in rows) / n:.2f}")
    print(f"  Catalog coverage:      {len(seen_ids) / len(songs):.2f} "
          f"({len(seen_ids)} distinct songs recommended across {n} users)")
    print(f"  Guardrails:            {'ALL PASSED' if ok else 'FAILURES - see above'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if run_evaluation() else 1)
