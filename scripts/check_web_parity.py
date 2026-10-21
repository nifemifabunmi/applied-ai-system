"""Verify the JS engine (web/engine.js) ranks exactly like the Python recommender."""
import json
import subprocess
import sys

from src.recommender import load_songs, recommend_songs

PROFILES = [
    {"genre": "pop", "mood": "happy", "energy": 0.8},
    {"genre": "rock", "mood": "intense", "energy": 0.95},
    {"genre": "lofi", "mood": "chill", "energy": 0.4, "likes_acoustic": True},
    {"genre": "jazz", "mood": "relaxed", "energy": 0.5, "likes_acoustic": True},
    {"genre": "blues", "mood": "sad", "energy": 0.9},
    {"genre": "nonexistent", "mood": "none", "energy": 0.1},
    {"genre": "pop", "energy": 0.9, "valence": 0.85, "danceability": 0.8, "tempo_bpm": 125, "acousticness": 0.2},
    {"mood": "chill", "energy": 0.3, "acousticness": 0.8},
    {},
]
JS = """
const e = require('./web/engine.js'), fs = require('fs');
const songs = JSON.parse(fs.readFileSync(0, 'utf8'));
const out = JSON.parse(process.env.PROFILES).map(p =>
  [3, 5, 10].map(k => e.recommend(p, songs, k).results.map(r => [r.song.id, r.adjusted])));
console.log(JSON.stringify(out));
"""


def main() -> int:
    songs = load_songs("data/songs.csv")
    env = {"PROFILES": json.dumps(PROFILES), "PATH": "/usr/bin:/bin:/usr/local/bin"}
    out = subprocess.run(["node", "-e", JS], input=json.dumps(songs), capture_output=True, text=True, env=env)
    if out.returncode:
        print(out.stderr)
        return 1
    js_results = json.loads(out.stdout)
    bad = 0
    for profile, per_k in zip(PROFILES, js_results):
        for k, js in zip([3, 5, 10], per_k):
            py = [(s["id"], score) for s, score, _ in recommend_songs(profile, songs, k=k)]
            same_ids = [a for a, _ in py] == [a for a, _ in js]
            same_scores = all(abs(a[1] - b[1]) < 1e-9 for a, b in zip(py, js))
            if not (same_ids and same_scores):
                bad += 1
                print("MISMATCH", profile, k, py, js)
    print("JS and Python agree on all profiles" if not bad else f"{bad} mismatches")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
