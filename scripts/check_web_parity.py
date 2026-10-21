"""Verify the JS engine (web/engine.js) ranks exactly like the Python recommender,
on both catalogs, with and without liked songs."""
import json
import subprocess
import sys

from src.recommender import load_songs, recommend_detailed

CATALOGS = {"small": "data/songs.csv", "large": "data/catalog_large.csv"}
BASE = [
    {"genre": "pop", "mood": "happy", "energy": 0.8},
    {"genre": "rock", "mood": "intense", "energy": 0.95},
    {"genre": "jazz", "mood": "relaxed", "energy": 0.5, "likes_acoustic": True},
    {"genre": "blues", "mood": "sad", "energy": 0.9},
    {"genre": "nonexistent", "mood": "none", "energy": 0.1},
    {"genre": "pop", "energy": 0.9, "valence": 0.85, "danceability": 0.8, "tempo_bpm": 125, "acousticness": 0.2},
    {"mood": "chill", "energy": 0.3, "acousticness": 0.8},
    {"energy": 7.0},
    {},
]
LIKED = [   # liked_ids are filled in per catalog (the first few ids, and a spread of ids)
    {"genre": "pop", "mood": "happy", "energy": 0.8, "liked": [0, 1]},
    {"energy": 0.4, "likes_acoustic": True, "liked": [2, 3, 4]},
    {"genre": "jazz", "mood": "relaxed", "energy": 0.5, "liked": [5, 6, 7, 8, 9], "valence": 0.6},
]
KS = [3, 5, 10]
JS = """
const e = require('./web/engine.js'), fs = require('fs');
const { songs, profiles, ks } = JSON.parse(fs.readFileSync(0, 'utf8'));
console.log(JSON.stringify(profiles.map(p => ks.map(k => {
  const r = e.recommend(p, songs, k);
  return { tier: r.tier, pool: r.poolSize, picks: r.results.map(x => [x.song.id, x.adjusted, x.explanation]) };
}))));
"""


def main() -> int:
    bad = total = 0
    for name, path in CATALOGS.items():
        songs = load_songs(path)
        ids = [s["id"] for s in songs]
        profiles = list(BASE)
        for p in LIKED:
            profiles.append({**{k: v for k, v in p.items() if k != "liked"}, "liked_ids": [ids[i] for i in p["liked"]]})
        out = subprocess.run(["node", "-e", JS], input=json.dumps({"songs": songs, "profiles": profiles, "ks": KS}),
                             capture_output=True, text=True)
        if out.returncode:
            print(out.stderr)
            return 1
        js_results = json.loads(out.stdout)
        for profile, per_k in zip(profiles, js_results):
            for k, js in zip(KS, per_k):
                total += 1
                py = recommend_detailed(profile, songs, k=k)
                py_picks = [(r["song"]["id"], r["score"]) for r in py["results"]]
                same = (py["tier"] == js["tier"] and py["pool_size"] == js["pool"]
                        and [i for i, _ in py_picks] == [i for i, _, _ in js["picks"]]
                        and all(abs(a - b) < 1e-9 for (_, a), (_, b, _) in zip(py_picks, js["picks"])))
                if not same:
                    bad += 1
                    print("MISMATCH", name, profile, k, py["tier"], js["tier"], py_picks, js["picks"])
    print(f"JS and Python agree on all {total} cases (2 catalogs, with and without liked songs)" if not bad
          else f"{bad} of {total} cases differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
