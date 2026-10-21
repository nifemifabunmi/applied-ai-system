"""
Two-stage music recommender: retrieval -> scoring -> diversity-aware ranking -> explanation.

There is ONE scoring path (`score_song`) and ONE ranking path (`recommend_songs`).
The object-oriented `Recommender` class is a thin wrapper around them, so tests
on the class exercise the same logic the command-line app uses.
"""
import csv
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Tunable weights (all in one place so experiments are easy to run and report)
# --------------------------------------------------------------------------
WEIGHTS = {
    "genre": 1.25,
    "mood": 1.5,
    "energy": 2.0,            # max points for a perfect energy match
    "acoustic": 0.6,          # +/- acousticness * weight (likes_acoustic flag)
    "acoustic_target": 0.8,   # closeness to an explicit acousticness target
    "valence": 0.5,
    "danceability": 0.5,
    "tempo": 0.5,
    "popularity": 1.0,        # popularity/100 * weight (only if the song has it)
    "decade": 0.5,
    "detailed_mood": 0.7,
    "instrumental": 0.5,
    "language": 0.3,
    "similarity": 3.0,        # closeness to the user's liked songs (trade-off analysed in reports/v4_results.md)
}
ENERGY_TOLERANCE = 0.5        # energy gap at which the bonus reaches 0
TEMPO_TOLERANCE_BPM = 60.0    # tempo gap at which the bonus reaches 0
ARTIST_REPEAT_PENALTY = 0.7   # subtracted per earlier pick by the same artist
SIMILARITY_RADIUS = 1.0       # feature-space distance at which similarity reaches 0
SIMILARITY_FEATURES = ("energy", "valence", "danceability", "acousticness")
SIMILARITY_MODE = "centroid"  # "centroid": average sound of liked songs; "nearest": closest single liked song
FILTER_WHEN_LIKED = False     # with liked songs, search the whole catalog instead of only the seed genre/mood


@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
_OPTIONAL_FLOATS = ("instrumentalness",)
_OPTIONAL_STRS = ("release_decade", "detailed_mood", "language")


def load_songs(csv_path: str) -> List[Dict]:
    """
    Load songs from a CSV file.

    Required columns: id, title, artist, genre, mood, energy, tempo_bpm,
    valence, danceability, acousticness. Optional columns (popularity,
    release_decade, detailed_mood, instrumentalness, language) are only set
    when present and non-empty, so they never inject fake default values
    into scoring.
    """
    songs: List[Dict] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            song = {
                "id": int(row["id"]),
                "title": row["title"],
                "artist": row["artist"],
                "genre": row["genre"],
                "mood": row["mood"],
                "energy": float(row["energy"]),
                "tempo_bpm": float(row["tempo_bpm"]),
                "valence": float(row["valence"]),
                "danceability": float(row["danceability"]),
                "acousticness": float(row["acousticness"]),
            }
            if row.get("popularity"):
                song["popularity"] = int(row["popularity"])
            for key in _OPTIONAL_FLOATS:
                if row.get(key):
                    song[key] = float(row[key])
            for key in _OPTIONAL_STRS:
                if row.get(key):
                    song[key] = row[key]
            songs.append(song)
    return songs


# --------------------------------------------------------------------------
# Stage 1: retrieval (progressively relaxed, always yields >= k candidates)
# --------------------------------------------------------------------------
def retrieve_candidates_with_tier(user_prefs: Dict, songs: List[Dict], k: int) -> Tuple[List[Dict], str]:
    """
    Narrow the catalog using genre/mood, relaxing the filter until at least
    `k` candidates exist. Returns (candidates, tier) where tier is one of
    "all" (no genre/mood given), "both" (genre AND mood), "either"
    (genre OR mood) or "fallback" (whole catalog).
    """
    genre = user_prefs.get("genre")
    mood = user_prefs.get("mood")
    if not genre and not mood:
        return list(songs), "all"

    def matches_genre(s):
        return not genre or s["genre"] == genre

    def matches_mood(s):
        return not mood or s["mood"] == mood

    tiers = [
        ("both", [s for s in songs if matches_genre(s) and matches_mood(s)]),
        ("either", [s for s in songs if matches_genre(s) or matches_mood(s)]),
        ("fallback", list(songs)),
    ]
    for name, pool in tiers:
        if len(pool) >= k:
            logger.info("Retrieval tier '%s' -> %d candidates", name, len(pool))
            return pool, name
    return list(songs), "fallback"


def retrieve_candidates(user_prefs: Dict, songs: List[Dict], k: int) -> List[Dict]:
    """Candidate songs only (see `retrieve_candidates_with_tier`)."""
    return retrieve_candidates_with_tier(user_prefs, songs, k)[0]


# --------------------------------------------------------------------------
# Stage 2: scoring
# --------------------------------------------------------------------------
def _closeness(a: float, b: float, tolerance: float) -> float:
    """1.0 for identical values, falling linearly to 0.0 at `tolerance` apart."""
    return max(0.0, 1.0 - abs(a - b) / tolerance)


def sound_vector(song: Dict) -> List[float]:
    """A song's position in audio-feature space (all values roughly 0 to 1)."""
    return [song[f] for f in SIMILARITY_FEATURES] + [song["tempo_bpm"] / 200.0]


def taste_vector(liked_songs: List[Dict]) -> List[float]:
    """Average sound of the songs a user already likes."""
    vectors = [sound_vector(s) for s in liked_songs]
    return [sum(col) / len(col) for col in zip(*vectors)]


def sound_distance(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def similarity_to_taste(song: Dict, taste: List[float]) -> float:
    """1.0 for a song that sounds exactly like the user's taste, falling to 0.0 at SIMILARITY_RADIUS."""
    return max(0.0, 1.0 - sound_distance(sound_vector(song), taste) / SIMILARITY_RADIUS)


def taste_similarity(song: Dict, prefs: Dict) -> float:
    """Similarity (0 to 1) to the user's liked songs under the active SIMILARITY_MODE."""
    liked = prefs.get("liked_songs") or []
    if SIMILARITY_MODE == "nearest" and liked:
        v = sound_vector(song)
        return max(max(0.0, 1.0 - sound_distance(sound_vector(s), v) / SIMILARITY_RADIUS) for s in liked)
    return similarity_to_taste(song, prefs["taste_vector"])


def nearest_liked(song: Dict, liked_songs: List[Dict]) -> Optional[Dict]:
    if not liked_songs:
        return None
    v = sound_vector(song)
    return min(liked_songs, key=lambda s: sound_distance(sound_vector(s), v))


def score_breakdown(song: Dict, prefs: Dict) -> Dict[str, float]:
    """
    Points contributed by each scoring component for one song (0.0 when a
    component does not apply). `score_song` is the sum of these, so the
    breakdown can be shown to users without a second scoring path.
    """
    w = WEIGHTS
    parts = {k: 0.0 for k in (
        "genre", "mood", "energy", "acoustic", "valence", "danceability",
        "tempo", "popularity", "decade", "detailed_mood", "instrumental", "language", "similarity",
    )}

    if prefs.get("genre") and song["genre"] == prefs["genre"]:
        parts["genre"] = w["genre"]
    if prefs.get("mood") and song["mood"] == prefs["mood"]:
        parts["mood"] = w["mood"]

    # Smooth linear energy bonus (no cliffs between neighbouring values).
    parts["energy"] = w["energy"] * _closeness(song["energy"], prefs.get("energy", 0.5), ENERGY_TOLERANCE)

    # Acousticness: explicit target wins, otherwise the likes_acoustic flag.
    if "acousticness" in prefs:
        parts["acoustic"] = w["acoustic_target"] * _closeness(song["acousticness"], prefs["acousticness"], 1.0)
    elif prefs.get("likes_acoustic", False):
        parts["acoustic"] = song["acousticness"] * w["acoustic"]
    else:
        parts["acoustic"] = -song["acousticness"] * w["acoustic"]

    # Optional numeric targets (only used when the user supplies them).
    if "valence" in prefs:
        parts["valence"] = w["valence"] * _closeness(song["valence"], prefs["valence"], 1.0)
    if "danceability" in prefs:
        parts["danceability"] = w["danceability"] * _closeness(song["danceability"], prefs["danceability"], 1.0)
    if "tempo_bpm" in prefs:
        parts["tempo"] = w["tempo"] * _closeness(song["tempo_bpm"], prefs["tempo_bpm"], TEMPO_TOLERANCE_BPM)

    # Optional song metadata (only used when both sides have it).
    if "popularity" in song:
        parts["popularity"] = w["popularity"] * song["popularity"] / 100.0
    if prefs.get("release_decade") and song.get("release_decade") == prefs["release_decade"]:
        parts["decade"] = w["decade"]
    if prefs.get("detailed_mood") and song.get("detailed_mood") == prefs["detailed_mood"]:
        parts["detailed_mood"] = w["detailed_mood"]
    if prefs.get("prefers_instrumental") and "instrumentalness" in song:
        parts["instrumental"] = song["instrumentalness"] * w["instrumental"]
    if prefs.get("language") and song.get("language") == prefs["language"]:
        parts["language"] = w["language"]
    # Similarity to the user's liked songs (only when `taste_vector` was supplied).
    if prefs.get("taste_vector"):
        parts["similarity"] = w["similarity"] * taste_similarity(song, prefs)
    return parts


def score_song(song: Dict, prefs: Dict) -> float:
    """Score one song against a preference dict. Higher is better."""
    return sum(score_breakdown(song, prefs).values())


# --------------------------------------------------------------------------
# Stage 3: diversity-aware ranking (greedy, MMR-style)
# --------------------------------------------------------------------------
def rank_with_diversity(scored: List[Tuple[Dict, float]], k: int) -> List[Tuple[Dict, float]]:
    """
    Greedily pick k songs. At each step every remaining song is penalised by
    ARTIST_REPEAT_PENALTY for each time its artist was already picked, so a
    lower-scoring song by a new artist can overtake a repeat artist. Because
    the penalty is applied *during* selection (not after the top-k cut), it
    can actually change which songs make the list.
    """
    remaining = list(scored)
    picked: List[Tuple[Dict, float]] = []
    artist_counts: Dict[str, int] = {}
    while remaining and len(picked) < k:
        best_i, best_adj = 0, float("-inf")
        for i, (song, score) in enumerate(remaining):
            adj = score - ARTIST_REPEAT_PENALTY * artist_counts.get(song["artist"], 0)
            if adj > best_adj:
                best_i, best_adj = i, adj
        song, _ = remaining.pop(best_i)
        artist_counts[song["artist"]] = artist_counts.get(song["artist"], 0) + 1
        picked.append((song, best_adj))
    return picked


# --------------------------------------------------------------------------
# Stage 4: explanation
# --------------------------------------------------------------------------
def retrieve_similar_songs(target: Dict, all_songs: List[Dict], top_n: int = 2) -> List[Dict]:
    """Songs sharing genre, mood or artist with `target` (most overlap first)."""
    def overlap(s):
        return (s["genre"] == target["genre"]) + (s["mood"] == target["mood"]) + (s["artist"] == target["artist"])

    similar = [s for s in all_songs if s["id"] != target["id"] and overlap(s) > 0]
    similar.sort(key=overlap, reverse=True)
    return similar[:top_n]


def explain_song(song: Dict, prefs: Dict, all_songs: Optional[List[Dict]] = None) -> str:
    """Plain-language reasons this song was recommended."""
    reasons = []
    if prefs.get("genre") and song["genre"] == prefs["genre"]:
        reasons.append(f"matches your favorite genre '{prefs['genre']}'")
    if prefs.get("mood") and song["mood"] == prefs["mood"]:
        reasons.append(f"matches your favorite mood '{prefs['mood']}'")

    energy_diff = abs(song["energy"] - prefs.get("energy", 0.5))
    if energy_diff < 0.1:
        reasons.append("has energy level very close to your target")
    elif energy_diff < 0.3:
        reasons.append("has energy level somewhat close to your target")

    if "acousticness" not in prefs:
        if prefs.get("likes_acoustic", False) and song["acousticness"] > 0.5:
            reasons.append("is acoustic, which you prefer")
        elif not prefs.get("likes_acoustic", False) and song["acousticness"] < 0.3:
            reasons.append("is not very acoustic, matching your preference")

    if prefs.get("taste_vector") and taste_similarity(song, prefs) >= 0.6:
        close = nearest_liked(song, prefs.get("liked_songs", []))
        if close:
            reasons.append(f"sounds like '{close['title']}', which you like")
        else:
            reasons.append("sounds like the songs you like")
    if prefs.get("release_decade") and song.get("release_decade") == prefs["release_decade"]:
        reasons.append(f"is from the {song['release_decade']} you like")
    if prefs.get("language") and song.get("language") == prefs["language"]:
        reasons.append(f"is in {song['language']}")

    text = (
        f"This song {', '.join(reasons)}."
        if reasons
        else "This song may not perfectly match your preferences, but give it a listen!"
    )
    if all_songs:
        similar = retrieve_similar_songs(song, all_songs)
        if similar:
            titles = ", ".join(f"'{s['title']}' by {s['artist']}" for s in similar)
            text += f" Similar songs you might like: {titles}."
    return text


# --------------------------------------------------------------------------
# Public pipeline
# --------------------------------------------------------------------------
def recommend_detailed(user_prefs: Dict, songs: List[Dict], k: int = 5) -> Dict:
    """
    Run the full pipeline and return everything a UI needs:
      {"tier": str, "pool_size": int, "results": [
          {"song", "score" (after diversity penalty), "raw_score", "penalty",
           "breakdown" (points per scoring component), "explanation"}, ...]}
    """
    if k <= 0 or not songs:
        return {"tier": "all", "pool_size": 0, "results": []}

    prefs = dict(user_prefs)
    if "energy" in prefs and not 0.0 <= prefs["energy"] <= 1.0:
        logger.warning("energy %.2f outside [0, 1]; clamping", prefs["energy"])
        prefs["energy"] = min(1.0, max(0.0, prefs["energy"]))

    # Songs the user already likes shape the taste vector but are never recommended back.
    liked_ids = set(prefs.pop("liked_ids", None) or [])
    pool_songs = songs
    if liked_ids:
        liked = [s for s in songs if s["id"] in liked_ids]
        if liked:
            prefs["taste_vector"] = taste_vector(liked)
            prefs["liked_songs"] = liked
            pool_songs = [s for s in songs if s["id"] not in liked_ids]

    if "taste_vector" in prefs and not FILTER_WHEN_LIKED:
        # Liked songs describe taste better than one genre/mood label, so do not
        # confine the search to the seed genre/mood (v4 showed that cap hurts at scale).
        candidates, tier = list(pool_songs), "liked"
    else:
        candidates, tier = retrieve_candidates_with_tier(prefs, pool_songs, k)
    scored = [(s, score_song(s, prefs)) for s in candidates]
    raw = {s["id"]: score for s, score in scored}
    top = rank_with_diversity(scored, k)
    results = [
        {
            "song": s,
            "score": adjusted,
            "raw_score": raw[s["id"]],
            "penalty": raw[s["id"]] - adjusted,
            "breakdown": score_breakdown(s, prefs),
            "explanation": explain_song(s, prefs, songs),
        }
        for s, adjusted in top
    ]
    return {"tier": tier, "pool_size": len(candidates), "results": results}


def recommend_songs(user_prefs: Dict, songs: List[Dict], k: int = 5) -> List[Tuple[Dict, float, str]]:
    """
    Return up to `k` (song, score, explanation) tuples, best first. Returns
    exactly min(k, len(songs)) results.
    """
    detailed = recommend_detailed(user_prefs, songs, k)
    return [(r["song"], r["score"], r["explanation"]) for r in detailed["results"]]


# --------------------------------------------------------------------------
# OO wrapper (kept for the starter tests / API); delegates to the pipeline
# --------------------------------------------------------------------------
class Recommender:
    def __init__(self, songs: List[Song]):
        self.songs = songs
        self._by_id = {s.id: s for s in songs}
        self._dicts = [asdict(s) for s in songs]

    @staticmethod
    def _prefs(user: UserProfile) -> Dict:
        return {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }

    def score_song(self, song: Song, user: UserProfile) -> float:
        return score_song(asdict(song), self._prefs(user))

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        results = recommend_songs(self._prefs(user), self._dicts, k=k)
        return [self._by_id[s["id"]] for s, _, _ in results]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        return explain_song(asdict(song), self._prefs(user), self._dicts)
