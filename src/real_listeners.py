"""
Taste files: how real listeners contribute to the evaluation.

A listener picks songs they like in the app, downloads a small JSON file and
sends it to the project owner, who drops it in data/real_listeners/. Nothing
is uploaded automatically. A file holds only song ids (plus title and artist,
to catch catalog mismatches) and an optional pseudonym.
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SCHEMA = "vibefinder-taste-v1"
CATALOG = "large"        # taste files only make sense against the large catalog
MIN_LIKED = 8            # 3 seeds + at least 5 held-out songs
MAX_LIKED = 200


def song_key(song: Dict) -> str:
    return f"{song['title']} | {song['artist']}"


def clean_nickname(nickname: object) -> str:
    """Pseudonym only: letters, digits, space, dash, underscore (so no emails or handles)."""
    raw = str(nickname or "").strip()
    if "@" in raw:
        return "anonymous"
    name = re.sub(r"[^A-Za-z0-9 _-]", "", raw).strip()[:30]
    return name or "anonymous"


def build_taste_file(nickname: Optional[str], liked_ids: Sequence[int], songs: List[Dict],
                     exported_at: Optional[str] = None) -> Dict:
    by_id = {s["id"]: s for s in songs}
    ids = list(dict.fromkeys(int(i) for i in liked_ids))
    return {
        "schema": SCHEMA,
        "catalog": CATALOG,
        "catalog_size": len(songs),
        "nickname": clean_nickname(nickname),
        "liked_ids": ids,
        "liked_keys": [song_key(by_id[i]) for i in ids if i in by_id],
        "exported_at": exported_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "example": False,
    }


def validate_taste_file(data: Dict, songs: List[Dict]) -> List[str]:
    """Return a list of problems (empty means the file is usable)."""
    problems: List[str] = []
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return [f"schema must be '{SCHEMA}'"]
    if data.get("catalog") != CATALOG:
        problems.append(f"catalog must be '{CATALOG}'")
    ids, keys = data.get("liked_ids"), data.get("liked_keys")
    if not isinstance(ids, list) or not all(isinstance(i, int) and not isinstance(i, bool) for i in ids):
        return problems + ["liked_ids must be a list of integers"]
    if len(set(ids)) != len(ids):
        problems.append("liked_ids contains duplicates")
    if not MIN_LIKED <= len(ids) <= MAX_LIKED:
        problems.append(f"needs between {MIN_LIKED} and {MAX_LIKED} liked songs, found {len(ids)}")
    by_id = {s["id"]: s for s in songs}
    unknown = [i for i in ids if i not in by_id]
    if unknown:
        problems.append(f"{len(unknown)} ids are not in the catalog (was the catalog rebuilt?)")
    elif isinstance(keys, list) and len(keys) == len(ids):
        mismatched = [i for i, k in zip(ids, keys) if song_key(by_id[i]) != k]
        if mismatched:
            problems.append(f"{len(mismatched)} songs do not match the catalog (was the catalog rebuilt?)")
    else:
        problems.append("liked_keys must be a list with one entry per liked id")
    return problems


def load_real_listeners(directory: str, songs: List[Dict]) -> Tuple[List[Dict], List[Tuple[str, List[str]]]]:
    """
    Read every *.json in `directory`. Returns (valid_files, skipped) where skipped is a
    list of (filename, reasons). Files marked "example": true are ignored silently.
    """
    valid, skipped = [], []
    for path in sorted(Path(directory).glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as err:
            skipped.append((path.name, [f"could not read file: {type(err).__name__}"]))
            continue
        if isinstance(data, dict) and data.get("example"):
            continue
        problems = validate_taste_file(data, songs)
        if problems:
            skipped.append((path.name, problems))
        else:
            data = {**data, "nickname": clean_nickname(data.get("nickname")) or path.stem}
            valid.append(data)
    return valid, skipped
