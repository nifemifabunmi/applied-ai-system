"""Build the single-file web demo (docs/index.html) from web/template.html,
web/engine.js and data/songs.csv. `docs/` is what GitHub Pages can serve."""
import json
from pathlib import Path

from src.recommender import load_songs

ROOT = Path(__file__).resolve().parent.parent


FIELDS = ("id", "title", "artist", "genre", "mood", "energy", "tempo_bpm", "valence", "danceability", "acousticness")


def catalog_json(csv_name: str) -> str:
    songs = [{k: s[k] for k in FIELDS} for s in load_songs(str(ROOT / csv_name))]
    # "</" and the U+2028/2029 line separators are escaped so no song title can break out of the <script> tag.
    text = json.dumps(songs, ensure_ascii=False, separators=(",", ":"))
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def main() -> None:
    html = (ROOT / "web/template.html").read_text(encoding="utf-8")
    html = html.replace("/*ENGINE*/", (ROOT / "web/engine.js").read_text(encoding="utf-8"))
    html = html.replace("/*SONGS_SMALL*/", catalog_json("data/songs.csv"))
    html = html.replace("/*SONGS_LARGE*/", catalog_json("data/catalog_large.csv"))
    out = ROOT / "docs/index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
