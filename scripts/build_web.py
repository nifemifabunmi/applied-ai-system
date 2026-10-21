"""Build the single-file web demo (docs/index.html) from web/template.html,
web/engine.js and data/songs.csv. `docs/` is what GitHub Pages can serve."""
import json
from pathlib import Path

from src.recommender import load_songs

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    songs = [
        {k: s[k] for k in ("id", "title", "artist", "genre", "mood", "energy", "tempo_bpm",
                           "valence", "danceability", "acousticness")}
        for s in load_songs(str(ROOT / "data/songs.csv"))
    ]
    html = (ROOT / "web/template.html").read_text(encoding="utf-8")
    html = html.replace("/*ENGINE*/", (ROOT / "web/engine.js").read_text(encoding="utf-8"))
    # "</" is escaped so a song title can never close the <script> tag.
    html = html.replace("/*SONGS*/", json.dumps(songs, ensure_ascii=False).replace("</", "<\\/"))
    out = ROOT / "docs/index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size // 1024} KB, {len(songs)} songs)")


if __name__ == "__main__":
    main()
