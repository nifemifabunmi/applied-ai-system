"""
Evaluate on real listeners' taste files (data/real_listeners/*.json).

Usage:  python -m experiments.run_v5_real
"""
from pathlib import Path

from experiments.datasets import real_listener_draws
from experiments.run_v1 import build_report, run_listeners
from src.real_listeners import MAX_LIKED, MIN_LIKED, load_real_listeners
from src.recommender import load_songs

ROOT = Path(__file__).resolve().parent.parent
MIN_PEOPLE = 3


def build(songs, files, skipped, draws=20):
    head = "# v5 results: real listeners\n\n"
    if len(files) < MIN_PEOPLE:
        body = (f"Not enough real listeners yet: found {len(files)} valid taste file(s), need at least "
                f"{MIN_PEOPLE} (5 or more recommended).\n\n"
                f"Ask people to open the app, choose the Large catalog, add between {MIN_LIKED} and {MAX_LIKED} songs they like, "
                "press **Download my taste file**, and send you the file. Put the files in `data/real_listeners/` "
                "and run `python -m experiments.run_v5_real`. See `data/real_listeners/README.md`.\n")
    else:
        listeners = real_listener_draws(songs, files, draws=draws)
        results = run_listeners(listeners, with_ablation=False)[1]
        people = "\n".join(f"| {d['nickname']} | {len(d['liked_ids'])} |" for d in files)
        body = ("## Who took part\n\n| Pseudonym | Songs liked |\n|---|---|\n" + people + "\n\n"
                + build_report(listeners, results, None, dataset="real"))
    if skipped:
        body += "\n## Files that were skipped\n\n" + "\n".join(f"- `{n}`: {'; '.join(p)}" for n, p in skipped) + "\n"
    return head + body


def main():
    songs = load_songs(str(ROOT / "data/catalog_large.csv"))
    files, skipped = load_real_listeners(str(ROOT / "data/real_listeners"), songs)
    report = build(songs, files, skipped)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports/v5_real_listeners.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
