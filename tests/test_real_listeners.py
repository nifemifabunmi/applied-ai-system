import json
import random
import tempfile
from pathlib import Path

from experiments.datasets import real_listener_draws
from experiments.run_v5_real import build
from src.real_listeners import (MIN_LIKED, SCHEMA, build_taste_file, clean_nickname,
                                load_real_listeners, song_key, validate_taste_file)
from src.recommender import load_songs

LARGE = load_songs("data/catalog_large.csv")


def good_file(n=MIN_LIKED, start=0, nickname="Sam"):
    return build_taste_file(nickname, [s["id"] for s in LARGE[start:start + n]], LARGE)


def test_nickname_is_reduced_to_a_pseudonym():
    assert clean_nickname("Sam K.") == "Sam K"
    assert clean_nickname("sam@example.com") == "anonymous"
    assert clean_nickname("   ") == "anonymous" and clean_nickname(None) == "anonymous"
    assert clean_nickname(42) == "42"
    assert len(clean_nickname("x" * 100)) == 30


def test_build_taste_file_dedupes_and_records_keys():
    ids = [LARGE[0]["id"], LARGE[1]["id"], LARGE[0]["id"]]
    data = build_taste_file("Sam", ids, LARGE)
    assert data["schema"] == SCHEMA and data["liked_ids"] == ids[:2] and data["example"] is False
    assert data["liked_keys"] == [song_key(LARGE[0]), song_key(LARGE[1])]
    assert data["catalog_size"] == len(LARGE)


def test_valid_file_has_no_problems():
    assert validate_taste_file(good_file(), LARGE) == []


def test_validation_catches_each_kind_of_bad_file():
    f = good_file()
    assert validate_taste_file({**f, "schema": "other"}, LARGE)
    assert validate_taste_file({**f, "catalog": "small"}, LARGE)
    assert validate_taste_file(good_file(MIN_LIKED - 1), LARGE)                      # too few
    assert validate_taste_file({**f, "liked_ids": f["liked_ids"][:-1] + [10**6]}, LARGE)  # unknown id
    assert validate_taste_file({**f, "liked_ids": f["liked_ids"][:-1] + [f["liked_ids"][0]]}, LARGE)  # duplicate
    swapped = {**f, "liked_keys": list(reversed(f["liked_keys"]))}
    assert validate_taste_file(swapped, LARGE)                                       # keys do not match ids
    assert validate_taste_file({**f, "liked_ids": ["a"] * 8}, LARGE)
    assert validate_taste_file([], LARGE)


def test_loader_skips_examples_reports_bad_files_and_cleans_names():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "ok.json").write_text(json.dumps(good_file(nickname="a@b")), encoding="utf-8")
        (d / "example.json").write_text(json.dumps({**good_file(), "example": True}), encoding="utf-8")
        (d / "short.json").write_text(json.dumps(good_file(MIN_LIKED - 2)), encoding="utf-8")
        (d / "broken.json").write_text("{not json", encoding="utf-8")
        valid, skipped = load_real_listeners(str(d), LARGE)
    assert [v["nickname"] for v in valid] == ["anonymous"]
    assert sorted(n for n, _ in skipped) == ["broken.json", "short.json"]


def test_shipped_data_folder_contains_no_real_listeners_yet():
    valid, skipped = load_real_listeners("data/real_listeners", LARGE)
    assert valid == [] and skipped == []          # only the ignored example file


def test_draws_keep_seeds_out_of_the_answer_and_group_by_person():
    files = [good_file(10, 0, "A"), good_file(10, 100, "B")]
    listeners = real_listener_draws(LARGE, files, draws=4)
    assert len(listeners) == 8 and {l["group"] for l in listeners} == {"A", "B"}
    for l in listeners:
        assert len(l["seeds"]) == 3 and len(l["relevant"]) == 7
        assert {s["id"] for s in l["seeds"]}.isdisjoint(l["relevant"])


def test_report_without_enough_people_explains_what_to_do():
    text = build(LARGE, [good_file()], [])
    assert "Not enough real listeners" in text and "Download my taste file" in text


def test_report_with_enough_people_runs_end_to_end():
    rng = random.Random(3)
    files = [build_taste_file(f"P{i}", [s["id"] for s in rng.sample(LARGE, 10)], LARGE) for i in range(3)]
    text = build(LARGE, files, [("bad.json", ["too few"])], draws=2)
    assert "Who took part" in text and "P0" in text and "Not enough" not in text
    assert "This system + liked songs (default blend)" in text and "`bad.json`" in text
