"""Runs the real Streamlit app headlessly. Skipped when Streamlit is not installed (it runs in CI)."""
import pytest
from pathlib import Path

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402


def run_app(**state):
    app_path = Path(__file__).resolve().parents[1] / "app.py"
    at = AppTest.from_file(app_path, default_timeout=60)
    for key, value in state.items():
        at.session_state[key] = value
    return at.run()


def test_app_runs_with_default_settings():
    assert not run_app().exception


def test_app_runs_on_the_large_catalog_with_liked_songs():
    at = run_app(catalog="Large (about 2,000 songs)", liked=[1, 2, 3])
    assert not at.exception


def test_app_survives_a_preset_genre_missing_from_the_chosen_catalog():
    at = run_app(catalog="Large (about 2,000 songs)", genre="lofi", mood="happy")
    assert not at.exception
