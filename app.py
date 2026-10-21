"""
VibeFinder: Streamlit frontend for the music recommender.

Run with:  streamlit run app.py

This app imports and runs the real recommender (src/recommender.py); it has
no scoring logic of its own.
"""
import html

import streamlit as st

from src.recommender import load_songs, recommend_detailed

st.set_page_config(page_title="VibeFinder", page_icon="🎧", layout="wide")

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
PRESETS = {
    "Upbeat pop": {"genre": "pop", "mood": "happy", "energy": 0.8, "acoustic": "Avoid it"},
    "Study lofi": {"genre": "lofi", "mood": "chill", "energy": 0.4, "acoustic": "Prefer it"},
    "Gym rock": {"genre": "rock", "mood": "intense", "energy": 0.95, "acoustic": "Avoid it"},
    "Late-night jazz": {"genre": "jazz", "mood": "relaxed", "energy": 0.5, "acoustic": "Prefer it"},
    "Sad but loud blues": {"genre": "blues", "mood": "sad", "energy": 0.9, "acoustic": "Avoid it"},
}
TUNE = {  # key in prefs -> (label, min, max, step, default)
    "acousticness": ("Acousticness", 0.0, 1.0, 0.05, 0.5),
    "valence": ("Positivity", 0.0, 1.0, 0.05, 0.6),
    "danceability": ("Danceability", 0.0, 1.0, 0.05, 0.6),
    "tempo_bpm": ("Tempo (bpm)", 60, 180, 5, 110),
}
GROUPS = [  # label, colour, scoring components that belong to it
    ("Genre", "#1f9a8a", ["genre"]),
    ("Mood", "#e0a030", ["mood"]),
    ("Energy", "#5b5fef", ["energy"]),
    ("Sound", "#d9567b", ["acoustic", "valence", "danceability", "tempo"]),
    ("Sounds like your songs", "#8a9a2b", ["similarity"]),
]
TIER_TEXT = {
    "both": "matched your genre and mood",
    "either": "matched your genre or mood (too few matched both)",
    "fallback": "too few matches, so the whole catalog was used",
    "all": "no genre or mood chosen, so every song was considered",
    "liked": "your liked songs were used, so every other song was considered",
}
MINUS = "\u2212"
ANY = "Any"


CATALOGS = {
    "Small demo (50 songs)": "data/songs.csv",
    "Large (about 2,000 songs)": "data/catalog_large.csv",
}


@st.cache_data
def get_songs(path):
    return load_songs(path)


st.session_state.setdefault("catalog", "Small demo (50 songs)")
songs = get_songs(CATALOGS[st.session_state["catalog"]])


def options_for(key):
    counts = {}
    for s in songs:
        counts[s[key]] = counts.get(s[key], 0) + 1
    return sorted(counts, key=lambda v: (-counts[v], v)), counts


genres, genre_counts = options_for("genre")
moods, mood_counts = options_for("mood")

# --------------------------------------------------------------------------
# State: defaults and preset buttons
# --------------------------------------------------------------------------
defaults = {"genre": "pop", "mood": "happy", "energy": 0.8, "acoustic": "Avoid it", "k": 5, "liked": []}
for key, value in defaults.items():
    st.session_state.setdefault(key, value)

# Switching catalogs: liked-song ids belong to the old catalog, and a chosen genre or
# mood may not exist in the new one (a preset can also pick one that does not), so reset those.
if st.session_state.get("_last_catalog") != st.session_state["catalog"]:
    st.session_state["liked"] = []
    st.session_state["_last_catalog"] = st.session_state["catalog"]
if st.session_state["genre"] not in [ANY] + genres:
    st.session_state["genre"] = ANY
if st.session_state["mood"] not in [ANY] + moods:
    st.session_state["mood"] = ANY


def apply_preset(name):
    for key, value in PRESETS[name].items():
        st.session_state[key] = value
    for key in TUNE:
        st.session_state[f"use_{key}"] = False
    st.session_state["liked"] = []


# --------------------------------------------------------------------------
# Sidebar: taste controls
# --------------------------------------------------------------------------
with st.sidebar:
    st.radio("Catalog", list(CATALOGS), key="catalog")
    st.header("Your taste")
    st.selectbox("Genre", [ANY] + genres, key="genre",
                 format_func=lambda g: g if g == ANY else f"{g} ({genre_counts[g]})")
    st.selectbox("Mood", [ANY] + moods, key="mood",
                 format_func=lambda m: m if m == ANY else f"{m} ({mood_counts[m]})")
    st.slider("Energy", 0.0, 1.0, step=0.05, key="energy")
    st.radio("Acoustic sound", ["Prefer it", "Avoid it"], key="acoustic", horizontal=True)

    with st.expander("Fine-tune the sound"):
        st.caption("Turn a target on to add it to the score. Acousticness replaces the setting above.")
        for key, (label, lo, hi, step, default) in TUNE.items():
            st.checkbox(label, key=f"use_{key}")
            st.slider(label, lo, hi, default, step=step, key=f"val_{key}",
                      disabled=not st.session_state.get(f"use_{key}", False),
                      label_visibility="collapsed")

    title_of = {s["id"]: f"{s['title']} ({s['artist']})" for s in songs}
    st.multiselect("Songs you already like (optional)", list(title_of), key="liked",
                   format_func=lambda i: title_of[i],
                   help="Recommendations will sound like these and never repeat them.")

    st.select_slider("Songs to show", options=[3, 5, 10], key="k")

prefs = {
    "genre": "" if st.session_state["genre"] == ANY else st.session_state["genre"],
    "mood": "" if st.session_state["mood"] == ANY else st.session_state["mood"],
    "energy": st.session_state["energy"],
    "likes_acoustic": st.session_state["acoustic"] == "Prefer it",
}
for key in TUNE:
    if st.session_state.get(f"use_{key}", False):
        prefs[key] = st.session_state[f"val_{key}"]
if st.session_state["liked"]:
    prefs["liked_ids"] = list(st.session_state["liked"])

# --------------------------------------------------------------------------
# Main: run the real pipeline and show every step
# --------------------------------------------------------------------------
st.title("VibeFinder")
st.write("Describe your taste and see which songs fit, with the points behind every pick.")

preset_cols = st.columns(len(PRESETS))
for col, name in zip(preset_cols, PRESETS):
    col.button(name, on_click=apply_preset, args=(name,), use_container_width=True)

detail = recommend_detailed(prefs, songs, k=st.session_state["k"])
results = detail["results"]

c1, c2, c3 = st.columns(3)
c1.metric("Songs in the catalog", len(songs))
c2.metric("Candidates", detail["pool_size"])
c3.metric("Picks", len(results))
st.caption(f"Candidates {TIER_TEXT[detail['tier']]}. Picks are spread across artists.")
if st.session_state["liked"]:
    st.caption("Liked songs pull the results toward music that sounds like them. How strongly is a trade-off "
               "between trusting genre labels and trusting sound; see reports/v4_results.md.")


def group_points(breakdown, parts):
    return sum(breakdown[p] for p in parts)


max_positive = max(
    [sum(max(0.0, group_points(r["breakdown"], g[2])) for g in GROUPS) for r in results] + [0.001]
)

st.markdown(
    " &nbsp; ".join(f'<span style="color:{c}">&#9632;</span> {label}' for label, c, _ in GROUPS),
    unsafe_allow_html=True,
)

for rank, r in enumerate(results, start=1):
    song = r["song"]
    with st.container(border=True):
        left, right = st.columns([5, 1])
        left.subheader(f"{rank}. {song['title']}")
        left.caption(f"{song['artist']}  |  {song['genre']} / {song['mood']}  |  energy {song['energy']:.2f}")
        right.metric("Score", f"{r['score']:.2f}")

        points = [(label, color, group_points(r["breakdown"], parts)) for label, color, parts in GROUPS]
        segments = "".join(
            f'<div style="width:{v / max_positive * 100:.1f}%;background:{c}" title="{html.escape(label)} {v:.2f}"></div>'
            for label, c, v in points if v > 0
        )
        st.markdown(
            '<div style="display:flex;height:12px;border-radius:6px;overflow:hidden;'
            f'background:rgba(128,128,128,0.2)">{segments}</div>',
            unsafe_allow_html=True,
        )
        chips = [
            f'<span style="color:{c if v > 0 else "gray"}">&#9632;</span> {html.escape(label)} '
            f'{"+" if v > 0 else MINUS}{abs(v):.2f}'
            for label, c, v in points if abs(v) > 0.004
        ]
        if r["penalty"] > 0.004:
            chips.append(f'<span style="color:gray">&#9632;</span> Repeat artist {MINUS}{r["penalty"]:.2f}')
        st.markdown(" &nbsp; ".join(chips), unsafe_allow_html=True)

        st.write(r["explanation"].replace("$", "\\$"))

st.caption(
    "Scores come from a transparent rule-based model, not machine learning. "
    "Each bar is split by what earned the points. A pick by an artist already on the list "
    "loses 0.70 points, so one artist can\u2019t fill the page."
)
