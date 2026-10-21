# Music Recommender: Retrieval + Ranking with Explanations

## Original Project
**Music Recommender Simulation** was a simple content-based recommender that suggests songs from user preferences for genre, mood and song features. The goal was to see how recommenders turn data into predictions, where bias creeps in, and how scoring choices change results.

## Summary
This version turns that starter into a small, testable recommender with retrieval, scoring, diversity-aware ranking, and explanations. It filters a song catalog, scores candidates, selects a diverse list, and writes a plain-language explanation for every recommendation. It is rule-based on purpose, so every score can be traced and explained. It does not use machine learning or an LLM.

## Architecture
![System Architecture](diagrams/architecture.png)

1. **Retrieval** (`retrieve_candidates`): narrows the catalog by genre/mood. The filter relaxes step by step (genre AND mood, then genre OR mood, then the whole catalog) until at least `k` candidates are available or the catalog is exhausted.
2. **Scoring** (`score_song`): one function used everywhere. Genre and mood matches, a smooth energy-closeness curve, and acousticness all count. Optional user targets such as valence, danceability, tempo, decade, detailed mood, instrumental preference, and language are used when supplied and supported by the data. Popularity is scored automatically when a song has that field. All weights live in `WEIGHTS`.
3. **Diversity-aware ranking** (`rank_with_diversity`): greedy selection that subtracts a penalty each time an artist repeats, so a slightly lower-scoring song by a new artist can win a slot.
4. **Explanation** (`explain_song`): lists why each song matched and suggests two similar songs (shared genre/mood/artist).

`Recommender` (the OO class) is a thin wrapper over the same functions, so class tests and the CLI exercise identical logic.

**Optional (v3/v4):** provide songs you already like and recommendations get a similarity bonus based on energy, valence, danceability, acousticness, and tempo. The app searches the full catalog rather than applying the genre/mood retrieval filter, excludes liked songs from results, and can explain which liked track sounds similar. The default similarity weight is a trade-off, not a proven optimum.

## Getting Started
Requires Python and pip. On Windows PowerShell:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt  # needed for tests, experiments, and rebuilding the large catalog
```

On macOS/Linux, activate the environment with `source .venv/bin/activate` instead. Then run commands from the project root:

```bash
streamlit run app.py      # interactive app using the Python recommender
python -m src.main        # command-line demo with four user profiles
python evaluation.py      # metrics and guardrail checks (exit code 1 on failure)
python -m pytest          # automated test suite
python -m experiments.run_v1  # compare baselines and run the v1 ablation study
python -m experiments.run_v2  # tune weights and measure sensitivity
python -m experiments.run_v3  # evaluate similarity on the small hand-labelled catalog
python -m experiments.run_v4  # compare rules/similarity under two large-catalog protocols
```

## Streamlit App (`app.py`)
The main frontend. It imports `src.recommender` directly (via `recommend_detailed`), so it shows exactly what the real system does, with no copy of the logic. Pick a catalog, genre, mood, energy, and optionally songs you already like. Recommendations show the score breakdown, retrieval funnel, repeat-artist penalty, and explanation. To deploy for free, push the repo to GitHub and create an app at share.streamlit.io pointing at `app.py`.

## Static Web Demo (optional)
A no-install, single-file copy lives in `docs/index.html`: pick a genre, mood and energy (or an example taste), and each recommendation shows a colour-coded score breakdown, the retrieval funnel (catalog, candidates, picks) and the written explanation.
```bash
python scripts/build_web.py            # rebuild docs/index.html from web/ + data/songs.csv
python scripts/check_web_parity.py     # confirm the JS engine ranks exactly like the Python one
python -m http.server -d docs 8000     # then open http://localhost:8000
```
The parity check requires Node.js in addition to the Python dependencies.
To host it free, go to GitHub, then Settings, then Pages, choose branch `main` and folder `/docs`.
Unlike the Streamlit app, this page uses `web/engine.js`, a JavaScript port of `src/recommender.py`, so the page runs entirely in the browser with no server. If you change weights or ranking in Python, mirror the change in `web/engine.js` and re-run the parity check.
The liked-song similarity feature is available in the Python recommender and Streamlit app; the static web demo does not include it.

## Data
- `data/songs.csv`: 50 songs, 48 artists, 27 genres, with 10 columns (id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness). This is the small demo catalog.
- `data/catalog_large.csv`: 1,985 tracks across 40 genres for the large-catalog app and synthetic experiments. Its mood labels are derived from energy and valence; hidden `gt_*` audio features are used only as evaluation ground truth and are not loaded by the recommender.
- `data/songs_new.csv`: 20-song sample that also has `popularity`, `release_decade`, `detailed_mood`, `instrumentalness` and `language`. These optional columns are supported by the scorer but are not in the main catalog.
- `data/eval_listeners.json`: hand-labelled listener seed songs and held-out liked songs for offline evaluation.
- `data/eval_archetypes.json`: synthetic genre-based archetypes used to generate repeatable evaluation listeners.

The large catalog was derived from a Spotify tracks dataset distributed through Kaggle. The original raw dataset is not included. To rebuild the catalog, obtain a compatible source file, review its license and terms, then run `python -m scripts.build_catalog path\to\dataset.csv`. The build tool requires the development dependencies. Check that the source terms permit redistribution before publishing the derived catalog.

## Sample Interaction
**Input:** `{"genre": "pop", "mood": "happy", "energy": 0.8}`
```
User profile: {'genre': 'pop', 'mood': 'happy', 'energy': 0.8}

Top recommendations:
- Sunrise City by Neon Echo
  Score: 4.56
  Reasons: This song matches your favorite genre 'pop', matches your favorite mood 'happy',
  has energy level very close to your target, is not very acoustic, matching your preference.
  Similar songs you might like: 'Gym Hero' by Max Pulse, 'Night Drive Loop' by Neon Echo.
----------------------------------------
```

## Evaluation (`python evaluation.py`, k=5, 7 profiles, 50-song catalog)
| Metric | Result |
|---|---|
| Mean genre match | 0.40 |
| Mean mood match | 0.43 |
| Mean artist diversity | 1.00 |
| Catalog coverage | 0.52 (26 of 50 songs recommended) |
| Guardrails (k results, no duplicates, explanations present) | all passed |

Genre match is low for rock, electronic and blues because the catalog has only one to two songs per genre. The system broadens retrieval and ranks the available candidates rather than failing. See `model_card.md`.

## Baselines and Ablation (v1)
Run `python -m experiments.run_v1` to compare the recommender with simple baselines and measure how individual system components affect results. Each listener's taste profile is derived from three seed songs; those seeds are removed from the candidate catalog, and recommendations are evaluated against that listener's held-out liked songs. Results are saved in `reports/v1_results.md`.

| Method | Precision@5 | NDCG@5 | NDCG@5 95% interval |
|---|---:|---:|---|
| Random (average of 500 draws) | 0.10 | 0.11 | 0.10 to 0.13 |
| Energy only | 0.31 | 0.31 | 0.10 to 0.54 |
| Genre only | 0.40 | 0.45 | 0.20 to 0.67 |
| Genre + mood rules, no scoring | 0.26 | 0.36 | 0.17 to 0.55 |
| Audio-feature nearest neighbours | 0.26 | 0.30 | 0.14 to 0.47 |
| **This system** | **0.46** | **0.59** | **0.41 to 0.77** |

The full system has the best average NDCG and a hit rate of 1.00 in this evaluation, but only seven listeners were labelled and the confidence intervals overlap. Treat this as an encouraging signal, not proof of general performance. The ablation suggests mood, genre, and energy contribute most to relevance; the diversity penalty and retrieval filter primarily improve variety and efficiency here.

## Weight Tuning and Sensitivity (v2)
Run `python -m experiments.run_v2` to compare the current weights with a deterministic random search over 800 configurations, evaluate tuning with leave-one-listener-out validation, and measure sensitivity to individual and joint weight changes. Outputs are saved to `reports/v2_results.md` and `reports/v2_sensitivity.png`.

The best configuration scored 0.62 NDCG@5 on all seven listeners, compared with 0.59 for the current weights. However, tuning on six listeners and evaluating on the held-out listener scored 0.56 versus 0.59 for the current weights. This suggests the apparent in-sample gain overfits the small evaluation set, so the project keeps its existing weights. The results are indicative, not conclusive, with only seven hand-labelled listeners. Sensitivity analysis shows mood and energy weights have the greatest effect; the full ranges and methodology are in the [v2 report](reports/v2_results.md) and [model card](model_card.md).

![Weight sensitivity and perturbation results](reports/v2_sensitivity.png)

## Similarity to Liked Songs (v3)
In the Streamlit app, optionally select songs you already like. The recommender excludes those songs from its results and adds a small score bonus for candidates that sound like them across energy, valence, danceability, acousticness, and tempo. Similarity can compare against the average sound of the liked songs or the closest individual liked song; the default is nearest. Run `python -m experiments.run_v3` to reproduce the comparison of both modes and several weights, including leave-one-listener-out evaluation. Full results are in [reports/v3_results.md](reports/v3_results.md).

On all seven labelled listeners, nearest-song similarity at weight 0.5 scored 0.593 NDCG@5 versus 0.586 for rules alone. In leave-one-listener-out testing, selecting among the candidate variants scored 0.549 versus 0.586 for rules alone. The small in-sample gain did not generalize. V4 therefore evaluates the feature as a trade-off under two synthetic large-catalog protocols rather than claiming improved real-user accuracy.

## Larger Catalog and Synthetic Evaluation (v4)
The 1,985-song catalog has no listening history, so the v4 evaluation uses 60 synthetic listeners (10 archetypes with six repeatable seed-song draws) under two deliberately different assumptions:

- **Genre taste:** a listener likes tracks in their archetype's listed genres; this favors genre-based rules.
- **Audio taste:** a listener likes the 100 tracks closest to their seed songs in a richer feature space, including speechiness, instrumentalness, liveness, and loudness. These hidden evaluation-only features are never supplied to the recommender; this favors similarity.

| Method (NDCG@5) | Genre taste | Audio taste |
|---|---:|---:|
| Rules only | 0.953 | 0.238 |
| Rules + centroid similarity (weight 3, no retrieval filter; shipped default) | 0.923 | 0.452 |
| Rules + centroid similarity (weight 8, no retrieval filter; worst-case selection) | 0.618 | 0.664 |
| Audio-feature nearest neighbours | 0.159 | 0.772 |

No method wins under both assumptions. The default similarity blend gives up about 0.03 NDCG under genre taste for a 0.21 gain under audio taste, compared with rules alone. A predeclared worst-case rule would choose weight 8 instead, illustrating that the shipped weight is a design trade-off, not an empirically proven optimum. Full comparisons are in [reports/v4_results.md](reports/v4_results.md); v1 and v3 reports also include large-catalog variants.

All large-catalog findings are synthetic and reflect the stated assumptions, not real user preferences. Since mood is derived from energy and valence, it is not independent human-labeled evidence. The static web demo remains on the small catalog and does not support the liked-song feature.

## Reliability & Testing
`tests/` includes parametrized reliability checks across six profiles and four values of `k`, as well as checks for:
- exactly `k` unique, score-sorted, explained results, including rare combinations and total mismatches;
- edge cases: `k=0`, empty catalog, `k` larger than the catalog, out-of-range energy;
- ranking quality: exact matches rank first, genre matches dominate when available;
- the energy score has no cliffs, and optional targets change ranking;
- the diversity penalty really swaps in a new artist;
- the class wrapper and functional pipeline agree;
- both CSVs load with valid value ranges;
- detailed score breakdowns agree with the final scores;
- baseline implementations and evaluation metrics behave correctly;
- the v1 experiment and report cover baseline comparisons and ablations;
- weight sampling is reproducible, ties prefer configurations near the defaults, and leave-one-listener-out validation excludes the held-out listener;
- liked songs are excluded from results, similarity scoring and explanations work, and empty or unknown liked-song IDs preserve existing recommendations.
- the large catalog is reproducible and clean, both synthetic listener protocols are valid, hidden ground-truth features do not leak to recommender scoring, and grouped validation resamples/holds out archetypes.

## Design Decisions and Trade-offs
- **Interpretability over accuracy:** simple weights and rule-based explanations, at the cost of no collaborative filtering or learned taste.
- **Progressive retrieval relaxation:** the system returns up to `k` results, broadening the candidate pool for rare tastes. If the catalog contains fewer than `k` songs, it returns all available songs; an empty catalog returns no results.
- **One scoring path:** removes drift between the tested class and the shipped function.
- **Logging:** the library uses `logging.getLogger(__name__)` and leaves handler setup to the application.

## Limitations and Responsible Use
- The 50-song catalog is tiny and uneven. The large catalog is broader, but its moods are derived from audio features and its evaluation is synthetic, not real listener evidence.
- Content-based scoring reinforces existing taste and can create filter bubbles. The diversity penalty only spreads artists, not genres.
- Explanations are rule-based and describe the scoring, not the user's intent.
- Intended for learning only, not for production streaming or high-stakes decisions.

## Collaboration with AI
AI helped structure the pipeline into retrieval, scoring, explanation and evaluation. One flawed suggestion was to create a new README instead of editing the existing one, which would have caused conflicts, a reminder to review AI-generated changes before applying them. A later AI code review found real bugs (an over-permissive retrieval filter, a diversity penalty applied after the top-k cut, duplicated scoring logic, and docs that didn't match the data), which are fixed in this version.
