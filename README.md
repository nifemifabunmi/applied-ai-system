# Music Recommender: Retrieval + Ranking with Explanations

## Original Project
**Music Recommender Simulation** was a simple content-based recommender that suggests songs from user preferences for genre, mood and song features. The goal was to see how recommenders turn data into predictions, where bias creeps in, and how scoring choices change results.

## Summary
This version turns that starter into a small, testable recommender with retrieval, scoring, diversity-aware ranking, and explanations. Song selection remains rule- and similarity-based so every score can be traced. Optionally, an AI service can write an explanation from the score facts; a local rule-based guardrail checks that text, and the app falls back to its rule-based explanation if the check fails or the API errors.

## Architecture
![System Architecture](diagrams/architecture.png)

1. **Catalog selection and loading**: the Streamlit app selects the small or large CSV catalog; the CLI uses the small catalog. Both call the shared recommender.
2. **Retrieval** (`retrieve_candidates`): normally narrows candidates by genre/mood and progressively relaxes the filter when needed. With liked songs, the default is to consider the full selected catalog.
3. **Scoring and ranking** (`score_song`, `rank_with_diversity`): a shared score combines rule-based preferences with optional sound similarity to liked songs, then selects a diverse list with an artist-repeat penalty.
4. **Rule-based explanation** (`explain_song`): provides reasons and similar-song suggestions.
5. **Optional AI writer and guardrail** (`src/explain_llm.py`): if enabled and configured, generates text from one recommendation's score facts and accepts it only when the local checks pass. Otherwise, the rule-based explanation is shown.

Offline v1-v5 experiments produce evaluation reports; they are separate from the app's recommendation runtime. Large-catalog results are synthetic until real listener files are collected. The optional catalog builder turns a compatible external CSV into the checked-in large catalog.

`Recommender` (the OO class) is a thin wrapper over the same functions, so class tests and the CLI exercise identical logic.

**Optional (v3-v5):** provide songs you already like and recommendations get a similarity bonus based on energy, valence, danceability, acousticness, and tempo. The app searches the full catalog rather than applying the genre/mood retrieval filter, excludes liked songs from results, and can explain which liked track sounds similar. The default similarity weight is a trade-off, not a proven optimum.

## Getting Started
Requires Python and pip. On Windows PowerShell:
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt  # needed for tests, experiments, and rebuilding the large catalog
```

To enable optional AI-written explanations in Windows PowerShell, set your Anthropic key in the current terminal session (do not commit it):
```powershell
$env:ANTHROPIC_API_KEY = "your-key"
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
python -m experiments.run_v5_guardrail  # test explanation checks without calling an AI service
python -m experiments.run_v5_real  # evaluate taste files (none collected in this repo yet)
python -m scripts.draw_architecture  # regenerate diagrams/architecture.png and .svg
```

## Streamlit App (`app.py`)
The main frontend imports `src.recommender` directly (via `recommend_detailed`). Pick a catalog, genre, mood, energy, and optionally songs you already like. Recommendations show the score breakdown, retrieval funnel, repeat-artist penalty, and explanation. AI writing is off by default. When enabled, selected recommendation facts (including song/catalog text and taste preferences) are sent to Anthropic; the local guardrail either accepts the returned text or uses the rule-based fallback. Keep the API key out of source control. To deploy on Streamlit Community Cloud, add the key as an app secret only if you want the optional AI feature.

The **Share your taste** panel exports a JSON file only after the user chooses to download it. Nothing is uploaded automatically. Ask participants for consent, use a pseudonym rather than a real name or email, and review the file before committing it.

## Static Web Demo (optional)
A no-install, single-file copy lives in `docs/index.html`: select either catalog, choose a genre/mood/energy and optionally liked songs, then inspect the score breakdown and explanation. It can download a taste file locally; it never calls the AI API, so its explanations remain rule-based.
```bash
python -m scripts.build_web            # rebuild docs/index.html from web/ + both catalogs
python -m scripts.check_web_parity     # compare the JS engine with Python
python -m http.server -d docs 8000     # then open http://localhost:8000
```
The parity check requires Node.js in addition to the Python dependencies.
To host it free, go to GitHub, then Settings, then Pages, choose branch `main` and folder `/docs`.
Unlike the Streamlit app, this page uses `web/engine.js`, a JavaScript port of `src/recommender.py`, so the page runs entirely in the browser with no server. If you change weights or ranking in Python, mirror the change in `web/engine.js` and re-run the parity check.
The page uses a JavaScript port in `web/engine.js`; CI checks the ranking parity and that `docs/index.html` is up to date.

## Data
- `data/songs.csv`: 50 songs, 48 artists, 27 genres, with 10 columns (id, title, artist, genre, mood, energy, tempo_bpm, valence, danceability, acousticness). This is the small demo catalog.
- `data/catalog_large.csv`: 1,985 tracks across 40 genres for the large-catalog app and synthetic experiments. Its mood labels are derived from energy and valence; hidden `gt_*` audio features are used only as evaluation ground truth and are not loaded by the recommender.
- `data/songs_new.csv`: 20-song sample that also has `popularity`, `release_decade`, `detailed_mood`, `instrumentalness` and `language`. These optional columns are supported by the scorer but are not in the main catalog.
- `data/eval_listeners.json`: hand-labelled listener seed songs and held-out liked songs for offline evaluation.
- `data/eval_archetypes.json`: synthetic archetypes based on five MUSIC-model preference factors. Genre-to-factor assignments include documented assumptions and source references.
- `data/real_listeners/`: opt-in taste files for offline evaluation; the shipped example is explicitly marked as non-real and ignored.

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

## Larger Catalog and Synthetic Evaluation (v4/v5)
The 1,985-song catalog has no listening history, so v4/v5 use 60 synthetic listeners (five research-based archetypes with 12 repeatable seed-song draws each) under two deliberately different assumptions:

- **Genre taste:** a listener likes tracks in genres mapped to their archetype; this favors genre-based rules. The five archetypes represent the MUSIC-model factors; genre mappings use the published STOMP-R scoring key where available, and assumptions for unmapped genres are documented in `data/eval_archetypes.json`.
- **Audio taste:** a listener likes the 100 tracks closest to their seed songs in a richer feature space, including speechiness, instrumentalness, liveness, and loudness. These hidden evaluation-only features are never supplied to the recommender; this favors similarity.

| Method (NDCG@5) | Genre taste | Audio taste |
|---|---:|---:|
| Random | 0.20 | 0.05 |
| Energy only | 0.22 | 0.16 |
| Genre only | 1.00 | 0.22 |
| Audio-feature nearest neighbours | 0.30 | 0.76 |
| Rules only | 0.92 | 0.24 |
| Rules + similarity to liked songs (default blend) | 0.87 | 0.43 |

No method wins under both assumptions. The default blend gives up about 0.05 NDCG under genre taste for a 0.19 gain under audio taste, compared with rules alone. These are synthetic benchmarks, not real listener preference evidence. Full comparisons are in [reports/v4_results.md](reports/v4_results.md); v1 and v3 reports also include large-catalog variants.

Mood labels in the large catalog are derived from energy and valence, so they are not independent human-labeled evidence.

## AI-Written Explanations with a Guardrail (v5)
AI-written explanations are optional and off by default. If enabled and configured with `ANTHROPIC_API_KEY`, the app sends a small JSON containing one recommended song's details, the user's selected preferences, score factors, and (when applicable) the liked song it resembles to Anthropic. Song/catalog strings are cleaned first and are treated as untrusted data, not instructions. The model does not choose songs or change scores.

A local heuristic guardrail checks the generated text for length/format, unsupported titles, artists, genres, moods and numbers, unsupported score claims, and whether it mentions a factor that earned points. If any check fails or the API request errors, the app displays the rule-based explanation instead. This is a defense-in-depth heuristic, not a guarantee that every false claim will be detected. Set `VIBEFINDER_LLM_MODEL` to use a different model supported by your Anthropic account. The guardrail experiment uses template-written examples, so its measured results do not establish real-model accuracy: see [reports/v5_guardrail.md](reports/v5_guardrail.md).

## Real-Listener Evaluation (v5)
Participants can choose 8 to 200 tracks from the large catalog and download a taste file. The file is not uploaded automatically; participants send it to the project owner, who can place consented files in `data/real_listeners/`. The evaluation validates catalog IDs and song names, holds out liked tracks, and reports skipped files. Run `python -m experiments.run_v5_real` to produce [reports/v5_real_listeners.md](reports/v5_real_listeners.md). No real listener files are included currently, so the checked-in report contains no real-listener findings. The static page also supports local taste-file download and never calls the AI API.

## Continuous Integration (v5)
GitHub Actions runs the test suite, evaluation guardrails, Python/JavaScript parity check, static-page rebuild check, and experiment smoke tests on Python 3.10-3.12. Dependabot checks Python and GitHub Actions dependencies. Dependencies use bounded version ranges; CI uploads the exact resolved `pip freeze` per Python version as an artifact.

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
- the large catalog is reproducible and clean, both synthetic listener protocols are valid, hidden ground-truth features do not leak to recommender scoring, and grouped validation resamples/holds out archetypes;
- AI explanations use only one recommendation's sanitized facts, are checked against the score, and safely fall back on rejection or API errors;
- taste-file export and validation handle consent-based real-listener submissions without uploading them automatically.

## Design Decisions and Trade-offs
- **Interpretability over accuracy:** simple weights and rule-based explanations, at the cost of no collaborative filtering or learned taste.
- **Progressive retrieval relaxation:** the system returns up to `k` results, broadening the candidate pool for rare tastes. If the catalog contains fewer than `k` songs, it returns all available songs; an empty catalog returns no results.
- **One scoring path:** removes drift between the tested class and the shipped function.
- **Logging:** the library uses `logging.getLogger(__name__)` and leaves handler setup to the application.

## Limitations and Responsible Use
- The 50-song catalog is tiny and uneven. The large catalog is broader, but its moods are derived from audio features and its evaluation is synthetic, not real listener evidence.
- Content-based scoring reinforces existing taste and can create filter bubbles. The diversity penalty only spreads artists, not genres.
- Rule-based explanations describe the scoring, not the user's intent. Optional AI text uses a third-party API and a heuristic guardrail that can miss subtle false claims.
- Real-listener files are not included; collect them only with consent, avoid personally identifying pseudonyms, and consider privacy before storing or publishing them.
- Intended for learning only, not for production streaming or high-stakes decisions.

## Collaboration with AI
AI helped structure the pipeline into retrieval, scoring, explanation and evaluation. One flawed suggestion was to create a new README instead of editing the existing one, which would have caused conflicts, a reminder to review AI-generated changes before applying them. A later AI code review found real bugs (an over-permissive retrieval filter, a diversity penalty applied after the top-k cut, duplicated scoring logic, and docs that didn't match the data), which are fixed in this version.
