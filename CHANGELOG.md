# Changelog

Each version adds at most three features, so every change can be explained and tested on its own.

## v4.0.0: What changes on a bigger catalog? (current)
1. **Large catalog** (`scripts/build_catalog.py`, `data/catalog_large.csv`): 1,985 songs, 40 genres, 1,673 artists, with moods derived from energy and valence. The Streamlit app can switch between the small and large catalogs.
2. **Two synthetic evaluation protocols** (`experiments/datasets.py`, `data/eval_archetypes.json`): genre-defined taste and audio-defined taste across 60 listener draws in 10 archetypes. Confidence intervals resample whole archetypes.
3. **Blend trade-off analysis** (`experiments/run_v4.py`, `reports/v4_results.md`): skips genre/mood retrieval when liked songs are supplied and documents a similarity weight of 3 as a trade-off, not a proven optimum.
Also: group-aware cross-validation, large-catalog tests, and an updated architecture diagram covering the app, shared recommender, optional catalog builder, and offline experiments. The raw source dataset is not included; review its terms before rebuilding or publishing derived data.

## v3.0.0: Does similarity to liked songs help?
1. **Optional similarity component** (`src/recommender.py`): bonus for songs similar to the user's liked songs using centroid or nearest-song feature-space similarity. Liked songs are not recommended back; explanations may name a similar liked track.
2. **Streamlit support** (`app.py`): a liked-song picker and score-breakdown category. The static web demo is unchanged.
3. **Evaluation** (`experiments/run_v3.py`, `reports/v3_results.md`): compare two similarity modes and four weights with leave-one-listener-out validation. No held-out improvement was observed, so the feature remains optional and its weight is small.

## v2.0.0: Are the weights any good?
1. **Weight tuning with honest validation** (`src/tuning.py`, `experiments/run_v2.py`): random search over genre, mood, energy and acoustic weights, scored with leave-one-listener-out. Result: tuning overfits (0.56 on unseen listeners vs 0.59 for the current weights), so the weights are unchanged.
2. **Sensitivity analysis** with a one-at-a-time sweep, a ±30% joint jitter and a chart (`reports/v2_sensitivity.png`).
3. **Documented weight choices** in the model card: value, purpose, evidence and sensitivity range for each weight.

## v1.0.0: Does it beat simple baselines?
1. **Hand-labelled evaluation set** (`data/eval_listeners.json`): 7 listeners, each with 3 seed songs (used to derive a taste profile, then excluded) and held-out liked songs as ground truth. Metrics: precision, recall, hit rate and NDCG at 5, with bootstrap intervals (`src/metrics.py`).
2. **Baselines** (`src/baselines.py`): random, energy only, genre only, genre+mood rules, and nearest neighbours in audio-feature space.
3. **Ablation study** (`experiments/run_v1.py`): remove genre, mood, energy, acoustic preference, the diversity penalty and the retrieval filter one at a time. Results: `reports/v1_results.md`.

## v0 (cleanup of the original project)
Retrieval that always returns k results, a diversity penalty applied during selection, one shared scoring path, a smooth energy curve, a Streamlit app, tests and accurate docs.

## Planned
- **v5:** LLM explanation layer with a guardrail check, CI that runs the tests, and pinned dependencies.

## TODO
- Collect real listening preferences. Both large-catalog protocols are synthetic and cannot establish real-world recommendation quality.
- Review and replace the hand-written listener labels and archetype genre lists with the author's own judgement.
- Port liked-song similarity and the large catalog to the static page, or retire that page.
