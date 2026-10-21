# Changelog

Versions v1 to v4 each added three features so every change could be explained and tested on its own; v5 bundles more at the owner's request.

## v5.0.0: Explanations, real evidence, and delivery (current)
This version bundles more than three features at the owner's request; they are grouped by theme.
**Explanations**
1. **AI explanation writer with a guardrail** (`src/explain_llm.py`, `experiments/run_v5_guardrail.py`, `reports/v5_guardrail.md`): Claude writes from the score's facts only; a rule-based guardrail compares the text with the score breakdown; failures and API errors fall back to the rule-based text. Wired into the Streamlit app.
**Evidence**
2. **Research-based archetypes:** the five MUSIC-model preference factors, with catalog genres assigned from the published STOMP-R scoring key (assumptions recorded in `data/eval_archetypes.json`). Large-catalog reports regenerated; conclusions unchanged.
3. **Real-listener pipeline** (`src/real_listeners.py`, `experiments/run_v5_real.py`, `data/real_listeners/`): taste-file export in the app and the static page, validation, and a report. No real files collected yet.
**Delivery**
4. **Static page updated:** both catalogs, liked-songs picker, similarity, taste-file export, HTML escaping; the JS engine now matches Python on 72 cases.
5. **CI** (GitHub Actions, Python 3.10 to 3.12) and Dependabot; dependency files with bounded versions; frozen versions uploaded as a CI artifact.
6. **Architecture diagram** redrawn (`scripts/draw_architecture.py`).
Also: the "default blend" row added to the baseline tables, and the evaluation report generators accept any list of listeners.

## v4.0.0: What changes on a bigger catalog?
1. **Large catalog** (`scripts/build_catalog.py`, `data/catalog_large.csv`): 1,985 songs, 40 genres, 1,673 artists from the Spotify tracks dataset, with mood derived from energy and valence. The Streamlit app can switch between catalogs.
2. **Two opposite synthetic evaluation protocols** (`experiments/datasets.py`, `data/eval_archetypes.json`): taste defined by genre, and taste defined by sound in a richer feature space. 60 listeners; 10 hand-picked archetypes, replaced in v5 by 5 research-based ones, with intervals that resample whole archetypes. A first single-protocol attempt was discarded because "genre only" scored a perfect 1.00 by construction.
3. **Blend trade-off analysis** (`experiments/run_v4.py`, `reports/v4_results.md`): the genre/mood retrieval filter caps similarity at scale, so it is now skipped when liked songs are given; the default blend weight is 3 (a documented judgement call, not a proven optimum).
Also: group-aware cross-validation helpers and 11 new tests. Weight tuning (v2) was not re-run on the large catalog, because tuning against synthetic ground truth would only recover the assumptions built into it.

## v3.0.0: Does similarity to liked songs help?
1. **Similarity component** (`src/recommender.py`): optional bonus for songs that sound like the user's liked songs, in two modes (average sound or nearest liked song). Liked songs are never recommended back and explanations can name the song a pick sounds like.
2. **App support:** a "Songs you already like" picker in the Streamlit app and a new colour in the score breakdown. The static page was brought up to parity in v5.
3. **Experiment** (`experiments/run_v3.py`, `reports/v3_results.md`): two designs x four weights, plus similarity-only, validated leave-one-listener-out. **Negative result:** no measurable accuracy gain (0.593 vs 0.586; 0.549 vs 0.586 when chosen out of sample), so the weight stays small and the feature is documented as optional.

## v2.0.0: Are the weights any good?
1. **Weight tuning with honest validation** (`src/tuning.py`, `experiments/run_v2.py`): random search over genre, mood, energy and acoustic weights, scored with leave-one-listener-out. Result: tuning overfits (0.56 on unseen listeners vs 0.59 for the current weights), so the weights are unchanged.
2. **Sensitivity analysis** with a one-at-a-time sweep, a ±30% joint jitter and a chart (`reports/v2_sensitivity.png`).
3. **Documented weight choices** in the model card: value, purpose, evidence and safe range for each weight.

## v1.0.0: Does it beat simple baselines?
1. **Hand-labelled evaluation set** (`data/eval_listeners.json`): 7 listeners, each with 3 seed songs (used to derive a taste profile, then excluded) and held-out liked songs as ground truth. Metrics: precision, recall, hit rate and NDCG at 5, with bootstrap intervals (`src/metrics.py`).
2. **Baselines** (`src/baselines.py`): random, energy only, genre only, genre+mood rules, and nearest neighbours in audio-feature space.
3. **Ablation study** (`experiments/run_v1.py`): remove genre, mood, energy, acoustic preference, the diversity penalty and the retrieval filter one at a time. Results: `reports/v1_results.md`.

## v0 (cleanup of the original project)
Retrieval that always returns k results, a diversity penalty applied during selection, one shared scoring path, a smooth energy curve, a Streamlit app, tests and accurate docs.

## TODO
- **Collect real listeners.** The pipeline is built; it needs at least 5 people's taste files in `data/real_listeners/`, then `python -m experiments.run_v5_real`. This is the strongest next piece of evidence and the point where a trained model becomes worth trying.
- **Try the real LLM.** Run the app with `ANTHROPIC_API_KEY` set, check how often AI text falls back and why, and tighten or loosen the guardrail from that.
- **Exact dependency pins:** download a `frozen-requirements-py*.txt` artifact from a passing CI run and commit it as `requirements.lock`.
- Review the hand-written listener labels in `data/eval_listeners.json` and the genre-to-factor assignments flagged "assumption" in `data/eval_archetypes.json`, and replace them with your own judgement where you disagree.
