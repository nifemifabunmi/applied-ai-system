# Changelog

Each version adds at most three features, so every change can be explained and tested on its own.

## v1.0.0: Does it beat simple baselines? (current)
1. **Hand-labelled evaluation set** (`data/eval_listeners.json`): 7 listeners, each with 3 seed songs (used to derive a taste profile, then excluded) and held-out liked songs as ground truth. Metrics: precision, recall, hit rate and NDCG at 5, with bootstrap intervals (`src/metrics.py`).
2. **Baselines** (`src/baselines.py`): random, energy only, genre only, genre+mood rules, and nearest neighbours in audio-feature space.
3. **Ablation study** (`experiments/run_v1.py`): remove genre, mood, energy, acoustic preference, the diversity penalty and the retrieval filter one at a time. Results: `reports/v1_results.md`.

## v0 (cleanup of the original project)
Retrieval that always returns k results, a diversity penalty applied during selection, one shared scoring path, a smooth energy curve, a Streamlit app, tests and accurate docs.

## Planned
- **v2:** weight tuning with leave-one-listener-out validation, sensitivity analysis, documented weight choices.
- **v3:** larger catalog and more listeners.
- **v4:** a learned or similarity-based component blended with the rules.
- **v5:** LLM explanation layer with a guardrail check, CI, refreshed architecture diagram, pinned dependencies.
