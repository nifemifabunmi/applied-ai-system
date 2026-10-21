# Changelog

Each version adds at most three features, so every change can be explained and tested on its own.

## v2.0.0: Are the weights any good? (current)
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
- **v3:** larger catalog and more listeners.
- **v4:** a learned or similarity-based component blended with the rules.
- **v5:** LLM explanation layer with a guardrail check, CI, refreshed architecture diagram, pinned dependencies.
