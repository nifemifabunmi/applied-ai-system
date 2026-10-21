# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 5.0**

---

## 2. Intended Use

This recommender suggests songs based on your mood, favorite genre, and energy level. It is designed for classroom learning and exploring how music recommenders work. It is not meant for real-world music streaming or commercial use.

---

## 3. How the Model Works

The model looks at your favorite genre, mood, and how much energy you want in a song. It scores each song higher if it matches your genre or mood, and if its energy is close to your target. It also considers if you like acoustic music, and, if you supply them, targets for valence, danceability and tempo. Retrieval first narrows the catalog and relaxes the filter until at least k songs are available; a diversity penalty then discourages repeated artists. The top scoring songs are recommended to you. The scoring rules are simple and use only a few features. Optionally, if you list songs you already like, the model also rewards songs that sound like them. The explanation text can optionally be written by an LLM (Claude) from the score's own facts and is shown only if a rule-based guardrail confirms it agrees with the score breakdown; otherwise the rule-based explanation is used.

---

## 4. Data

`data/songs.csv` has 50 songs by 48 artists across 27 genres (e.g. afrobeats, fuji, apala, pop, lofi, jazz). Each song has genre, mood, energy, tempo, valence, danceability and acousticness. Most genres have only one to four songs, so many tastes are thinly covered. `data/songs_new.csv` is a separate 20-song sample with extra optional columns (popularity, decade, detailed mood, instrumentalness, language) that the scorer supports but the main catalog lacks. `data/catalog_large.csv` (v4) has 1,985 songs across 40 genres, sampled from the Spotify tracks dataset on Kaggle. Its **mood labels are derived from energy and valence**, not assigned by people, so mood and energy are correlated by construction. It also holds hidden columns (`gt_*`) used only for evaluation.

---

## 5. Strengths

The system works well for users who like popular genres and common moods. It gives good results when your preferences match the songs in the dataset. The recommendations make sense for typical cases, like happy pop or chill lofi fans.

---

## 6. Limitations and Bias

One weakness discovered during testing is that the system can over-prioritize certain genres, such as pop or rock, if they are more common in the dataset. Users whose favorite genre or mood is not present in the data will rarely receive high-scoring recommendations, regardless of their other preferences. The current scoring logic also heavily rewards close matches on energy, which can create a filter bubble and limit variety for users with extreme or uncommon energy preferences. Valence, danceability and tempo only count when the user supplies them as targets, so profiles that give only genre, mood and energy ignore them. These factors can lead to biased recommendations and reduced diversity in the results.

---

## 7. Evaluation

For evaluation, I tested a variety of user profiles, including typical cases (e.g., pop/happy/high energy) and adversarial or edge cases (such as high energy with sad mood, or genres/moods not present in the dataset). I looked for whether the top recommendations matched the user's stated preferences and if the same songs appeared repeatedly. One surprise was how strongly the system favored songs that matched genre and energy, sometimes at the expense of mood or other features. For users with rare or conflicting preferences, the recommendations were often less relevant or repetitive, highlighting the system's sensitivity to dataset composition and scoring weights.

---

### Results (`python evaluation.py`, 7 profiles, k=5)
Mean genre match 0.40, mean mood match 0.43, mean artist diversity 1.00, catalog coverage 0.52. All guardrails passed (k results, no duplicates, explanations present). Genre match is lowest for rock, electronic and blues, which have one or two songs in the catalog, so the system fills the list with the nearest energy/mood matches. The current automated suite has 103 passing tests covering the recommender, Streamlit app, guardrail, listener-file handling, and experiments.

### Optional similarity component (v3)
If the user lists songs they already like, the model adds a bonus for songs whose sound (energy, valence, danceability, acousticness, tempo) is close to the average sound of those songs, searches the whole catalog instead of only the seed genre/mood, and never recommends the liked songs back. On the 7 hand-labelled listeners it did not improve accuracy (`reports/v3_results.md`). On the large catalog it helps a lot when taste is defined by sound and costs a little when taste is defined by genre (`reports/v4_results.md`), so the weight is a documented trade-off, not a proven optimum. It is similarity-based, not a trained model.

### Weight choices (v2)
The scoring weights were first set by hand. v2 checked them against the 7 hand-labelled listeners (`reports/v2_results.md`):

| Weight | Value | What it does | Evidence | Range that stays within 0.03 NDCG of the current score |
|---|---|---|---|---|
| genre | 1.25 | Points for an exact genre match | Removing it costs 0.11 NDCG (v1) | 0.5x to 4x (flat: not sensitive) |
| mood | 1.50 | Points for an exact mood match | Removing it costs 0.14 NDCG (v1), the largest | 0.75x to 1.5x (the most sensitive) |
| energy | 2.00 | Up to 2 points for energy closeness | Removing it costs 0.07; too much hurts (4x drops NDCG by 0.14) | 0.5x to 1.5x |
| acoustic | 0.60 | Reward or penalty from acousticness | Removing it costs only 0.02 | 0.75x to 2x |
| similarity (optional) | 3.0 | Closeness to the average sound of the user's liked songs, only when liked songs are given | Small catalog: no measurable gain (v3). Large catalog (v4/v5): +0.19 under audio taste for -0.05 under genre taste vs rules only | A dial, not a tuned optimum: weight 8 without the retrieval filter gives 0.69 / 0.70, weight 1 gives 0.91 / 0.33 (genre / audio taste) |

Searching 800 alternative weight sets found one that scores 0.62 against 0.59 on all seven listeners. When each listener was left out in turn (tuned on the other six, scored on the one left out), the tuned weights scored 0.56 against 0.59 for the current ones. So the improvement did not carry over to unseen listeners and **the weights were kept unchanged**. Jittering every weight by up to 30% at once gives 0.52 to 0.61, so the system is reasonably stable but not insensitive. These conclusions rest on 7 listeners and labels written by the project author, so they are indicative only.

### Evaluation protocols and what they do and do not show (v4, v5)
- **Hand-labelled listeners (small catalog, v1 to v3):** 7 listeners with labels written by the project author.
- **Synthetic listeners (large catalog, v4/v5):** 60 listeners from 5 archetypes. Archetypes are the five MUSIC-model preference factors (Rentfrow, Goldberg and Levitin, 2011), with catalog genres assigned using the STOMP-R scoring key from the Gosling lab at UT Austin; genres the key does not list are assigned to the nearest listed genre and flagged in `data/eval_archetypes.json`. Two opposite definitions of taste (by genre, and by sound in a richer feature space) bracket the answer: rules score 0.92 and 0.24, the default blend 0.87 and 0.43 (NDCG@5).
- **Real listeners (v5):** the pipeline exists (`src/real_listeners.py`, `experiments/run_v5_real.py`) but **no real listener files have been collected yet**, so there are no real-listener results.
- **Explanation guardrail (v5):** 98.7% of 450 true template explanations accepted; all 4,224 deliberately wrong ones rejected. The stricter limits match the prompt (two sentences and 45 words). The wrong examples were written to match the checks, so this shows the checks work, not that nothing can slip through. The real LLM was not run during development.

### Large-catalog evaluation caveats (v4)
There is no real listening data, so large-catalog results use two synthetic definitions of taste (by genre, and by sound in a richer feature space). They show behaviour under stated assumptions; they do not show how good the recommendations are for real people. Derived moods also make "mood match" partly redundant with energy. Archetypes come from a published preference model, but the model was built on survey responses from particular populations and a handful of broad genres; it is a reasonable grouping, not ground truth for any individual.

## 8. Future Work

The next step is to evaluate the recommender on consented real-listener taste files and measure how often optional AI-written explanations pass the guardrail in actual use. The guardrail is a heuristic; it should not be treated as proof that AI text cannot contain unsupported claims.

---

## 9. Personal Reflection

I learned that recommenders can be simple but still show patterns and biases. It was interesting to see how changing the weights or the data changed the results. This project made me think more about how music apps decide what to play next!

## 10. Intended Use and Non-Intended Use

This system is for learning and experimenting with music recommendation. It should not be used for real music streaming, commercial products, or making important decisions.
