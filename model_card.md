# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0**

---

## 2. Intended Use

This recommender suggests songs based on your mood, favorite genre, and energy level. It is designed for classroom learning and exploring how music recommenders work. It is not meant for real-world music streaming or commercial use.

---

## 3. How the Model Works

The model looks at your favorite genre, mood, and how much energy you want in a song. It scores each song higher if it matches your genre or mood, and if its energy is close to your target. It also considers if you like acoustic music, and, if you supply them, targets for valence, danceability and tempo. Retrieval first narrows the catalog and relaxes the filter until at least k songs are available; a diversity penalty then discourages repeated artists. The top scoring songs are recommended to you. The scoring rules are simple and use only a few features.

---

## 4. Data

`data/songs.csv` has 50 songs by 48 artists across 27 genres (e.g. afrobeats, fuji, apala, pop, lofi, jazz). Each song has genre, mood, energy, tempo, valence, danceability and acousticness. Most genres have only one to four songs, so many tastes are thinly covered. `data/songs_new.csv` is a separate 20-song sample with extra optional columns (popularity, decade, detailed mood, instrumentalness, language) that the scorer supports but the main catalog lacks.

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
Mean genre match 0.40, mean mood match 0.43, mean artist diversity 1.00, catalog coverage 0.52. All guardrails passed (k results, no duplicates, explanations present). Genre match is lowest for rock, electronic and blues, which have one or two songs in the catalog, so the system fills the list with the nearest energy/mood matches. The 56 automated tests cover k-guarantees, edge cases, ranking quality, smooth energy scoring, diversity, baseline metrics, and weight-tuning helpers.

### Optional similarity component (v3)
Users can optionally provide songs they already like. The recommender adds a small score bonus to songs that are close in energy, valence, danceability, acousticness, and tempo, and excludes the liked songs themselves from the candidate pool. Similarity can use the average feature vector (centroid) or the closest individual liked song (nearest; the default). Explanations may name a liked song with similar sound. The Streamlit app exposes the picker; the static browser demo does not.

On seven hand-labelled listeners, nearest similarity at weight 0.5 reached mean NDCG@5 0.593 versus 0.586 for rules alone, but leave-one-listener-out selection scored 0.549 versus 0.586 for rules alone (`reports/v3_results.md`). This small evaluation provides no evidence of a reliable accuracy gain, so the feature is optional and its default weight is kept small. It is similarity-based and not a trained model.

### Weight choices (v2)
The scoring weights were initially selected by hand. The v2 analysis (`python -m experiments.run_v2`) evaluated them against seven hand-labelled listeners using random search, leave-one-listener-out validation, and sensitivity tests. The tuned settings improved in-sample NDCG@5 from 0.59 to 0.62, but scored 0.56 versus 0.59 for the current weights on held-out listeners. The weights were therefore left unchanged; the in-sample result appears to overfit this small evaluation set.

| Weight | Current value | Role | Evidence and sensitivity |
|---|---:|---|---|
| Genre | 1.25 | Points for an exact genre match | Removing genre reduced v1 NDCG@5 by 0.11. Results stayed within 0.03 of baseline from 0.5x to 4x. |
| Mood | 1.50 | Points for an exact mood match | Removing mood reduced v1 NDCG@5 by 0.14, the largest ablation change. This was the most sensitive weight; the stable range was 0.75x to 1.5x. |
| Energy | 2.00 | Maximum points for a close energy match | Removing energy reduced v1 NDCG@5 by 0.07. Excessively high weight hurt; 4x reduced NDCG@5 by 0.14. |
| Acoustic | 0.60 | Reward or penalty based on acousticness preference | Removing it reduced v1 NDCG@5 by 0.02. The stable range was 0.75x to 2x. |
| Similarity (optional) | 0.50 | Closeness to the user's liked songs | No reliable held-out improvement in v3; kept small and only active when liked songs are supplied. |

Perturbing all four weights by up to 30% produced mean NDCG@5 values from 0.52 to 0.61 across 300 trials. See `reports/v2_results.md` and `reports/v2_sensitivity.png` for the full results. These measurements are exploratory only: seven author-labelled listeners are not enough to support general claims.

## 8. Future Work

I would add more features, like danceability and tempo, to make recommendations smarter. I would also try to make the results more diverse, so you see new songs sometimes. Finally, I would improve the explanations so users know why a song was picked.

---

## 9. Personal Reflection

I learned that recommenders can be simple but still show patterns and biases. It was interesting to see how changing the weights or the data changed the results. This project made me think more about how music apps decide what to play next!

## 10. Intended Use and Non-Intended Use

This system is for learning and experimenting with music recommendation. It should not be used for real music streaming, commercial products, or making important decisions.
