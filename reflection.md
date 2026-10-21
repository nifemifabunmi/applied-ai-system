# Reflection: Comparing User Profiles and Recommendations

- Comparing the 'pop, happy, high energy' profile to the 'lofi, chill, low energy' profile, I noticed that the pop profile consistently recommends upbeat, energetic songs like 'Sunrise City' and 'Gym Hero.' This makes sense because these songs match both the genre and energy preferences. In contrast, the lofi profile shifts toward slower, more relaxed tracks with higher acousticness, such as 'Midnight Coding' and 'Library Rain.'

- When testing a 'rock, intense, high energy' profile versus a 'blues, sad, high energy' profile, the rock profile surfaces 'Storm Runner' first (genre, mood and energy all match) and then falls back to nearby high-energy songs, while the blues profile gets one true match ('Bluesy Nights') followed by loosely related high-energy tracks. This shows that the system is more effective when the user's preferences align with the dataset.

- I also observed that 'Gym Hero' shows up for several pop and intense-mood profiles, because its genre and energy sit close to those preferences and the scoring rewards them. As a result, the same songs can dominate recommendations for similar profiles, especially when the dataset is small or lacks diversity.

- Overall, the differences in recommendations between profiles make sense given the scoring logic, but also highlight the importance of dataset variety and balanced weighting of features.

- After the fixes, every profile (even rare or nonsense ones) returns exactly 5 unique songs, and artist diversity is 1.00 across all test users. That is a real improvement over the earlier OR-only filter, which returned only 2 songs for rock/intense.
