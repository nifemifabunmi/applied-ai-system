# Real listener files

Put taste files from real people here (one `.json` per person). They are used by
`python -m experiments.run_v5_real` to test the recommender on real preferences.

**How a person contributes**
1. Open the Streamlit app (or the static page), choose the **Large** catalog.
2. Search for and add 8 to 200 songs they already like.
3. Optionally type a pseudonym (letters, digits, spaces, dashes or underscores; please not a real name or email). Email-like entries are replaced with `anonymous`.
4. Press **Download my taste file** and send the file to the project owner.

Nothing is uploaded automatically. A file contains only song ids, song titles and
artists, the pseudonym and a timestamp. Only add files from people who agreed to it,
and consider whether to commit them to a public repository.

**Selection bias:** people can only pick songs they recognise from the catalog, so
results say how well the system finds songs a person would *choose*, not everything they might enjoy.

Files marked `"example": true` (like `example_not_real.json`) are ignored.
