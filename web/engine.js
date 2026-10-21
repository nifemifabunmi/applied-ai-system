/*
 * JavaScript port of src/recommender.py for the static web demo.
 * Keep WEIGHTS and the pipeline in sync with the Python version;
 * scripts/check_web_parity.py compares the two on fixed profiles for both catalogs.
 * Not ported: optional song metadata (popularity, decade...) because the catalogs lack it,
 * and the "nearest" similarity mode (the Python default, "centroid", is what the page uses).
 */
(function (root) {
  'use strict';

  const WEIGHTS = {
    genre: 1.25, mood: 1.5, energy: 2.0, acoustic: 0.6, acoustic_target: 0.8,
    valence: 0.5, danceability: 0.5, tempo: 0.5, similarity: 3.0,
  };
  const ENERGY_TOLERANCE = 0.5;
  const TEMPO_TOLERANCE_BPM = 60;
  const ARTIST_REPEAT_PENALTY = 0.7;
  const SIMILARITY_RADIUS = 1.0;
  const PART_KEYS = ['genre', 'mood', 'energy', 'acoustic', 'valence', 'danceability', 'tempo', 'similarity'];

  const closeness = (a, b, tol) => Math.max(0, 1 - Math.abs(a - b) / tol);
  const has = (o, k) => o[k] !== undefined && o[k] !== null;

  // ---- similarity to liked songs ------------------------------------------
  const soundVector = (s) => [s.energy, s.valence, s.danceability, s.acousticness, s.tempo_bpm / 200];
  const soundDistance = (a, b) => Math.sqrt(a.reduce((sum, x, i) => sum + (x - b[i]) ** 2, 0));
  function tasteVector(liked) {
    const vs = liked.map(soundVector);
    return vs[0].map((_, i) => vs.reduce((sum, v) => sum + v[i], 0) / vs.length);
  }
  const similarityToTaste = (song, taste) => Math.max(0, 1 - soundDistance(soundVector(song), taste) / SIMILARITY_RADIUS);
  function nearestLiked(song, liked) {
    const v = soundVector(song);
    let best = null, bestD = Infinity;
    for (const s of liked) {
      const d = soundDistance(soundVector(s), v);
      if (d < bestD) { best = s; bestD = d; }
    }
    return best;
  }

  // ---- stage 1: retrieval -------------------------------------------------
  function retrieveCandidates(prefs, songs, k) {
    const { genre, mood } = prefs;
    if (!genre && !mood) return { pool: songs.slice(), tier: 'all' };
    const g = (s) => !genre || s.genre === genre;
    const m = (s) => !mood || s.mood === mood;
    const tiers = [
      ['both', songs.filter((s) => g(s) && m(s))],
      ['either', songs.filter((s) => g(s) || m(s))],
      ['fallback', songs.slice()],
    ];
    for (const [tier, pool] of tiers) if (pool.length >= k) return { pool, tier };
    return { pool: songs.slice(), tier: 'fallback' };
  }

  // ---- stage 2: scoring ---------------------------------------------------
  function scoreBreakdown(song, prefs) {
    const w = WEIGHTS;
    const p = {};
    PART_KEYS.forEach((key) => { p[key] = 0; });
    if (prefs.genre && song.genre === prefs.genre) p.genre = w.genre;
    if (prefs.mood && song.mood === prefs.mood) p.mood = w.mood;
    p.energy = w.energy * closeness(song.energy, has(prefs, 'energy') ? prefs.energy : 0.5, ENERGY_TOLERANCE);
    if (has(prefs, 'acousticness')) p.acoustic = w.acoustic_target * closeness(song.acousticness, prefs.acousticness, 1);
    else if (prefs.likes_acoustic) p.acoustic = song.acousticness * w.acoustic;
    else p.acoustic = -song.acousticness * w.acoustic;
    if (has(prefs, 'valence')) p.valence = w.valence * closeness(song.valence, prefs.valence, 1);
    if (has(prefs, 'danceability')) p.danceability = w.danceability * closeness(song.danceability, prefs.danceability, 1);
    if (has(prefs, 'tempo_bpm')) p.tempo = w.tempo * closeness(song.tempo_bpm, prefs.tempo_bpm, TEMPO_TOLERANCE_BPM);
    if (prefs.taste_vector) p.similarity = w.similarity * similarityToTaste(song, prefs.taste_vector);
    return p;
  }

  const scoreSong = (song, prefs) => PART_KEYS.reduce((sum, k) => sum + scoreBreakdown(song, prefs)[k], 0);

  // ---- stage 3: diversity-aware ranking -------------------------------------
  function rankWithDiversity(scored, k) {
    const remaining = scored.slice();
    const picked = [];
    const counts = {};
    while (remaining.length && picked.length < k) {
      let bestI = 0, bestAdj = -Infinity;
      remaining.forEach((item, i) => {
        const adj = item.score - ARTIST_REPEAT_PENALTY * (counts[item.song.artist] || 0);
        if (adj > bestAdj) { bestI = i; bestAdj = adj; }
      });
      const [item] = remaining.splice(bestI, 1);
      counts[item.song.artist] = (counts[item.song.artist] || 0) + 1;
      picked.push({ ...item, adjusted: bestAdj, penalty: item.score - bestAdj });
    }
    return picked;
  }

  // ---- stage 4: explanation -------------------------------------------------
  function similarSongs(target, all, n = 2) {
    const overlap = (s) => (s.genre === target.genre) + (s.mood === target.mood) + (s.artist === target.artist);
    return all.filter((s) => s.id !== target.id && overlap(s) > 0)
      .sort((a, b) => overlap(b) - overlap(a)).slice(0, n);
  }

  function explainSong(song, prefs, all) {
    const reasons = [];
    if (prefs.genre && song.genre === prefs.genre) reasons.push(`matches your favorite genre ‘${prefs.genre}’`);
    if (prefs.mood && song.mood === prefs.mood) reasons.push(`matches your favorite mood ‘${prefs.mood}’`);
    const diff = Math.abs(song.energy - (has(prefs, 'energy') ? prefs.energy : 0.5));
    if (diff < 0.1) reasons.push('has energy very close to your target');
    else if (diff < 0.3) reasons.push('has energy somewhat close to your target');
    if (!has(prefs, 'acousticness')) {
      if (prefs.likes_acoustic && song.acousticness > 0.5) reasons.push('is acoustic, which you prefer');
      else if (!prefs.likes_acoustic && song.acousticness < 0.3) reasons.push('is not very acoustic, matching your preference');
    }
    if (prefs.taste_vector && similarityToTaste(song, prefs.taste_vector) >= 0.6) {
      const close = nearestLiked(song, prefs.liked_songs || []);
      reasons.push(close ? `sounds like ‘${close.title}’, which you like` : 'sounds like the songs you like');
    }
    let text = reasons.length
      ? `This song ${reasons.join(', ')}.`
      : 'This song may not match your preferences closely, but give it a listen.';
    const sim = similarSongs(song, all);
    if (sim.length) text += ` If you like it, try ${sim.map((s) => `‘${s.title}’ by ${s.artist}`).join(' or ')}.`;
    return text;
  }

  // ---- full pipeline --------------------------------------------------------
  function recommend(userPrefs, songs, k = 5) {
    if (k <= 0 || !songs.length) return { tier: 'all', poolSize: 0, results: [] };
    const prefs = { ...userPrefs };
    if (has(prefs, 'energy')) prefs.energy = Math.min(1, Math.max(0, prefs.energy));

    const likedIds = new Set(prefs.liked_ids || []);
    delete prefs.liked_ids;
    let pool = songs;
    if (likedIds.size) {
      const liked = songs.filter((s) => likedIds.has(s.id));
      if (liked.length) {
        prefs.taste_vector = tasteVector(liked);
        prefs.liked_songs = liked;
        pool = songs.filter((s) => !likedIds.has(s.id));
      }
    }
    let candidates, tier;
    if (prefs.taste_vector) { candidates = pool.slice(); tier = 'liked'; }
    else ({ pool: candidates, tier } = retrieveCandidates(prefs, pool, k));

    const scored = candidates.map((song) => {
      const breakdown = scoreBreakdown(song, prefs);
      return { song, breakdown, score: PART_KEYS.reduce((s, key) => s + breakdown[key], 0) };
    });
    const results = rankWithDiversity(scored, k).map((r) => ({ ...r, explanation: explainSong(r.song, prefs, songs) }));
    return { tier, poolSize: candidates.length, results };
  }

  const api = { WEIGHTS, retrieveCandidates, scoreBreakdown, scoreSong, rankWithDiversity, explainSong, recommend };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.VibeEngine = api;
})(typeof window !== 'undefined' ? window : globalThis);
