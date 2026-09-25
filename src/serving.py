"""Phase 7: online recommendations from an exported bundle (src/export_serving.py).

NumPy + SciPy + LightGBM only. This module must never import torch (directly or through
src.evaluate / src.ranker / src.baselines): PyTorch and LightGBM can't share a process on macOS.

One request, known user with history (the headline two-stage system):
  1. retrieval   user vector = normalise(e + MLP(e)), e = mean embedding of the last hist_len
                 positives; cosine against every movie; seen and untrained movies masked;
                 top-200 by (-score, index), the same rule as evaluate.top_k
  2. features    the ranker's 15 headline features for those 200, mirroring
                 ranker.build_features (checked by scripts/serving_parity.py)
  3. ranking     LightGBM scores; top-k by (-score, index)
Unknown user IDs (and known users with no positives) get the popularity fallback, minus anything
they've already rated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

K_CANDIDATES = 200
RECENT_PROFILE = 10


def top_k_desc(scores: np.ndarray, k: int) -> np.ndarray:
    """Top-k indices by (-score, index) over finite scores (evaluate.top_k's rule, one row)."""
    finite = np.flatnonzero(np.isfinite(scores))
    if len(finite) <= k:
        cand = finite
    else:
        kth = np.partition(scores[finite], len(finite) - k)[len(finite) - k]
        cand = finite[scores[finite] >= kth]
    return cand[np.lexsort((cand, -scores[cand]))][:k]


def rank_desc(x: np.ndarray) -> np.ndarray:
    """0-based rank by descending value, NaN last, stable (ranker._rank_desc, one row)."""
    key = np.where(np.isnan(x), -np.inf, x)
    order = np.argsort(-key, kind="stable")
    rank = np.empty(len(x), dtype=np.float32)
    rank[order] = np.arange(len(x))
    return rank


class Recommender:
    def __init__(self, bundle_dir, threads: int = 0):
        """threads: LightGBM prediction threads (0 = LightGBM's default). See src/serve.py."""
        import lightgbm as lgb
        self.threads = threads
        d = Path(bundle_dir)
        self.manifest = json.loads((d / "manifest.json").read_text())
        t = np.load(d / "two_tower.npz")
        self.E = t["E"].astype(np.float32)
        norm = np.maximum(np.linalg.norm(self.E, axis=1, keepdims=True), 1e-12)
        self.V = self.E / norm
        self.W1, self.b1, self.W2, self.b2 = t["W1"], t["b1"], t["W2"], t["b2"]
        self.hist_len = int(t["hist_len"])
        self.untrained = ~t["trained"]
        u = np.load(d / "users.npz")
        self.user_ids, self.offsets, self.seq = u["user_ids"], u["offsets"], u["items"]
        self.seen_indptr, self.seen_indices = u["seen_indptr"], u["seen_indices"]
        self.B = np.load(d / "ease_B.npy")
        self.keep = np.load(d / "ease_keep.npy")
        self.col = np.full(len(self.E), -1, dtype=np.int64)
        self.col[self.keep] = np.arange(len(self.keep))
        f = np.load(d / "features.npz")
        self.item_stats, self.user_stats = f["item_stats"], f["user_stats"]
        self.user_genres = f["user_genres"]
        self.genres = sp.csr_matrix((np.ones(len(f["genres_indices"]), dtype=np.float32),
                                     f["genres_indices"], f["genres_indptr"]),
                                    shape=(len(self.E), int(f["n_genres"])))
        self.genome_rows, self.genome_row = f["genome_rows"], f["genome_row"]
        self.has_genome = f["has_genome"]
        self.popular = np.load(d / "popular.npy")
        m = json.loads((d / "movies.json").read_text())
        self.movie_ids, self.titles = np.asarray(m["movie_ids"]), m["titles"]
        self.booster = lgb.Booster(model_file=str(d / "ranker.txt"))
        self.features = self.manifest["features"]
        if self.booster.feature_name() != self.features:
            raise ValueError("ranker features don't match the bundle manifest")
        self.k_candidates = int(self.manifest.get("k_candidates", K_CANDIDATES))

    # -- pieces ------------------------------------------------------------------------------
    def row(self, user_id):
        r = int(np.searchsorted(self.user_ids, user_id))
        return r if r < len(self.user_ids) and self.user_ids[r] == user_id else None

    def history(self, r):
        return self.seq[self.offsets[r]:self.offsets[r + 1]]

    def seen(self, r):
        return self.seen_indices[self.seen_indptr[r]:self.seen_indptr[r + 1]]

    def retrieve(self, r):
        hist = self.history(r)
        e = self.E[hist[-self.hist_len:]].mean(0)
        z = np.maximum(e @ self.W1.T + self.b1, 0.0) @ self.W2.T + self.b2
        u = e + z
        u = u / max(np.linalg.norm(u), 1e-12)
        s = (self.V @ u.astype(np.float32)).astype(np.float64)
        s[self.untrained] = -np.inf
        s[self.seen(r)] = -np.inf
        cand = top_k_desc(s, self.k_candidates)
        return cand, s[cand]

    def features_for(self, r, cand, tt_score):
        hist = self.history(r)
        F = {"tt_score": tt_score, "tt_rank": np.arange(len(cand), dtype=np.float64)}
        hc = self.col[hist]
        ease_full = self.B[hc[hc >= 0]].sum(0)
        c = self.col[cand]
        es = np.where(c >= 0, ease_full[np.maximum(c, 0)], np.nan)
        F["ease_score"], F["ease_rank"] = es, rank_desc(es.astype(np.float32))
        ist, ust = self.item_stats[cand], self.user_stats[r]
        F["item_log_n"], F["item_log_pos"], F["item_mean"] = ist[:, 0], ist[:, 1], ist[:, 2]
        F["item_days_since_last"] = (ust[3] - ist[:, 4]) / 86_400.0
        F["item_age_days"] = (ust[3] - ist[:, 3]) / 86_400.0
        F["user_log_n"], F["user_log_pos"], F["user_mean"] = (np.full(len(cand), ust[j])
                                                               for j in range(3))
        ov = self.genres[cand].toarray() * self.user_genres[r][None, :]
        F["genre_overlap"], F["genre_max"] = ov.sum(1), ov.max(1)
        for name, h in (("genome_sim_all", hist), ("genome_sim_recent", hist[-RECENT_PROFILE:])):
            h = h[self.has_genome[h]]
            if len(h) == 0:
                F[name] = np.full(len(cand), np.nan)
                continue
            p = self.genome_rows[self.genome_row[h]].sum(0)
            p = p / np.linalg.norm(p)
            has = self.has_genome[cand]
            sim = np.full(len(cand), np.nan)
            sim[has] = self.genome_rows[self.genome_row[cand[has]]] @ p
            F[name] = sim
        F["has_genome"] = self.has_genome[cand].astype(np.float64)
        return np.column_stack([np.asarray(F[n], dtype=np.float32) for n in self.features])

    # -- one request ---------------------------------------------------------------------------
    def recommend(self, user_id: int, k: int = 10) -> dict:
        t0 = time.perf_counter()
        r = self.row(user_id)
        if r is None or len(self.history(r)) == 0:
            seen = set() if r is None else set(self.seen(r).tolist())
            top = [i for i in self.popular[:k + len(seen)] if i not in seen][:k]
            return self._response(user_id, "popularity_fallback", top, [None] * len(top),
                                  {"total_ms": (time.perf_counter() - t0) * 1e3})
        cand, tt_score = self.retrieve(r)
        t1 = time.perf_counter()
        X = self.features_for(r, cand, tt_score)
        t2 = time.perf_counter()
        pred = self.booster.predict(X, num_threads=self.threads) if self.threads else \
            self.booster.predict(X)
        order = np.lexsort((cand, -pred))[:k]
        t3 = time.perf_counter()
        return self._response(user_id, "two_stage", cand[order].tolist(), pred[order].tolist(),
                              {"retrieval_ms": (t1 - t0) * 1e3, "features_ms": (t2 - t1) * 1e3,
                               "ranking_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3})

    # -- demo page helpers (read-only) ---------------------------------------------------------
    def profile(self, user_id: int, n: int = 10) -> dict:
        """The user's most recent liked movies (newest first) from the served training data."""
        r = self.row(user_id)
        if r is None:
            return {"user_id": int(user_id), "known": False, "n_liked": 0, "n_rated": 0, "recent": []}
        hist = self.history(r)
        recent = hist[-n:][::-1]
        return {"user_id": int(user_id), "known": True, "n_liked": int(len(hist)),
                "n_rated": int(len(self.seen(r))),
                "recent": [{"movieId": int(self.movie_ids[i]), "title": self.titles[i]} for i in recent]}

    def sample_user(self, rng=None) -> int:
        """A random known user with at least one liked movie."""
        rng = rng or np.random.default_rng()
        rows = np.flatnonzero(np.diff(self.offsets) > 0)
        return int(self.user_ids[rng.choice(rows)])

    def _response(self, user_id, strategy, idx, scores, timings):
        return {"user_id": int(user_id), "strategy": strategy,
                "items": [{"movieId": int(self.movie_ids[i]), "title": self.titles[i],
                           "score": None if s is None else float(s)}
                          for i, s in zip(idx, scores)],
                "timings_ms": {k: round(v, 3) for k, v in timings.items()}}
