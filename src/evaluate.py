"""Phase 2: the evaluation harness every model is scored with.

One fixed protocol, so model comparisons are like-for-like:

- Catalog: all 62,423 movies in movies.csv, indexed by sorted movieId (`ItemIndex`). Every model
  scores the whole catalog. A movie a model can't score (never seen in its training data, or cut
  by EASE's item filter) gets -inf. A relevant movie that's cold for a model is simply a miss.
- Population (`EvalData`): users with >= 1 positive (rating >= 4) in train and >= 1 positive in the
  target slice (val or test), exactly as counted in results/data_stats.csv. Every model is scored
  on exactly these users, and the exclusions are reported.
- Masking: every item the user rated (any rating) in the model's training data (train when
  scoring val, train + val when scoring test) is set to -inf before ranking.
- Top-k: ordered by (-score, item index), so ties resolve the same way on every run. Masked or
  unscoreable (-inf) items are never recommended; a list may come back shorter than k.
- Metrics, averaged per user with the MEAN (medians are for across seeds):
    Recall@10 (capped)  hits / min(|relevant|, 10)
    NDCG@10             binary gains, ideal DCG over min(|relevant|, 10) positions
    AUC                 P(score of a held-out positive > score of a sampled negative), ties = 0.5.
                        Negatives: 100 per user, sampled once with a fixed seed from movies with >= 1
                        rating in the training data that the user never rated in ANY slice. Shared
                        by all models.
    Coverage            distinct recommended movies / movies with >= 1 rating in the training data
                        (train: 51,195 when scoring val; train + val when scoring test).
- Two-stage models (the Phase 6 ranker) use the same interface: candidates get ranker scores,
  everything else -inf, so non-candidates rank below every candidate.

Models implement `score(user_ids) -> (len(user_ids), n_items) float array`.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import scipy.sparse as sp
import torch

from src.data_prep import POSITIVE_THRESHOLD

K = 10
N_NEGATIVES = 100
NEGATIVE_SEED = 20260922
SLICES = ("val", "test")


# ---------------------------------------------------------------------------------------------
# Item index
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class ItemIndex:
    """movieId <-> column index, over the full catalog, sorted by movieId."""
    movie_ids: np.ndarray  # int, sorted, unique

    @classmethod
    def from_movies_csv(cls, path) -> "ItemIndex":
        ids = pd.read_csv(path, usecols=["movieId"])["movieId"].to_numpy()
        return cls.from_ids(ids)

    @classmethod
    def from_ids(cls, ids) -> "ItemIndex":
        ids = np.unique(np.asarray(ids, dtype=np.int64))
        return cls(ids)

    def __len__(self):
        return len(self.movie_ids)

    def to_index(self, movie_ids) -> np.ndarray:
        movie_ids = np.asarray(movie_ids)
        idx = np.searchsorted(self.movie_ids, movie_ids)
        idx = np.minimum(idx, len(self.movie_ids) - 1)
        if not np.array_equal(self.movie_ids[idx], movie_ids):
            raise KeyError("movieId not in catalog")
        return idx


# ---------------------------------------------------------------------------------------------
# Evaluation data: population, masks, relevant sets, negatives
# ---------------------------------------------------------------------------------------------

@dataclass
class EvalData:
    scheme: str
    slice: str
    items: ItemIndex
    users: np.ndarray            # userIds evaluated, sorted
    seen: sp.csr_matrix          # users x items, 1 = rated in the model's training data
    relevant: sp.csr_matrix      # users x items, 1 = positive in the target slice
    negatives: np.ndarray        # users x N_NEGATIVES item indices, for AUC
    train_items: np.ndarray      # bool per item: >= 1 rating in the model's training data
    counts: dict = field(default_factory=dict)

    @property
    def n_users(self):
        return len(self.users)

    def subsample(self, n: int, seed: int) -> "EvalData":
        """Fixed random subset of users (for fast tuning). Report numbers on the full set."""
        if n >= self.n_users:
            return self
        rows = np.sort(np.random.default_rng(seed).choice(self.n_users, n, replace=False))
        return EvalData(self.scheme, self.slice, self.items, self.users[rows],
                        self.seen[rows], self.relevant[rows], self.negatives[rows],
                        self.train_items, {**self.counts, "subsample": n, "subsample_seed": seed})


def load_ratings(splits_path, scheme: str) -> pd.DataFrame:
    col = f"split_{scheme}"
    t = pq.read_table(splits_path, columns=["userId", "movieId", "rating", col],
                      read_dictionary=[col])
    df = t.to_pandas()
    return df.rename(columns={col: "split"})


def build_eval_data(ratings: pd.DataFrame, items: ItemIndex, scheme: str, slice_: str,
                    n_negatives: int = N_NEGATIVES, seed: int = NEGATIVE_SEED) -> EvalData:
    """ratings: userId, movieId, rating, split (train/val/test) for one scheme."""
    if slice_ not in SLICES:
        raise ValueError(slice_)
    training = ("train",) if slice_ == "val" else ("train", "val")
    in_training = ratings["split"].isin(training).to_numpy()
    in_target = (ratings["split"] == slice_).to_numpy()
    pos = (ratings["rating"] >= POSITIVE_THRESHOLD).to_numpy()
    in_train = (ratings["split"] == "train").to_numpy()

    uid = ratings["userId"].to_numpy()
    has_train_pos = np.unique(uid[in_train & pos])
    has_target_pos = np.unique(uid[in_target & pos])
    all_users = np.unique(uid)
    users = np.intersect1d(has_train_pos, has_target_pos)  # sorted

    row_of = pd.Series(np.arange(len(users)), index=users)
    keep = np.isin(uid, users)
    rows = row_of.reindex(uid[keep]).to_numpy()
    cols = items.to_index(ratings["movieId"].to_numpy()[keep])

    def csr(mask):
        m = mask[keep]
        data = np.ones(int(m.sum()), dtype=np.int8)
        return sp.csr_matrix((data, (rows[m], cols[m])), shape=(len(users), len(items)))

    seen = csr(in_training)
    relevant = csr(in_target & pos)
    ever_rated = csr(np.ones(len(ratings), dtype=bool))

    train_items = np.zeros(len(items), dtype=bool)
    train_items[items.to_index(ratings["movieId"].to_numpy()[in_training])] = True

    negatives = sample_negatives(ever_rated, train_items, n_negatives, seed)
    counts = {
        "users_total": len(all_users),
        "users_no_train_positive": len(all_users) - len(has_train_pos),
        "users_no_target_positive": len(has_train_pos) - len(users),
        "users_evaluated": len(users),
        "coverage_denominator": int(train_items.sum()),
    }
    return EvalData(scheme, slice_, items, users, seen, relevant, negatives, train_items, counts)


def sample_negatives(ever_rated: sp.csr_matrix, pool_mask: np.ndarray, n: int, seed: int):
    """n distinct items per user from pool_mask, excluding anything the user ever rated."""
    rng = np.random.default_rng(seed)
    pool = np.flatnonzero(pool_mask)
    out = np.empty((ever_rated.shape[0], n), dtype=np.int32)
    for i in range(ever_rated.shape[0]):
        rated = ever_rated.indices[ever_rated.indptr[i]:ever_rated.indptr[i + 1]]
        if len(pool) - np.isin(pool, rated).sum() < n:
            raise ValueError(f"user row {i}: fewer than {n} unrated items to sample from")
        chosen = np.empty(0, dtype=np.int64)
        while len(chosen) < n:
            draw = rng.choice(pool, size=2 * n)
            draw = draw[~np.isin(draw, rated)]
            chosen = pd.unique(np.concatenate([chosen, draw]))  # keeps draw order
        out[i] = chosen[:n]
    return out


# ---------------------------------------------------------------------------------------------
# Ranking primitives
# ---------------------------------------------------------------------------------------------

def mask_seen(scores: np.ndarray, seen_rows: sp.csr_matrix) -> np.ndarray:
    """Set seen items to -inf in place (scores must be a writable copy)."""
    r = np.repeat(np.arange(seen_rows.shape[0]), np.diff(seen_rows.indptr))
    scores[r, seen_rows.indices] = -np.inf
    return scores


def top_k(scores: np.ndarray, k: int = K, window: int = 64) -> np.ndarray:
    """Top-k per row by (-score, item index), finite scores only. Returns (B, k), -1 padded.

    Exact answer = items scoring strictly above the row's k-th best score, ordered by
    (-score, index), then items tied AT that score in ascending index order, until k are taken.
    torch.topk over a window of k + `window` items gives the k-th score exactly and always
    contains every strictly-better item. If the tie group at the k-th score also fits inside the
    window, the row is solved from the window in one vectorized lexsort. Otherwise (huge tie
    groups, e.g. a model that scores most movies 0), the row takes its tied items with one linear
    scan, never a sort over all ties. (np.partition was ~20x slower: its selection degrades on
    rows with many equal scores.)
    """
    scores = np.atleast_2d(scores)
    b, n = scores.shape
    out = np.full((b, k), -1, dtype=np.int64)
    if n == 0 or k == 0 or b == 0:
        return out
    kk = min(k, n)
    m = min(n, kk + window)
    tv, ti = torch.from_numpy(np.ascontiguousarray(scores)).topk(m, dim=1)
    vals, idx = tv.numpy(), ti.numpy()
    kth = vals[:, kk - 1]
    # Tie group at the k-th score fully inside the window (or no finite item to tie with).
    complete = (m == n) | (vals[:, m - 1] < kth) | ~np.isfinite(kth)

    rows = np.flatnonzero(complete)
    if len(rows):
        v, c = vals[rows], idx[rows]
        r, j = np.nonzero((v >= kth[rows, None]) & np.isfinite(v))
        cv, cc = v[r, j], c[r, j]
        order = np.lexsort((cc, -cv, r))
        r, cc = r[order], cc[order]
        pos = np.arange(len(r)) - np.searchsorted(r, np.arange(len(rows)))[r]
        keep = pos < kk
        out[rows[r[keep]], pos[keep]] = cc[keep]

    for i in np.flatnonzero(~complete):
        t = kth[i]
        better = vals[i] > t  # all strictly-better items are inside the window
        bc, bv = idx[i][better], vals[i][better]
        bc = bc[np.lexsort((bc, -bv))]
        ties = np.flatnonzero(scores[i] == t)[:kk - len(bc)]  # ascending index
        top = np.concatenate([bc, ties])
        out[i, :len(top)] = top
    return out


def user_metrics(top: np.ndarray, relevant: np.ndarray, k: int = K):
    """Capped recall and NDCG for one user. top: item indices (-1 = empty slot)."""
    top = top[top >= 0]
    hits = np.isin(top, relevant).astype(np.float64)
    denom = min(len(relevant), k)
    recall = hits.sum() / denom
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((hits * discounts[:len(hits)]).sum())
    idcg = float(discounts[:denom].sum())
    return recall, dcg / idcg, int(hits.sum())


def auc_score(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    """P(pos > neg) over all pairs, ties count 0.5 (including -inf vs -inf)."""
    p = pos_scores[:, None]
    q = neg_scores[None, :]
    return float((p > q).mean() + 0.5 * (p == q).mean())


# ---------------------------------------------------------------------------------------------
# Evaluate a model
# ---------------------------------------------------------------------------------------------

@dataclass
class EvalResult:
    summary: dict
    per_user: pd.DataFrame


def evaluate(model, data: EvalData, k: int = K, batch_size: int = 512) -> EvalResult:
    """Score `data.users` with `model.score` and return summary + per-user metrics."""
    n_items = len(data.items)
    per_user = {"userId": data.users, "n_relevant": np.diff(data.relevant.indptr),
                "hits": np.zeros(data.n_users, dtype=np.int32),
                "recall": np.full(data.n_users, np.nan), "ndcg": np.full(data.n_users, np.nan),
                "auc": np.full(data.n_users, np.nan)}
    recommended = np.zeros(n_items, dtype=bool)
    all_seen = np.diff(data.seen.indptr) >= n_items  # catalog exhausted: nothing to rank

    for start in range(0, data.n_users, batch_size):
        stop = min(start + batch_size, data.n_users)
        # Own C-ordered copy: never mutate the model's array, and np.array() on a broadcast
        # array would otherwise pick column-major order, which makes every row op slow.
        scores = np.array(model.score(data.users[start:stop]), dtype=np.float64, order="C")
        if scores.shape != (stop - start, n_items):
            raise ValueError(f"model returned {scores.shape}, expected {(stop - start, n_items)}")
        if not (scores < np.inf).all():  # one pass: False for NaN and +inf
            raise ValueError("model returned NaN or +inf scores")
        mask_seen(scores, data.seen[start:stop])
        top = top_k(scores, k)
        for j in range(stop - start):
            u = start + j
            if all_seen[u]:
                continue
            rel = data.relevant.indices[data.relevant.indptr[u]:data.relevant.indptr[u + 1]]
            r, n, h = user_metrics(top[j], rel, k)
            per_user["recall"][u], per_user["ndcg"][u], per_user["hits"][u] = r, n, h
            per_user["auc"][u] = auc_score(scores[j, rel], scores[j, data.negatives[u]])
            t = top[j][top[j] >= 0]
            recommended[t] = True

    df = pd.DataFrame(per_user)
    ok = ~all_seen
    covered = recommended & data.train_items
    summary = {
        "scheme": data.scheme, "slice": data.slice,
        "recall@10": float(df.recall[ok].mean()),
        "ndcg@10": float(df.ndcg[ok].mean()),
        "auc": float(df.auc[ok].mean()),
        "coverage": float(covered.sum() / data.train_items.sum()),
        "n_items_recommended": int(recommended.sum()),
        "coverage_denominator": int(data.train_items.sum()),
        "n_users": int(ok.sum()),
        "skipped_all_seen": int(all_seen.sum()),
        "excluded_no_train_positive": data.counts.get("users_no_train_positive"),
        "excluded_no_target_positive": data.counts.get("users_no_target_positive"),
    }
    return EvalResult(summary, df[ok].reset_index(drop=True))


# ---------------------------------------------------------------------------------------------
# Comparing and recording runs
# ---------------------------------------------------------------------------------------------

def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str, n_boot: int = 1000,
                     seed: int = 0) -> dict:
    """Mean of (a - b) over the same users, with a 95% bootstrap CI (users resampled)."""
    m = a[["userId", metric]].merge(b[["userId", metric]], on="userId", suffixes=("_a", "_b"))
    if len(m) != len(a) or len(m) != len(b):
        raise ValueError("paired bootstrap needs both runs on exactly the same users")
    d = (m[f"{metric}_a"] - m[f"{metric}_b"]).to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"metric": metric, "mean_diff": float(d.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "n_users": len(d)}


RESULT_COLUMNS = ["model", "scheme", "slice", "seed", "recall@10", "ndcg@10", "auc", "coverage",
                  "n_items_recommended", "coverage_denominator", "n_users", "skipped_all_seen",
                  "excluded_no_train_positive", "excluded_no_target_positive", "subsample",
                  "config"]


def result_row(model: str, seed: int, config: dict, result: EvalResult, data: EvalData) -> dict:
    return {**{c: None for c in RESULT_COLUMNS}, **result.summary, "model": model, "seed": seed,
            "subsample": data.counts.get("subsample"),
            "config": json.dumps(config, sort_keys=True)}


def append_results(rows: list[dict], path) -> None:
    """Append run summaries to a results CSV (created with a header if missing)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.6f}" if isinstance(r[c], float) else r[c])
                        for c in RESULT_COLUMNS})


def save_per_user(result: EvalResult, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    result.per_user.to_parquet(path, index=False)


def summarize_seeds(df: pd.DataFrame, metrics=("recall@10", "ndcg@10", "auc", "coverage")):
    """Median and spread (min, max) over seeds, per model/scheme/slice."""
    g = df.groupby(["model", "scheme", "slice"])
    out = g[list(metrics)].agg(["median", "min", "max"])
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    out["n_seeds"] = g["seed"].nunique()
    return out.reset_index()


# ---------------------------------------------------------------------------------------------
# Reference models for checking the harness itself
# ---------------------------------------------------------------------------------------------

class OracleModel:
    """Scores each user's relevant items highest: must reach recall = NDCG = AUC = 1."""

    def __init__(self, data: EvalData):
        self.data, self.row = data, {u: i for i, u in enumerate(data.users)}

    def score(self, users):
        rows = [self.row[u] for u in users]
        return self.data.relevant[rows].toarray().astype(np.float64)


class RandomModel:
    """Uniform random scores: recall ~ 10 / catalog size, AUC ~ 0.5."""

    def __init__(self, n_items: int, seed: int = 0):
        self.n_items, self.seed = n_items, seed

    def score(self, users):
        # Seeded per user, so a user's scores don't depend on how users are batched.
        return np.stack([np.random.default_rng([self.seed, int(u)]).random(self.n_items)
                         for u in users])
