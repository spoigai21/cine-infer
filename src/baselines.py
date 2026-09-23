"""Phase 3: the four baselines, all scored through the Phase 2 harness.

Every model trains on the same input: the binary matrix of positives (rating >= 4, §1.6) from the
training slice (train when scoring val; train + val for the final test run, §2.1). Ratings below
4 are not model input. They are only masked at evaluation, which the harness does.

Models implement `score(user_ids) -> (B, n_items)` over the full catalog. Movies a model can't
score get -inf:

- MostPopular   count of train positives per movie. No hyperparameters.
- ItemKNN       cosine similarity between movies over the binary matrix, top-k neighbours per
                movie, similarity shrinkage. Movies with < min_pos train positives are dropped
                (-inf): the dense movie x movie matrix doesn't fit otherwise.
- EASE          closed form (Steck 2019), same movie cutoff for the same reason.
- ImplicitALS   Spark MLlib ALS with implicitPrefs=True. Movies with no train positive have no
                factor (-inf).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data_prep import POSITIVE_THRESHOLD
from src.evaluate import ItemIndex, top_k


@dataclass
class TrainData:
    """Binary positives matrix over ALL users (not just evaluated ones) and the full catalog."""
    X: sp.csr_matrix          # users x items, float32, 1 = positive in the training slice
    user_ids: np.ndarray      # sorted userIds; row i of X is user_ids[i]
    items: ItemIndex
    training: tuple           # which slices went in, e.g. ("train",)

    @property
    def item_pos(self) -> np.ndarray:
        return np.asarray(self.X.sum(axis=0)).ravel()

    def rows(self, user_ids) -> np.ndarray:
        user_ids = np.asarray(user_ids)
        r = np.searchsorted(self.user_ids, user_ids)
        r = np.minimum(r, len(self.user_ids) - 1)
        if not np.array_equal(self.user_ids[r], user_ids):
            raise KeyError("userId not in training data")
        return r


def build_train_data(ratings: pd.DataFrame, items: ItemIndex, training=("train",)) -> TrainData:
    """ratings: userId, movieId, rating, split (one scheme), as from evaluate.load_ratings."""
    user_ids = np.unique(ratings["userId"].to_numpy())
    m = (ratings["split"].isin(training) & (ratings["rating"] >= POSITIVE_THRESHOLD)).to_numpy()
    rows = np.searchsorted(user_ids, ratings["userId"].to_numpy()[m])
    cols = items.to_index(ratings["movieId"].to_numpy()[m])
    X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                      shape=(len(user_ids), len(items)))
    X.sum_duplicates()
    return TrainData(X, user_ids, items, tuple(training))


def _scatter(sub_scores: np.ndarray, keep: np.ndarray, n_items: int) -> np.ndarray:
    """Place scores for the kept movie columns into a full-catalog array, -inf elsewhere."""
    out = np.full((sub_scores.shape[0], n_items), -np.inf)
    out[:, keep] = sub_scores
    return out


# ---------------------------------------------------------------------------------------------
# Most popular
# ---------------------------------------------------------------------------------------------

class MostPopular:
    name = "most_popular"

    def __init__(self, td: TrainData):
        self.pop = td.item_pos.astype(np.float64)
        self.config = {}

    def score(self, user_ids):
        return np.broadcast_to(self.pop, (len(user_ids), len(self.pop)))


# ---------------------------------------------------------------------------------------------
# Item-kNN
# ---------------------------------------------------------------------------------------------

class ItemGram:
    """X^T X over movies with >= min_pos positives. Shared by ItemKNN configs and EASE."""

    def __init__(self, td: TrainData, min_pos: int):
        self.min_pos = min_pos
        self.keep = np.flatnonzero(td.item_pos >= min_pos)
        Xk = td.X[:, self.keep].tocsc().astype(np.float64)
        self.G = (Xk.T @ Xk).toarray()  # dense, n_keep x n_keep, float64


class ItemKNN:
    name = "item_knn"

    def __init__(self, td: TrainData, gram: ItemGram, k: int, shrink: float):
        self.td, self.keep = td, gram.keep
        self.config = {"k": k, "shrink": shrink, "min_pos": gram.min_pos}
        G = gram.G
        norms = np.sqrt(np.diag(G))
        S = (G / (np.outer(norms, norms) + shrink)).astype(np.float32)
        np.fill_diagonal(S, 0.0)
        n = S.shape[0]
        k = min(k, n - 1)
        # Keep the k most similar neighbours of each target movie j (column j of S), ties broken
        # by movie index so the model is identical on every run (argpartition picks arbitrarily).
        sel = S.T.astype(np.float64)                     # row j = similarities to target j
        np.fill_diagonal(sel, -np.inf)                   # a movie is never its own neighbour
        nbr = np.vstack([top_k(sel[i:i + 2048], k) for i in range(0, n, 2048)])  # n x k
        cols = np.repeat(np.arange(n), k)
        rows = nbr.ravel()
        ok = rows >= 0
        self.S = sp.csr_matrix((S[rows[ok], cols[ok]], (rows[ok], cols[ok])), shape=(n, n))
        self.Xk = td.X[:, self.keep].tocsr()

    def score(self, user_ids):
        h = self.Xk[self.td.rows(user_ids)]
        return _scatter((h @ self.S).toarray(), self.keep, len(self.td.items))


# ---------------------------------------------------------------------------------------------
# EASE
# ---------------------------------------------------------------------------------------------

class EASE:
    name = "ease"

    def __init__(self, td: TrainData, gram: ItemGram, lam: float):
        self.td, self.keep = td, gram.keep
        self.config = {"lam": lam, "min_pos": gram.min_pos}
        G = gram.G.copy()
        G[np.diag_indices_from(G)] += lam
        P = np.linalg.inv(G)
        B = P / (-np.diag(P))
        B[np.diag_indices_from(B)] = 0.0
        self.B = B.astype(np.float32)
        self.Xk = td.X[:, self.keep].tocsr()

    def score(self, user_ids):
        h = self.Xk[self.td.rows(user_ids)]
        return _scatter(h @ self.B, self.keep, len(self.td.items))


# ---------------------------------------------------------------------------------------------
# Implicit ALS (Spark MLlib)
# ---------------------------------------------------------------------------------------------

def positives_for_spark(spark, splits_path, scheme: str, training):
    """Train positives as a Spark DataFrame (userId, movieId, label = 1.0), read from parquet."""
    from pyspark.sql import functions as F
    col = f"split_{scheme}"
    return (spark.read.parquet(str(splits_path))
            .filter(F.col(col).isin(*training) & (F.col("rating") >= POSITIVE_THRESHOLD))
            .select("userId", "movieId", F.lit(1.0).alias("label")))


class ImplicitALS:
    name = "implicit_als"

    def __init__(self, td: TrainData, positives, rank: int, reg: float, alpha: float,
                 max_iter: int = 15, seed: int = 42):
        from pyspark.ml.recommendation import ALS
        self.td = td
        self.config = {"rank": rank, "reg": reg, "alpha": alpha, "max_iter": max_iter}
        self.seed = seed
        t = time.time()
        model = ALS(userCol="userId", itemCol="movieId", ratingCol="label",
                    implicitPrefs=True,  # NOT the default: the default fits explicit ratings
                    rank=rank, regParam=reg, alpha=alpha, maxIter=max_iter, seed=seed,
                    numUserBlocks=10, numItemBlocks=10, checkpointInterval=5).fit(positives)
        self.fit_seconds = time.time() - t
        U = model.userFactors.toPandas()
        V = model.itemFactors.toPandas()
        n_items = len(td.items)
        # Users with no positives have no factor: a zero vector (all-equal scores).
        self.U = np.zeros((len(td.user_ids), rank), dtype=np.float32)
        self.U[td.rows(U["id"].to_numpy())] = np.stack(U["features"].to_numpy())
        self.V = np.zeros((n_items, rank), dtype=np.float32)
        self.has_factor = np.zeros(n_items, dtype=bool)
        vi = td.items.to_index(V["id"].to_numpy())
        self.V[vi] = np.stack(V["features"].to_numpy())
        self.has_factor[vi] = True

    def score(self, user_ids):
        s = (self.U[self.td.rows(user_ids)] @ self.V.T).astype(np.float64)
        s[:, ~self.has_factor] = -np.inf
        return s
