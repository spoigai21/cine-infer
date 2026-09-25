"""Phase 6: second-stage ranker over the two-tower model's top-200 candidates.

Pipeline for one TRAINING SET (train_core -> ranker labels from train_tail; train -> validation;
train + val -> the Phase 6b test run). Everything a feature sees comes from that one training set:

  candidates   two-tower retriever trained on that set; its top K = 200 per user, after masking
               every movie the user rated in that set (K is fixed: prediction #6 was committed
               for "the ranker over 200 candidates")
  features     per (user, candidate), all float32, NaN = missing (LightGBM handles it):
    tt_score, tt_rank                      the retriever's own score and rank
    ease_score, ease_rank                  EASE (tuned config) fit on the same set; NaN when EASE
                                           dropped the movie (movie cutoff)
    item_log_n, item_log_pos, item_mean    Phase 1 item features of that set
    item_days_since_last, item_age_days    relative to the user's last rating in that set
    user_log_n, user_log_pos, user_mean    Phase 1 user features of that set
    genre_overlap, genre_max               user genre shares (Phase 1) x the movie's genres
    genome_sim_all, genome_sim_recent      cosine between the movie's tag-genome vector and the
                                           mean genome of the user's positives (all / last 10);
                                           NaN when either side has no genome
    has_genome
  ranker       LightGBM LambdaRank, one query group per user, labels = candidate is a positive in
               the label slice. Users with no positive among their candidates carry no ranking
               signal and are dropped (counted).

The LightGBM ranker runs in a separate process (src/lgb_ranker.py): this module writes features
to disk and reads predictions back through PrecomputedScorer, which plugs into the Phase 2
harness (candidates get ranker scores, everything else -inf).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.evaluate import ItemIndex, top_k

K_CANDIDATES = 200
RECENT_PROFILE = 10
FEATURES = ["tt_score", "tt_rank", "ease_score", "ease_rank",
            "item_log_n", "item_log_pos", "item_mean", "item_days_since_last", "item_age_days",
            "user_log_n", "user_log_pos", "user_mean",
            "genre_overlap", "genre_max",
            "genome_sim_all", "genome_sim_recent", "has_genome"]
EASE_FEATURES = ["ease_score", "ease_rank"]
DAY = 86_400.0


# ---------------------------------------------------------------------------------------------
# Static item data (not time-dependent): genres and the tag genome
# ---------------------------------------------------------------------------------------------

@dataclass
class ItemContent:
    genres: sp.csr_matrix       # items x genres, 0/1
    genre_names: list
    genome: np.ndarray          # items x 1128, L2-normalised; zero rows where missing
    has_genome: np.ndarray      # bool per item

    @property
    def genome_rows(self):      # only the movies that have a genome vector
        if not hasattr(self, "_rows"):
            self._rows = self.genome[self.has_genome]
        return self._rows

    @property
    def genome_row(self):       # item index -> row in genome_rows (-1 if none)
        if not hasattr(self, "_row"):
            self._row = np.full(len(self.has_genome), -1, dtype=np.int64)
            self._row[self.has_genome] = np.arange(self.has_genome.sum())
        return self._row

    @classmethod
    def load(cls, movies_csv, genome_parquet, items: ItemIndex) -> "ItemContent":
        m = pd.read_csv(movies_csv)
        g = m.assign(genre=m.genres.str.split("|")).explode("genre")
        g = g[g.genre != "(no genres listed)"]
        names = sorted(g.genre.unique())
        col = {n: i for i, n in enumerate(names)}
        genres = sp.csr_matrix((np.ones(len(g), dtype=np.float32),
                                (items.to_index(g.movieId.to_numpy()), g.genre.map(col))),
                               shape=(len(items), len(names)))
        genome = np.zeros((len(items), 0), dtype=np.float32)
        has = np.zeros(len(items), dtype=bool)
        if genome_parquet is not None:
            gp = pd.read_parquet(genome_parquet)
            vec = np.stack(gp.genome.to_numpy()).astype(np.float32)
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)
            genome = np.zeros((len(items), vec.shape[1]), dtype=np.float32)
            idx = items.to_index(gp.movieId.to_numpy())
            genome[idx] = vec
            has[idx] = True
        return cls(genres, names, genome, has)


# ---------------------------------------------------------------------------------------------
# Everything that depends on ONE training set
# ---------------------------------------------------------------------------------------------

@dataclass
class TrainingSetContext:
    name: str                   # train_core | train | train_val
    retriever: object           # TwoTowerScorer trained on this set
    ease: object | None         # baselines.EASE fit on this set (None = no EASE features)
    seq: object                 # two_tower.Sequences of this set's positives (for profiles)
    seen: sp.csr_matrix         # users (seq.user_ids rows) x items: rated ANY rating in this set
    items: ItemIndex
    content: ItemContent
    item_stats: np.ndarray      # items x 5: log n, log pos, mean, first ts, last ts (NaN absent)
    user_stats: np.ndarray      # users x 4: log n, log pos, mean, last ts (NaN absent)
    user_genres: np.ndarray     # users x genres, share of positives
    k: int = K_CANDIDATES
    extra: dict = field(default_factory=dict)

    @classmethod
    def build(cls, name, retriever, ease, seq, ratings_fine, training, features_dir, items,
              content, k=K_CANDIDATES):
        """ratings_fine: evaluate.load_ratings(..., fine=True); training: fine slice names."""
        users = seq.user_ids
        r = ratings_fine[ratings_fine.split.isin(training)]
        rows = np.searchsorted(users, r.userId.to_numpy())
        seen = sp.csr_matrix((np.ones(len(r), dtype=np.int8),
                              (rows, items.to_index(r.movieId.to_numpy()))),
                             shape=(len(users), len(items)))
        it = pd.read_parquet(f"{features_dir}/{name}/items.parquet")
        item_stats = np.full((len(items), 5), np.nan, dtype=np.float64)
        ii = items.to_index(it.movieId.to_numpy())
        item_stats[ii] = np.column_stack([np.log1p(it.n_ratings), np.log1p(it.n_positives),
                                          it.mean_rating, it.first_rated_ts, it.last_rated_ts])
        us = pd.read_parquet(f"{features_dir}/{name}/users.parquet")
        user_stats = np.full((len(users), 4), np.nan, dtype=np.float64)
        ui = np.searchsorted(users, us.userId.to_numpy())
        user_stats[ui] = np.column_stack([np.log1p(us.n_ratings), np.log1p(us.n_positives),
                                          us.mean_rating, us.last_rated_ts])
        ug = pd.read_parquet(f"{features_dir}/{name}/user_genres.parquet")
        gcol = {n: i for i, n in enumerate(content.genre_names)}
        user_genres = np.zeros((len(users), len(gcol)), dtype=np.float32)
        user_genres[np.searchsorted(users, ug.userId.to_numpy()), ug.genre.map(gcol)] = ug.share
        return cls(name, retriever, ease, seq, seen, items, content, item_stats, user_stats,
                   user_genres, k)


def candidates(ctx: TrainingSetContext, user_ids):
    """Top-k retriever candidates per user after masking everything rated in the training set.
    Returns (item indices B x k, retriever scores B x k)."""
    rows = ctx.seq.rows(user_ids)
    s = np.array(ctx.retriever.score(user_ids), dtype=np.float64, order="C")
    seen = ctx.seen[rows]
    s[np.repeat(np.arange(len(rows)), np.diff(seen.indptr)), seen.indices] = -np.inf
    cand = top_k(s, ctx.k)  # -1 = empty slot (fewer than k scoreable movies; not on ML-25M)
    sc = np.take_along_axis(s, np.maximum(cand, 0), axis=1)
    return cand, np.where(cand >= 0, sc, -np.inf)


def _rank_desc(x):
    """0-based rank of each column by descending value per row; NaN ranks last."""
    key = np.where(np.isnan(x), -np.inf, x)
    order = np.argsort(-key, axis=1, kind="stable")
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(x.shape[1])[None, :].repeat(len(x), 0), axis=1)
    return rank.astype(np.float32)


def build_features(ctx: TrainingSetContext, user_ids, use_ease: bool = True):
    """Candidates + feature tensor (B x k x len(FEATURES)) for these users."""
    user_ids = np.asarray(user_ids)
    rows = ctx.seq.rows(user_ids)
    cand_raw, tt = candidates(ctx, user_ids)
    valid = cand_raw >= 0
    cand = np.maximum(cand_raw, 0)  # gather with a dummy index; empty slots are NaN-ed below
    B, k = cand.shape
    F = {}
    F["tt_score"] = tt.astype(np.float32)
    F["tt_rank"] = np.broadcast_to(np.arange(k, dtype=np.float32), (B, k))
    if use_ease and ctx.ease is not None:
        es = np.take_along_axis(np.asarray(ctx.ease.score(user_ids)), cand, axis=1)
        es = np.where(np.isfinite(es) & valid, es, np.nan).astype(np.float32)  # empty slots: NaN,
        F["ease_score"], F["ease_rank"] = es, _rank_desc(es)  # so they can't shift real ranks
    else:
        F["ease_score"] = F["ease_rank"] = np.full((B, k), np.nan, dtype=np.float32)
    ist = ctx.item_stats[cand]                       # B x k x 5
    ust = ctx.user_stats[rows]                       # B x 4
    F["item_log_n"], F["item_log_pos"], F["item_mean"] = ist[..., 0], ist[..., 1], ist[..., 2]
    user_last = ust[:, 3:4]
    F["item_days_since_last"] = (user_last - ist[..., 4]) / DAY
    F["item_age_days"] = (user_last - ist[..., 3]) / DAY
    for j, name in enumerate(["user_log_n", "user_log_pos", "user_mean"]):
        F[name] = np.broadcast_to(ust[:, j:j + 1], (B, k))
    ig = ctx.content.genres[cand.ravel()].toarray().reshape(B, k, -1)
    ov = ig * ctx.user_genres[rows][:, None, :]
    F["genre_overlap"], F["genre_max"] = ov.sum(-1), ov.max(-1)
    sim_all, sim_recent = genome_similarity(ctx, rows, cand)
    F["genome_sim_all"], F["genome_sim_recent"] = sim_all, sim_recent
    F["has_genome"] = ctx.content.has_genome[cand].astype(np.float32)
    X = np.stack([np.asarray(F[f], dtype=np.float32) for f in FEATURES], axis=-1)
    X[~valid] = np.nan
    return cand_raw, X


def genome_similarity(ctx: TrainingSetContext, rows, cand):
    """Cosine(movie genome, mean genome of the user's positives), all and last-N profiles."""
    G, has = ctx.content.genome, ctx.content.has_genome
    B, k = cand.shape
    if G.shape[1] == 0:
        nan = np.full((B, k), np.nan, dtype=np.float32)
        return nan, nan
    out = []
    for recent in (None, RECENT_PROFILE):
        lens = ctx.seq.lengths[rows]
        starts = ctx.seq.offsets[rows] + (0 if recent is None else np.maximum(0, lens - recent))
        stops = ctx.seq.offsets[rows] + lens
        n = stops - starts
        r = np.repeat(np.arange(B), n)
        it = ctx.seq.items[np.arange(n.sum()) + np.repeat(starts - np.cumsum(n) + n, n)]
        keep = has[it]
        P = sp.csr_matrix((np.ones(keep.sum(), dtype=np.float32), (r[keep], it[keep])),
                          shape=(B, len(has))) @ G                     # B x dims (sum)
        norm = np.linalg.norm(P, axis=1, keepdims=True)
        P = np.divide(P, norm, out=np.zeros_like(P), where=norm > 0)
        # similarities against the genome movies only (B x 13.8k), then pick the candidates;
        # G[cand] would be B x k x 1128 (~0.9 GB per batch)
        gi = ctx.content.genome_row
        S = P @ ctx.content.genome_rows.T
        sim = np.where(has[cand], np.take_along_axis(S, np.maximum(gi[cand], 0), axis=1), np.nan)
        sim[norm[:, 0] == 0] = np.nan
        out.append(sim.astype(np.float32))
    return out[0], out[1]


# ---------------------------------------------------------------------------------------------
# Ranker training data + model
# ---------------------------------------------------------------------------------------------

def labelled_rows(ctx: TrainingSetContext, user_ids, label_items: sp.csr_matrix, label_rows,
                  use_ease=True, batch=1024):
    """Features + labels for ranker training. label_items: users x items positives of the label
    slice, row label_rows[i] for user_ids[i]. Users with no positive candidate are dropped."""
    Xs, ys, groups, dropped = [], [], [], 0
    for b in range(0, len(user_ids), batch):
        u = user_ids[b:b + batch]
        cand, X = build_features(ctx, u, use_ease)
        valid = cand >= 0
        L = label_items[label_rows[b:b + batch]]
        y = np.asarray(L[np.arange(len(u))[:, None], np.maximum(cand, 0)].todense(),
                       dtype=np.float32) * valid
        keep = y.sum(1) > 0
        dropped += int((~keep).sum())
        for i in np.flatnonzero(keep):  # one query group per user, empty slots left out
            Xs.append(X[i][valid[i]])
            ys.append(y[i][valid[i]])
            groups.append(int(valid[i].sum()))
    if not Xs:
        raise ValueError("no user has a label-slice positive among their candidates")
    return np.concatenate(Xs), np.concatenate(ys), np.array(groups), dropped


# ---------------------------------------------------------------------------------------------
# Hand-off to the LightGBM process (src/lgb_ranker.py) through files
#
# PyTorch and LightGBM each bring their own OpenMP runtime on macOS; in one process, LightGBM's
# multithreaded training segfaults. So this (PyTorch) process only builds features and scores
# results, and LightGBM runs in a separate process that never imports torch.
# ---------------------------------------------------------------------------------------------

def write_training_rows(ctx, user_ids, label_items, label_rows, out_dir, use_ease=True):
    """Ranker-training rows -> out_dir/train_{X,y,groups}.npy + meta.json."""
    import json
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y, groups, dropped = labelled_rows(ctx, user_ids, label_items, label_rows, use_ease)
    np.save(out / "train_X.npy", X)
    np.save(out / "train_y.npy", y)
    np.save(out / "train_groups.npy", groups)
    meta = {"features": FEATURES, "train_users": int(len(user_ids)),
            "train_users_kept": int(len(groups)), "train_users_dropped_no_positive": dropped,
            "train_rows": int(len(y)), "train_positives": int(y.sum()), "k": ctx.k}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def write_eval_features(ctx, user_ids, out_dir, use_ease=True, batch=1024):
    """Candidates + features for scoring -> out_dir/eval_{users,cand,X}.npy (X memory-mapped)."""
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    user_ids = np.asarray(user_ids)
    B, k = len(user_ids), ctx.k
    cand_mm = np.lib.format.open_memmap(out / "eval_cand.npy", "w+", np.int32, (B, k))
    X_mm = np.lib.format.open_memmap(out / "eval_X.npy", "w+", np.float32, (B, k, len(FEATURES)))
    for b in range(0, B, batch):
        cand, X = build_features(ctx, user_ids[b:b + batch], use_ease)
        cand_mm[b:b + len(cand)] = cand
        X_mm[b:b + len(cand)] = X
    cand_mm.flush()
    X_mm.flush()
    np.save(out / "eval_users.npy", user_ids)


def run_lgb(data_dir, config, seed, features=None, rows=None, fixed_rounds=None,
            save_model=None, pred_out=None, predict=True):
    """Train + predict in a separate process (src/lgb_ranker.py). Returns its JSON report."""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path
    pred_out = Path(pred_out or tempfile.mktemp(suffix=".npy")).resolve()
    data_dir = Path(data_dir).resolve()
    save_model = Path(save_model).resolve() if save_model is not None else None
    repo = Path(__file__).resolve().parents[1]  # `-m src.lgb_ranker` must run from the repo root
    cmd = [sys.executable, "-m", "src.lgb_ranker", "--data", str(data_dir),
           "--config", json.dumps(config), "--seed", str(seed), "--pred-out", str(pred_out)]
    if features is not None:
        cmd += ["--features", ",".join(features)]
    if rows is not None:
        rows_path = pred_out.with_suffix(".rows.npy")
        np.save(rows_path, np.asarray(rows))
        cmd += ["--rows", str(rows_path)]
    if fixed_rounds is not None:
        cmd += ["--fixed-rounds", str(fixed_rounds)]
    if save_model is not None:
        cmd += ["--save-model", str(save_model)]
    if not predict:
        cmd += ["--no-predict"]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=repo)
    if res.returncode != 0:
        raise RuntimeError(f"lgb_ranker failed ({res.returncode}):\n{res.stderr[-3000:]}")
    report = json.loads(res.stdout.strip().splitlines()[-1])
    report["pred_path"] = str(pred_out)
    return report


class PrecomputedScorer:
    """Harness adapter over saved candidates + ranker predictions. Candidates always rank above
    every non-candidate, in ranker order.

    Without `fallback`, non-candidates get -inf. With `fallback` (the retriever that produced the
    candidates), non-candidates keep the retriever's score and candidates are shifted above the
    row's best retriever score. The top-k is the same either way (K candidates >> k), but AUC
    needs it: with -inf, every held-out positive the retriever missed ties with nearly every
    sampled negative, and AUC reflects the candidate cut-off rather than the ranking."""
    name = "two_stage"

    def __init__(self, users, cand, pred, n_items, config=None, fallback=None):
        self.users = np.asarray(users)
        self.order = np.argsort(self.users)
        self.cand, self.pred, self.n_items = cand, pred, n_items
        self.config = config or {}
        self.fallback = fallback

    def score(self, user_ids):
        user_ids = np.asarray(user_ids)
        pos = np.searchsorted(self.users[self.order], user_ids)
        idx = self.order[np.minimum(pos, len(self.order) - 1)]
        if not np.array_equal(self.users[idx], user_ids):
            raise KeyError("no precomputed candidates for some users")
        cand, pred = np.asarray(self.cand[idx]), np.asarray(self.pred[idx], dtype=np.float64)
        r, c = np.nonzero(cand >= 0)
        if self.fallback is None:
            out = np.full((len(user_ids), self.n_items), -np.inf)
            out[r, cand[r, c]] = pred[r, c]
            return out
        out = np.array(self.fallback.score(user_ids), dtype=np.float64)
        fin = np.where(np.isfinite(out), out, -np.inf).max(1)
        top = np.where(np.isfinite(fin), fin, 0.0)
        low = np.where(cand >= 0, pred, np.inf).min(1)
        shift = top - np.where(np.isfinite(low), low, 0.0) + 1.0  # min candidate = top + 1
        out[r, cand[r, c]] = pred[r, c] + shift[r]
        return out


def recall_ceiling(ctx: TrainingSetContext, data, ks=(50, 100, 200, 500), batch=1024):
    """Mean share of each user's relevant items that reach the retriever's top-k (per k)."""
    kmax = max(ks)
    old, ctx.k = ctx.k, kmax
    hits = {k: [] for k in ks}
    try:
        for b in range(0, data.n_users, batch):
            u = data.users[b:b + batch]
            cand, _ = candidates(ctx, u)
            rel = data.relevant[b:b + batch]
            for i in range(len(u)):
                r = rel.indices[rel.indptr[i]:rel.indptr[i + 1]]
                for k in ks:
                    c = cand[i, :k]
                    hits[k].append(np.isin(r, c[c >= 0]).mean())
    finally:
        ctx.k = old
    return {k: float(np.mean(v)) for k, v in hits.items()}
