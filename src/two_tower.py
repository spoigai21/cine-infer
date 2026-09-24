"""Phase 5: two-tower retrieval model (PyTorch).

- Item tower: an embedding per movie.
- User tower: mean of the embeddings of the user's most recent `hist_len` positives (the item
  table is shared), plus a residual MLP. It's built from history, not a user-ID embedding, so new
  history works without retraining.
- Score: cosine similarity / temperature. Both vectors are L2-normalised.
- Loss: softmax over in-batch negatives (+ optional uniform random negatives), with logQ
  correction (Yi et al. 2019), and masking of duplicate movies in the batch (false negatives).

Training pairs (§5.2): for the positive at position t of a user's time-ordered positives
((timestamp, movieId) order, as in §1.2), the history is the positives at positions < t, truncated
to the most recent `hist_len`. Nothing rated after the target is used. Pairs with an empty
history are skipped. Each epoch samples at most `pairs_per_user` pairs per user, so heavy users
don't dominate (the metrics weight users equally).

Scoring (the harness interface): the user vector comes from the most recent `hist_len` positives
of the training data (train when scoring val, train + val for the final test run, §2.1). Movies
with no positive in the training data get -inf: their embeddings were never trained.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data_prep import POSITIVE_THRESHOLD
from src.evaluate import ItemIndex


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


# ---------------------------------------------------------------------------------------------
# Data: per-user time-ordered positives
# ---------------------------------------------------------------------------------------------

@dataclass
class Sequences:
    """Positives of the training data, per user, oldest -> newest. Users are ALL users."""
    user_ids: np.ndarray   # sorted userIds; user row i <-> user_ids[i]
    items: np.ndarray      # int64 item indices, concatenated per user in time order
    offsets: np.ndarray    # int64, len n_users + 1; user i's items are items[offsets[i]:offsets[i+1]]
    n_items: int
    training: tuple
    timestamps: np.ndarray = None  # int64, aligned with items

    @property
    def lengths(self):
        return np.diff(self.offsets)

    def rows(self, user_ids):
        user_ids = np.asarray(user_ids)
        r = np.minimum(np.searchsorted(self.user_ids, user_ids), len(self.user_ids) - 1)
        if not np.array_equal(self.user_ids[r], user_ids):
            raise KeyError("userId not in training data")
        return r

    def trained_items(self) -> np.ndarray:
        m = np.zeros(self.n_items, dtype=bool)
        m[self.items] = True
        return m


def load_sequences(splits_path, items: ItemIndex, training=("train",), scheme="user") -> Sequences:
    col = f"split_{scheme}"
    t = pq.read_table(splits_path, columns=["userId", "movieId", "rating", "timestamp", col],
                      read_dictionary=[col]).to_pandas()
    user_ids = np.unique(t["userId"].to_numpy())
    t = t[t[col].isin(training) & (t["rating"] >= POSITIVE_THRESHOLD)]
    t = t.sort_values(["userId", "timestamp", "movieId"], kind="stable")
    rows = np.searchsorted(user_ids, t["userId"].to_numpy())
    counts = np.bincount(rows, minlength=len(user_ids))
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return Sequences(user_ids, items.to_index(t["movieId"].to_numpy()).astype(np.int64),
                     offsets, len(items), tuple(training), t["timestamp"].to_numpy(np.int64))


def all_pairs(seq: Sequences):
    """(user row, position t) for every positive with a non-empty history (t >= 1)."""
    n = seq.lengths
    users = np.repeat(np.arange(len(n)), np.maximum(n - 1, 0))
    first = np.repeat(np.cumsum(np.maximum(n - 1, 0)) - np.maximum(n - 1, 0), np.maximum(n - 1, 0))
    pos = np.arange(len(users)) - first + 1
    return users.astype(np.int64), pos.astype(np.int64)


def history_ends(seq: Sequences, users, pos, strict_time: bool):
    """Where each pair's history stops (exclusive, local position).

    tiebreak:    everything before position t, including same-second ratings ordered by movieId.
    strict_time: only ratings with an earlier timestamp than the target. The movieId order inside
                 a same-second burst is an artifact (it's also how the per-user split cut those
                 bursts), so a sequence model must not be able to learn it.
    """
    if not strict_time:
        return pos
    g = group_starts(seq)
    return g[seq.offsets[users] + pos] - seq.offsets[users]


def group_starts(seq: Sequences) -> np.ndarray:
    """For each flat position, the first position of its (user, timestamp) group."""
    n = len(seq.items)
    new = np.ones(n, dtype=bool)
    new[1:] = seq.timestamps[1:] != seq.timestamps[:-1]
    new[seq.offsets[:-1][seq.lengths > 0]] = True  # user boundaries always start a group
    return np.maximum.accumulate(np.where(new, np.arange(n), 0))


def sample_epoch_index(users, cap, rng):
    """Shuffled pair indices, keeping at most `cap` random pairs per user (None = all)."""
    idx = np.arange(len(users))
    if cap is not None:
        order = np.lexsort((rng.random(len(users)), users))
        u = users[order]
        rank = np.arange(len(u)) - np.searchsorted(u, u)
        idx = order[rank < cap]
    return idx[rng.permutation(len(idx))]


def sample_epoch(users, pos, cap, rng):
    i = sample_epoch_index(users, cap, rng)
    return users[i], pos[i]


def history_bags(seq: Sequences, users, ends, hist_len):
    """Flat item indices + bag offsets for histories items[offsets[u] + max(0, end - L) : end]."""
    starts = seq.offsets[users] + np.maximum(0, ends - hist_len)
    stops = seq.offsets[users] + ends
    lens = stops - starts
    bag_off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    idx = np.arange(lens.sum()) + np.repeat(starts - bag_off, lens)
    return seq.items[idx], bag_off


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------

class TwoTower(nn.Module):
    def __init__(self, n_items: int, dim: int, hidden: int):
        super().__init__()
        self.item = nn.Embedding(n_items, dim)
        nn.init.normal_(self.item.weight, std=0.1)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, dim))

    def user_vec(self, hist_items, bag_offsets):
        e = F.embedding_bag(hist_items, self.item.weight, bag_offsets, mode="mean")
        return F.normalize(e + self.mlp(e), dim=-1)

    def item_vec(self, ids=None):
        w = self.item.weight if ids is None else self.item(ids)
        return F.normalize(w, dim=-1)


def in_batch_loss(u, v_pos, targets, log_q, tau, neg_ids=None, v_neg=None, log_q_neg=None):
    """Softmax over in-batch positives (+ optional extra negatives), logQ-corrected."""
    logits = (u @ v_pos.T) / tau - log_q[targets][None, :]
    dup = targets[:, None] == targets[None, :]
    dup.fill_diagonal_(False)
    logits = logits.masked_fill(dup, float("-inf"))
    if neg_ids is not None:
        extra = (u @ v_neg.T) / tau - log_q_neg
        extra = extra.masked_fill(neg_ids[None, :] == targets[:, None], float("-inf"))
        logits = torch.cat([logits, extra], dim=1)
    return F.cross_entropy(logits, torch.arange(len(u), device=u.device))


# ---------------------------------------------------------------------------------------------
# Training + scoring
# ---------------------------------------------------------------------------------------------

DEFAULTS = {"dim": 64, "hidden": 256, "tau": 0.05, "lr": 1e-3, "batch": 4096, "hist_len": 50,
            "pairs_per_user": 100, "n_uniform": 0, "max_epochs": 10, "strict_time": False}


class TwoTowerScorer:
    """Harness adapter: score(user_ids) -> (B, n_items), -inf for untrained movies."""
    name = "two_tower"

    def __init__(self, model, seq: Sequences, hist_len: int, config: dict, dev):
        self.model, self.seq, self.hist_len, self.config, self.dev = model, seq, hist_len, config, dev
        self.untrained = torch.from_numpy(~seq.trained_items()).to(dev)
        with torch.no_grad():
            self.V = model.item_vec().detach()

    @torch.no_grad()
    def score(self, user_ids):
        rows = self.seq.rows(user_ids)
        h, off = history_bags(self.seq, rows, self.seq.lengths[rows], self.hist_len)
        u = self.model.user_vec(torch.from_numpy(h).to(self.dev), torch.from_numpy(off).to(self.dev))
        s = (u @ self.V.T).masked_fill(self.untrained[None, :], float("-inf"))
        return s.cpu().double().numpy()  # MPS has no float64: convert on CPU


def train_two_tower(seq: Sequences, config: dict, seed: int, eval_fn=None, fixed_epochs=None,
                    log=print, dev=None):
    """Train; return (scorer, info).

    With eval_fn (a callable scorer -> metric), the model is scored after every epoch and the best
    epoch's weights are kept; info["epochs"] records it (early stopping ON VALIDATION only).
    With fixed_epochs, trains exactly that many epochs (the §2.1 refit: no early stopping).
    """
    cfg = {**DEFAULTS, **config}
    dev = dev or device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TwoTower(seq.n_items, cfg["dim"], cfg["hidden"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    users, pos = all_pairs(seq)
    ends = history_ends(seq, users, pos, cfg["strict_time"])
    keep = ends > 0  # strict_time: pairs whose whole history shares their second are dropped
    users, pos, ends = users[keep], pos[keep], ends[keep]
    trained = np.flatnonzero(seq.trained_items())
    n_epochs = fixed_epochs or cfg["max_epochs"]
    best = (-np.inf, 0, None)
    history = []
    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        idx = sample_epoch_index(users, cfg["pairs_per_user"], rng)
        eu, ep, ee = users[idx], pos[idx], ends[idx]
        targets_all = seq.items[seq.offsets[eu] + ep]
        # logQ: each movie's probability of being a batch target under THIS epoch's sampling
        q = np.bincount(targets_all, minlength=seq.n_items) / len(targets_all)
        log_q = torch.from_numpy(np.log(np.maximum(q, 1e-12))).float().to(dev)
        log_q_uni = float(np.log(1.0 / len(trained)))
        model.train()
        total, steps = 0.0, 0
        for b in range(0, len(eu), cfg["batch"]):
            bu, be = eu[b:b + cfg["batch"]], ee[b:b + cfg["batch"]]
            if len(bu) < 2:
                continue
            h, off = history_bags(seq, bu, be, cfg["hist_len"])
            tgt = torch.from_numpy(targets_all[b:b + cfg["batch"]]).to(dev)
            u = model.user_vec(torch.from_numpy(h).to(dev), torch.from_numpy(off).to(dev))
            kw = {}
            if cfg["n_uniform"]:
                neg = torch.from_numpy(rng.choice(trained, cfg["n_uniform"])).to(dev)
                kw = {"neg_ids": neg, "v_neg": model.item_vec(neg), "log_q_neg": log_q_uni}
            loss = in_batch_loss(u, model.item_vec(tgt), tgt, log_q, cfg["tau"], **kw)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            steps += 1
        rec = {"epoch": epoch, "loss": total / max(steps, 1), "pairs": len(eu),
               "seconds": round(time.time() - t0, 1)}
        if eval_fn is not None:
            model.eval()
            rec["val"] = eval_fn(TwoTowerScorer(model, seq, cfg["hist_len"], cfg, dev))
            if rec["val"] > best[0]:
                best = (rec["val"], epoch, copy.deepcopy(model.state_dict()))
        history.append(rec)
        log(f"    epoch {epoch}: loss {rec['loss']:.4f}"
            + (f", val {rec['val']:.5f}" if "val" in rec else "") + f" ({rec['seconds']}s)")
        if eval_fn is not None and epoch - best[1] >= 2:  # 2 epochs without improvement
            break
    if eval_fn is not None:
        model.load_state_dict(best[2])
        epochs = best[1]
    else:
        epochs = n_epochs
    model.eval()
    out_cfg = {k: cfg[k] for k in DEFAULTS if k != "max_epochs"}
    out_cfg["epochs"] = epochs
    scorer = TwoTowerScorer(model, seq, cfg["hist_len"], out_cfg, dev)
    return scorer, {"epochs": epochs, "history": history, "device": str(dev)}


# ---------------------------------------------------------------------------------------------
# Persistence (weights live in models/, git-ignored)
# ---------------------------------------------------------------------------------------------

def save_scorer(scorer: TwoTowerScorer, path, seed: int):
    """Weights + everything needed to rebuild the scorer against the same training data."""
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.cpu() for k, v in scorer.model.state_dict().items()},
                "config": scorer.config, "seed": seed, "n_items": scorer.seq.n_items,
                "training": list(scorer.seq.training), "n_positives": int(len(scorer.seq.items))},
               path)


def load_scorer(path, seq: Sequences, dev=None) -> TwoTowerScorer:
    """Rebuild a saved scorer. `seq` must be the training data it was trained on (checked)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if (ck["n_items"], tuple(ck["training"]), ck["n_positives"]) != \
            (seq.n_items, tuple(seq.training), len(seq.items)):
        raise ValueError(f"{path} was trained on different data than the sequences given")
    dev = dev or device()
    cfg = ck["config"]
    model = TwoTower(ck["n_items"], cfg["dim"], cfg["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.to(dev).eval()
    return TwoTowerScorer(model, seq, cfg["hist_len"], cfg, dev)
