"""Phase 7: export everything the server needs into one bundle directory (models/serving/).

The server (src/serving.py) must not import torch: PyTorch and LightGBM can't share a process
on macOS (OpenMP), and plain NumPy is simpler to serve anyway. So this PyTorch-side step converts
the two-tower model to NumPy arrays and saves every table the ranker's features need, all from
the SAME training data the served models were trained on (train + val, the data before test):

  two_tower.npz        item embeddings E, user-tower MLP weights (W1, b1, W2, b2), hist_len
  users.npz            userIds, time-ordered positives (offsets/items), seen (CSR, any rating)
  ease_B.npy, ease_keep.npy   EASE item-item weights (tuned config) and its movie columns
  features.npz         item stats, user stats, user genre shares, movie genres, genome rows
  popular.npy          movies by train + val positives (cold-start fallback)
  movies.json          movieId + title per catalog index
  ranker.txt           the LightGBM ranker used on test (trained on val labels, headline features)
  manifest.json        provenance: source models, configs, feature order, counts

Usage: `make export` (seed 42's models).
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src import baselines as bl
from src import evaluate as ev
from src import ranker as rk
from src import two_tower as tt

SPLITS = Path("data/splits.parquet")
MOVIES = Path("data/ml-25m/movies.csv")
FEATURES_DIR = Path("data/features")
GENOME = FEATURES_DIR / "genome.parquet"
OUT = Path("models/serving")
TV_FINE = ("train_core", "train_tail", "val")


def export_bundle(out, retriever_path, ranker_path, ease_cfg, headline, splits=SPLITS,
                  movies=MOVIES, features_dir=FEATURES_DIR, genome=GENOME, seed=42):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    items = ev.ItemIndex.from_movies_csv(movies)
    seq = tt.load_sequences(splits, items, ("train", "val"))
    fine = ev.load_ratings(splits, "user", fine=True)
    retriever = tt.load_scorer(retriever_path, seq)
    coarse = ev.load_ratings(splits, "user")
    td = bl.build_train_data(coarse, items, ("train", "val"))
    ease = bl.EASE(td, bl.ItemGram(td, ease_cfg["min_pos"]), ease_cfg["lam"])
    content = rk.ItemContent.load(movies, genome, items)
    ctx = rk.TrainingSetContext.build("train_val", retriever, ease, seq, fine, TV_FINE,
                                      features_dir, items, content)

    # two-tower -> numpy (weights on CPU; see two_tower.TwoTower for the maths)
    sd = {k: v.detach().cpu().numpy() for k, v in retriever.model.state_dict().items()}
    np.savez(out / "two_tower.npz", E=sd["item.weight"].astype(np.float32),
             W1=sd["mlp.0.weight"], b1=sd["mlp.0.bias"], W2=sd["mlp.2.weight"], b2=sd["mlp.2.bias"],
             hist_len=np.int64(retriever.hist_len), trained=seq.trained_items())
    seen = ctx.seen.tocsr()
    np.savez(out / "users.npz", user_ids=seq.user_ids, offsets=seq.offsets,
             items=seq.items.astype(np.int32), seen_indptr=seen.indptr.astype(np.int64),
             seen_indices=seen.indices.astype(np.int32))
    np.save(out / "ease_B.npy", ease.B.astype(np.float32))
    np.save(out / "ease_keep.npy", ease.keep)
    np.savez(out / "features.npz", item_stats=ctx.item_stats, user_stats=ctx.user_stats,
             user_genres=ctx.user_genres, genres_indptr=content.genres.indptr,
             genres_indices=content.genres.indices, n_genres=np.int64(content.genres.shape[1]),
             genome_rows=content.genome_rows, genome_row=content.genome_row,
             has_genome=content.has_genome)
    np.save(out / "popular.npy", np.lexsort((np.arange(len(items)), -td.item_pos)))
    titles = pd.read_csv(movies).set_index("movieId").title
    (out / "movies.json").write_text(json.dumps(
        {"movie_ids": items.movie_ids.tolist(), "titles": titles.reindex(items.movie_ids).tolist()}))
    shutil.copy(ranker_path, out / "ranker.txt")
    manifest = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": seed,
                "retriever": str(retriever_path), "retriever_config": retriever.config,
                "ranker": str(ranker_path), "ease_config": ease_cfg, "features": list(headline),
                "k_candidates": rk.K_CANDIDATES, "recent_profile": rk.RECENT_PROFILE,
                "training_data": "train + val (per-user split)", "n_items": len(items),
                "n_users": int(len(seq.user_ids))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args()
    from src.tune_ranker import HEADLINE
    b = pd.read_csv("results/baselines.csv")
    ease_cfg = json.loads(b[b.model == "ease"].iloc[0].config)
    t = time.time()
    m = export_bundle(a.out, Path(f"models/two_tower_tv_seed{a.seed}.pt"),
                      Path(f"models/test_two_stage_seed{a.seed}.txt"), ease_cfg, HEADLINE,
                      seed=a.seed)
    size = sum(f.stat().st_size for f in a.out.iterdir()) / 1e6
    print(f"exported {a.out} ({size:.0f} MB, {time.time() - t:.0f}s): {m['n_users']} users, "
          f"{m['n_items']} movies, features {m['features']}")


if __name__ == "__main__":
    main()
