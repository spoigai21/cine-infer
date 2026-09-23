"""Generate the synthetic test fixture in tests/fixtures/.

The fixture is synthetic, NOT a MovieLens sample (the license forbids redistributing MovieLens
data), but it keeps the same schema and deliberately includes the edge cases later phases must
handle:

- users with exactly 20 ratings (the MovieLens minimum) and a few heavier users
- a user with only 8 ratings (fewer than 10, so Phase 1 routes them to train only)
- rating bursts: many ratings sharing one timestamp (tests the movieId tiebreak)
- a user whose most recent ratings are all below 4 (no relevant items in val/test)
- users who start rating late in the timeline (cold users under the global-cutoff split)
- a movie with no ratings at all, and titles containing commas and quotes

Deterministic: re-running produces byte-identical files.
"""
import csv
from pathlib import Path

import numpy as np

SEED = 42
OUT = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
N_MOVIES = 40
DAY = 86_400
T0 = 1_000_000_000  # 2001-09-09

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]
SPECIAL_TITLES = {
    1: "Amityville: A New Generation, The (1993)",
    2: 'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb (1964)',
    3: 'Crouching Tiger, Hidden Dragon (Wo hu cang long) (2000)',
    4: '"Great Performances" Cats (1998)',
}
UNRATED_MOVIE = N_MOVIES  # stays in movies.csv, never rated


def main():
    rng = np.random.default_rng(SEED)
    rows = []

    def add_user(uid, n, start, spacing, burst_at=None, low_tail=0):
        movies = rng.choice(np.arange(1, N_MOVIES), size=n, replace=False)
        ts = start + np.cumsum(rng.integers(1, spacing, size=n))
        if burst_at is not None:  # a block of ratings sharing one timestamp
            ts[burst_at:burst_at + 6] = ts[burst_at]
            ts = np.maximum.accumulate(ts)
        ratings = rng.choice(np.arange(1, 11) / 2, size=n,
                             p=[.02, .03, .05, .07, .10, .15, .20, .18, .12, .08])
        if low_tail:
            ratings[-low_tail:] = rng.choice([1.0, 2.0, 2.5, 3.0, 3.5], size=low_tail)
        rows.extend(zip([uid] * n, movies.tolist(), ratings.tolist(), ts.tolist()))

    add_user(1, 20, T0, 30 * DAY)
    add_user(2, 20, T0 + 100 * DAY, 20 * DAY, burst_at=10)
    add_user(3, 30, T0 + 50 * DAY, 25 * DAY)
    add_user(4, 25, T0, 40 * DAY, low_tail=6)          # no positives in val/test
    add_user(5, 20, T0 + 900 * DAY, 3 * DAY)           # late starter
    add_user(6, 22, T0 + 1000 * DAY, 2 * DAY, burst_at=14)
    add_user(7, 8, T0 + 200 * DAY, 30 * DAY)           # fewer than 10 ratings
    add_user(8, 30, T0 + 10 * DAY, 30 * DAY, burst_at=0)
    add_user(9, 25, T0 + 300 * DAY, 15 * DAY)

    rows.sort(key=lambda r: (r[0], r[3], r[1]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tiny_ratings.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["userId", "movieId", "rating", "timestamp"])
        w.writerows((u, m, r, t) for u, m, r, t in rows)

    with open(OUT / "tiny_movies.csv", "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")  # QUOTE_MINIMAL, like MovieLens
        w.writerow(["movieId", "title", "genres"])
        for m in range(1, N_MOVIES + 1):
            title = SPECIAL_TITLES.get(m, f"Synthetic Movie {m} ({1980 + m})")
            k = int(rng.integers(1, 4))
            genres = "|".join(sorted(rng.choice(GENRES, size=k, replace=False)))
            w.writerow([m, title, genres if m != UNRATED_MOVIE else "(no genres listed)"])

    print(f"wrote {len(rows)} ratings and {N_MOVIES} movies to {OUT}")


if __name__ == "__main__":
    main()
