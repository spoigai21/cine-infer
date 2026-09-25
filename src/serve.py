"""Phase 7: the recommendation API.

GET /                           the demo page (src/static/demo.html): pick a user, see their
                                recent liked movies and live recommendations with timings
GET /health                     liveness + whether the model bundle is loaded
GET /recommend/{user_id}?k=10   top-k movies: two-stage for known users with history, the
                                popularity fallback otherwise (see src/serving.py)
GET /users/random               a random known user id (demo page)
GET /users/{user_id}/profile    that user's most recent liked movies (demo page)

The bundle (src/export_serving.py, `make export`) is loaded once at startup from
$CINEINFER_BUNDLE (default models/serving). This process never imports torch.

Threads: $CINEINFER_THREADS=N caps BLAS / OpenMP / LightGBM threads at N (set before NumPy is
imported, hence the top of this module); 0 leaves the libraries' own threading alone. Default 1:
a request is a few small single-user operations, and the libraries' thread pools only contend.
Measured after prediction #6 was settled (results/latency_threads.csv, 4 alternating runs):
default threading p50 5.2-6.8 ms / p99 25-60 ms, and it pushed the load average from 2 to 17 by
itself; single-threaded p50 3.2 ms / p99 6.8-19 ms.
Run: `make serve` (uvicorn on port 8000).
"""
import os

THREADS = os.environ.get("CINEINFER_THREADS", "1")
if THREADS != "0":
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, THREADS)
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse

BUNDLE = os.environ.get("CINEINFER_BUNDLE", "models/serving")
state = {"recommender": None, "error": None}


@asynccontextmanager
async def lifespan(app):
    if os.path.exists(os.path.join(BUNDLE, "manifest.json")):
        from src.serving import Recommender
        try:
            rec = Recommender(BUNDLE, threads=int(THREADS))
            # Warm-up before taking traffic: the first requests after loading are ~10x slower
            # while the embedding table and feature arrays get paged in (the load test discards
            # 50 warm-up requests for the same reason). 50 random users, plus the fallback path.
            import numpy as np
            rng = np.random.default_rng(0)
            for _ in range(int(os.environ.get("CINEINFER_WARMUP", "50"))):
                rec.recommend(rec.sample_user(rng))
            rec.recommend(-1)
            state["recommender"] = rec
        except Exception as e:  # keep /health up and say why
            state["error"] = f"{type(e).__name__}: {e}"
    else:
        state["error"] = f"no model bundle at {BUNDLE} (run `make export`)"
    yield
    state["recommender"] = None


app = FastAPI(title="CineInfer", lifespan=lifespan)


@app.get("/health")
def health():
    rec = state["recommender"]
    return {"status": "ok", "model_loaded": rec is not None,
            **({"model": {k: rec.manifest[k] for k in ("seed", "created", "training_data")}}
               if rec else {"detail": state["error"]})}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=100)):
    rec = state["recommender"]
    if rec is None:
        raise HTTPException(503, state["error"] or "model not loaded")
    return rec.recommend(user_id, k)


def _rec():
    rec = state["recommender"]
    if rec is None:
        raise HTTPException(503, state["error"] or "model not loaded")
    return rec


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def demo_page():
    return (Path(__file__).parent / "static" / "demo.html").read_text()


@app.get("/users/random")
def random_user():
    return {"user_id": _rec().sample_user()}


@app.get("/users/{user_id}/profile")
def profile(user_id: int, n: int = Query(10, ge=1, le=50)):
    return _rec().profile(user_id, n)
