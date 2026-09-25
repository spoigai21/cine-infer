"""Phase 7: the recommendation API.

GET /health                     liveness + whether the model bundle is loaded
GET /recommend/{user_id}?k=10   top-k movies: two-stage for known users with history, the
                                popularity fallback otherwise (see src/serving.py)

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

from fastapi import FastAPI, HTTPException, Query

BUNDLE = os.environ.get("CINEINFER_BUNDLE", "models/serving")
state = {"recommender": None, "error": None}


@asynccontextmanager
async def lifespan(app):
    if os.path.exists(os.path.join(BUNDLE, "manifest.json")):
        from src.serving import Recommender
        try:
            state["recommender"] = Recommender(BUNDLE, threads=int(THREADS))
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
