"""Phase 7: the recommendation API.

GET /health                     liveness + whether the model bundle is loaded
GET /recommend/{user_id}?k=10   top-k movies: two-stage for known users with history, the
                                popularity fallback otherwise (see src/serving.py)

The bundle (src/export_serving.py, `make export`) is loaded once at startup from
$CINEINFER_BUNDLE (default models/serving). This process never imports torch.
Run: `make serve` (uvicorn on port 8000).
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

BUNDLE = os.environ.get("CINEINFER_BUNDLE", "models/serving")
state = {"recommender": None, "error": None}


@asynccontextmanager
async def lifespan(app):
    if os.path.exists(os.path.join(BUNDLE, "manifest.json")):
        from src.serving import Recommender
        try:
            state["recommender"] = Recommender(BUNDLE)
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
