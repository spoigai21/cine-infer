"""FastAPI service. Phase 0 exposes only /health; /recommend arrives in Phase 7."""
from fastapi import FastAPI

app = FastAPI(title="CineInfer")


@app.get("/health")
def health():
    return {"status": "ok"}
