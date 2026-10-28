import time
from typing import Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from schemas import PredictResponse
from utils import read_image_to_numpy, brightness_mean_u8, variance_of_laplacian
from model import get_model

app = FastAPI(title="AgroDry QFM-CNN API", version="0.2.0")

# CORS (tighten for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your domain(s) in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "ts": time.time()}

@app.get("/version")
def version() -> dict[str, str]:
    m = get_model()
    return {"api": app.version, "model_backend": m.cfg.backend, "model_version": m.cfg.version}

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> Any:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    raw = await file.read()
    pil, arr = read_image_to_numpy(raw)

    # --- Quality checks (robust even if OpenCV is missing) ---
    notes = []
    light_val = None
    sharp_val = None
    try:
        light_val = brightness_mean_u8(arr)  # 0..1
        notes.append(f"brightness={light_val:.2f}")
    except Exception:
        notes.append("brightness=n/a (opencv missing)")

    try:
        sharp_val = variance_of_laplacian(arr)  # float
        notes.append(f"sharpness={sharp_val:.0f}")
    except Exception:
        notes.append("sharpness=n/a (opencv missing)")

    light_ok = (light_val is not None) and (light_val > 0.22)
    sharp_ok = (sharp_val is not None) and (sharp_val > 50)

    # --- Inference (moisture only) ---
    model_out = get_model().predict(arr)  # expects dict with keys: moisture, model, latency_ms

    resp = {
        **model_out,
        "quality": {
            "lighting_ok": bool(light_ok),
            "sharpness_ok": bool(sharp_ok),
            "notes": notes,
        },
    }
    return JSONResponse(resp)

# (OPTIONAL) serve your frontend from the same origin:
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "AgroDry_frontend.html")
