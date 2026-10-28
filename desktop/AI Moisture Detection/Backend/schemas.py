from pydantic import BaseModel
from typing import Optional, List

class Moisture(BaseModel):
    value_pct: float
    confidence: Optional[float] = None

class Quality(BaseModel):
    lighting_ok: bool
    sharpness_ok: bool
    notes: List[str] = []

class ModelInfo(BaseModel):
    version: str
    backend: str

class PredictResponse(BaseModel):
    moisture: Moisture
    quality: Quality
    model: ModelInfo
    latency_ms: int
