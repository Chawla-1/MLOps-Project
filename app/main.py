"""FastAPI prediction service for MP141 PG Room Rent (Patna)."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import time
from typing import Any

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd

from app.schemas import HealthResponse, PredictionRequest, PredictionResponse
from src.logger import get_logger

DEFAULT_MODEL_PATH = Path("models/model.joblib")
FEATURE_COLUMNS = [
    "distance_to_campus_km",
    "room_area_sqft",
    "sharing",
    "meals_per_day",
    "building_age_yr",
    "wifi_mbps",
    "furnishing",
    "locality",
]

logger = get_logger("mp141_service")


def _load_model_from_disk(model_path: Path) -> Any:
    """Load the serialized scikit-learn pipeline from disk."""
    if not model_path.exists():
        return None
    return joblib.load(model_path)


def get_model(fastapi_app: FastAPI) -> Any:
    """Return the cached model or load it on demand from MODEL_PATH."""
    model = getattr(fastapi_app.state, "model", None)
    if model is not None:
        return model

    model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
    model = _load_model_from_disk(model_path)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model artifact not found at '{model_path}'.",
        )

    fastapi_app.state.model = model
    return model


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    """Preload the model artifact on startup when available."""
    if getattr(fastapi_app.state, "model", None) is None:
        model_path = Path(os.getenv("MODEL_PATH", str(DEFAULT_MODEL_PATH)))
        model = _load_model_from_disk(model_path)
        fastapi_app.state.model = model
        if model is None:
            logger.warning(
                "Model artifact not found at startup; /predict will return 503 until available.",
                extra={"extra_info": {"model_path": str(model_path)}},
            )
        else:
            logger.info(
                "Model artifact loaded successfully.",
                extra={"extra_info": {"model_path": str(model_path)}},
            )
    yield


app = FastAPI(
    title="MP141 PG Room Rent Prediction API",
    description="Predicts fair monthly PG room rent in Patna.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Health check endpoint for Kubernetes readiness and liveness probes."""
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Predict monthly PG room rent (rent_rs) from validated room features."""
    start_time = time.perf_counter()
    model = get_model(app)

    features = pd.DataFrame([payload.model_dump()], columns=FEATURE_COLUMNS)
    raw_pred = model.predict(features)[0]
    predicted_rent = round(float(raw_pred), 2)

    latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
    logger.info(
        "prediction generated",
        extra={
            "extra_info": {
                "prediction": predicted_rent,
                "latency_ms": latency_ms,
                "locality": payload.locality,
            }
        },
    )
    return PredictionResponse(prediction=predicted_rent)
