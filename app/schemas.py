"""Pydantic schemas for the MP141 PG Room Rent Prediction Service."""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Request payload for POST /predict enforcing client validation rules."""

    model_config = ConfigDict(extra="forbid")

    distance_to_campus_km: float = Field(
        ...,
        ge=0,
        description="Distance from PG room to campus in kilometers.",
    )
    room_area_sqft: float = Field(
        ...,
        gt=0,
        description="Room area in square feet.",
    )
    sharing: int = Field(
        ...,
        ge=1,
        description="Number of students sharing the room.",
    )
    meals_per_day: int = Field(
        ...,
        ge=0,
        description="Number of meals included per day.",
    )
    building_age_yr: float = Field(
        ...,
        ge=0,
        le=75,
        description="Age of the building in years (must be <= 75).",
    )
    wifi_mbps: float = Field(
        ...,
        ge=0,
        description="Offered Wi-Fi speed in Mbps (must be >= 0).",
    )
    furnishing: str = Field(
        ...,
        min_length=1,
        description="Furnishing category (unfurnished, semi, full).",
    )
    locality: str = Field(
        ...,
        min_length=1,
        description="Locality name (Sector East, Sector West, Market Side).",
    )


class PredictionResponse(BaseModel):
    """Response payload for POST /predict."""

    prediction: float = Field(..., description="Predicted monthly rent in rupees.")


class HealthResponse(BaseModel):
    """Response payload for GET /health."""

    status: str = Field(default="ok", description="Service health status.")
