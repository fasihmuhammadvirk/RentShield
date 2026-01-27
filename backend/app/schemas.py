"""
Pydantic schemas for RentShield API request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification for rental listings."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ListingInput(BaseModel):
    """Input schema for rental listing analysis."""
    
    city: str = Field(..., description="City name (e.g., Berlin, Munich)")
    postcode: str = Field(..., description="German postcode (e.g., 10115)")
    living_space: float = Field(..., gt=0, description="Living space in square meters")
    rooms: float = Field(..., gt=0, description="Number of rooms")
    rent: float = Field(..., gt=0, description="Monthly rent in euros (cold rent)")
    description: Optional[str] = Field(default="", description="Listing description text")
    year_built: Optional[int] = Field(default=None, ge=1800, le=2030, description="Year the building was constructed")
    floor: Optional[int] = Field(default=None, ge=0, le=50, description="Floor number")
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "city": "Berlin",
                    "postcode": "10115",
                    "living_space": 55.0,
                    "rooms": 2.0,
                    "rent": 950.0,
                    "description": "Beautiful apartment in Mitte, deposit required before viewing",
                    "year_built": 1990,
                    "floor": 3
                }
            ]
        }
    }


class PricePrediction(BaseModel):
    """Response schema for price prediction."""
    
    predicted_rent: float = Field(..., description="Predicted monthly rent in euros")
    predicted_rent_per_sqm: float = Field(..., description="Predicted rent per square meter")
    actual_rent: float = Field(..., description="Actual listed rent")
    actual_rent_per_sqm: float = Field(..., description="Actual rent per square meter")
    overpricing_pct: float = Field(..., description="Overpricing percentage (positive = overpriced)")
    is_overpriced: bool = Field(..., description="Whether the listing is significantly overpriced")


class ScamPrediction(BaseModel):
    """Response schema for scam risk prediction."""
    
    scam_score: float = Field(..., ge=0, le=1, description="Overall scam probability (0-1)")
    nlp_score: float = Field(..., ge=0, le=1, description="NLP-based scam probability")
    anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly detection score")
    rules_score: float = Field(..., ge=0, le=1, description="Rule-based red flag score")
    risk_level: RiskLevel = Field(..., description="Risk level classification")
    reasons: List[str] = Field(default_factory=list, description="Detected red flags and reasons")


class FullPrediction(BaseModel):
    """Combined response for full listing analysis."""
    
    # Price analysis
    predicted_rent: float = Field(..., description="Predicted fair monthly rent")
    predicted_rent_per_sqm: float = Field(..., description="Predicted fair rent per sqm")
    actual_rent: float = Field(..., description="Listed rent")
    actual_rent_per_sqm: float = Field(..., description="Listed rent per sqm")
    overpricing_pct: float = Field(..., description="Overpricing percentage")
    is_overpriced: bool = Field(..., description="Significantly overpriced flag")
    
    # Scam analysis
    scam_score: float = Field(..., ge=0, le=1, description="Overall scam score")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    reasons: List[str] = Field(default_factory=list, description="Detected issues")
    
    # Component scores (for transparency)
    nlp_score: float = Field(..., description="NLP component score")
    anomaly_score: float = Field(..., description="Anomaly component score")
    rules_score: float = Field(..., description="Rules component score")


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(default="healthy")
    models_loaded: bool = Field(..., description="Whether ML models are loaded")
    version: str = Field(default="1.0.0")


class BulkListingInput(BaseModel):
    """Input schema for bulk listing analysis."""
    
    listings: List[ListingInput] = Field(..., description="List of listings to analyze")


class BulkPredictionResponse(BaseModel):
    """Response schema for bulk analysis."""
    
    predictions: List[FullPrediction] = Field(..., description="Predictions for each listing")
    total_processed: int = Field(..., description="Number of listings processed")
    high_risk_count: int = Field(..., description="Number of high/critical risk listings")
