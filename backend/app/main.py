"""
RentShield Germany - FastAPI Backend
Main application entry point with REST API endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from .schemas import (
    ListingInput,
    PricePrediction,
    ScamPrediction,
    FullPrediction,
    HealthResponse,
    BulkListingInput,
    BulkPredictionResponse,
    RiskLevel
)
from .core.pricing import PricingPredictor
from .core.scam import ScamDetector


# Global model instances
pricing_predictor: PricingPredictor = None
scam_detector: ScamDetector = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Loads ML models on startup.
    """
    global pricing_predictor, scam_detector
    
    print("=" * 50)
    print("🏠 RentShield Germany - Starting up...")
    print("=" * 50)
    
    # Initialize predictors
    artifacts_path = Path(__file__).parent / "artifacts"
    
    # Load pricing model
    pricing_predictor = PricingPredictor(artifacts_path)
    pricing_loaded = pricing_predictor.load_model()
    
    # Load scam detection models
    scam_detector = ScamDetector(artifacts_path)
    scam_loaded = scam_detector.load_models()
    
    if not pricing_loaded:
        print("⚠ Pricing model not found - using heuristic fallback")
    if not scam_loaded:
        print("⚠ Scam models not found - using rule-based fallback")
    
    print("=" * 50)
    print("✅ RentShield API ready!")
    print("=" * 50)
    
    yield
    
    # Cleanup on shutdown
    print("👋 RentShield shutting down...")


# Create FastAPI app
app = FastAPI(
    title="RentShield Germany API",
    description="""
    🏠 **RentShield Germany** - Protect yourself from overpriced and scam rental listings.
    
    This API provides:
    - **Fair rent estimation** based on German market data
    - **Overpricing detection** to identify inflated prices
    - **Scam risk scoring** using ML and rule-based analysis
    - **Explainable outputs** for transparent decision-making
    
    ## How it works
    
    1. Submit a rental listing with city, postcode, size, rooms, rent, and description
    2. Our ML models analyze the listing against known patterns
    3. Receive a risk assessment with predicted fair rent and scam indicators
    
    ## Risk Levels
    
    - **LOW**: Listing appears legitimate
    - **MEDIUM**: Some concerns, verify before proceeding
    - **HIGH**: Multiple red flags detected
    - **CRITICAL**: Strong scam indicators, avoid this listing
    """,
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Streamlit URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    """Root endpoint redirect to docs."""
    return {
        "message": "Welcome to RentShield Germany API",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint.
    Returns system status and whether ML models are loaded.
    """
    models_loaded = (
        pricing_predictor is not None and 
        pricing_predictor.model_loaded and
        scam_detector is not None
    )
    
    return HealthResponse(
        status="healthy",
        models_loaded=models_loaded,
        version="1.0.0"
    )


@app.post("/predict/price", response_model=PricePrediction, tags=["Predictions"])
async def predict_price(listing: ListingInput):
    """
    Predict fair rent for a listing.
    
    Returns the predicted fair rent based on location, size, and other features,
    along with overpricing analysis.
    """
    if pricing_predictor is None:
        raise HTTPException(status_code=503, detail="Pricing model not initialized")
    
    result = pricing_predictor.predict(
        city=listing.city,
        postcode=listing.postcode,
        living_space=listing.living_space,
        rooms=listing.rooms,
        rent=listing.rent,
        year_built=listing.year_built,
        floor=listing.floor
    )
    
    return PricePrediction(**result)


@app.post("/predict/scam", response_model=ScamPrediction, tags=["Predictions"])
async def predict_scam(listing: ListingInput):
    """
    Analyze scam risk for a listing.
    
    Returns scam probability and detected red flags based on
    description analysis, price anomalies, and rule-based checks.
    """
    if scam_detector is None:
        raise HTTPException(status_code=503, detail="Scam detector not initialized")
    
    # Get predicted rent for anomaly detection
    predicted_rent = pricing_predictor.predict_rent_per_sqm(
        city=listing.city,
        postcode=listing.postcode,
        living_space=listing.living_space,
        rooms=listing.rooms,
        year_built=listing.year_built,
        floor=listing.floor
    ) * listing.living_space
    
    result = scam_detector.analyze(
        description=listing.description or "",
        rent=listing.rent,
        living_space=listing.living_space,
        predicted_rent=predicted_rent
    )
    
    return ScamPrediction(
        scam_score=result["scam_score"],
        nlp_score=result["nlp_score"],
        anomaly_score=result["anomaly_score"],
        rules_score=result["rules_score"],
        risk_level=RiskLevel(result["risk_level"]),
        reasons=result["reasons"]
    )


@app.post("/predict/full", response_model=FullPrediction, tags=["Predictions"])
async def predict_full(listing: ListingInput):
    """
    Complete listing analysis combining price and scam prediction.
    
    This is the recommended endpoint for full risk assessment.
    Returns:
    - Fair rent prediction with overpricing percentage
    - Scam risk score with component breakdowns
    - Risk level classification
    - Detected red flags and reasons
    """
    if pricing_predictor is None or scam_detector is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    # Price prediction
    price_result, price_reasons = pricing_predictor.get_price_analysis(
        city=listing.city,
        postcode=listing.postcode,
        living_space=listing.living_space,
        rooms=listing.rooms,
        rent=listing.rent,
        year_built=listing.year_built,
        floor=listing.floor
    )
    
    # Scam analysis
    scam_result = scam_detector.analyze(
        description=listing.description or "",
        rent=listing.rent,
        living_space=listing.living_space,
        predicted_rent=price_result["predicted_rent"]
    )
    
    # Combine reasons
    all_reasons = price_reasons + scam_result["reasons"]
    unique_reasons = list(dict.fromkeys(all_reasons))[:6]
    
    return FullPrediction(
        # Price analysis
        predicted_rent=price_result["predicted_rent"],
        predicted_rent_per_sqm=price_result["predicted_rent_per_sqm"],
        actual_rent=price_result["actual_rent"],
        actual_rent_per_sqm=price_result["actual_rent_per_sqm"],
        overpricing_pct=price_result["overpricing_pct"],
        is_overpriced=price_result["is_overpriced"],
        # Scam analysis
        scam_score=scam_result["scam_score"],
        risk_level=RiskLevel(scam_result["risk_level"]),
        reasons=unique_reasons,
        # Component scores
        nlp_score=scam_result["nlp_score"],
        anomaly_score=scam_result["anomaly_score"],
        rules_score=scam_result["rules_score"]
    )


@app.post("/predict/bulk", response_model=BulkPredictionResponse, tags=["Predictions"])
async def predict_bulk(bulk_input: BulkListingInput):
    """
    Analyze multiple listings at once.
    
    Useful for bulk CSV processing. Returns predictions for each listing
    along with summary statistics.
    """
    if pricing_predictor is None or scam_detector is None:
        raise HTTPException(status_code=503, detail="Models not initialized")
    
    predictions = []
    high_risk_count = 0
    
    for listing in bulk_input.listings:
        # Get full prediction for each listing
        price_result, price_reasons = pricing_predictor.get_price_analysis(
            city=listing.city,
            postcode=listing.postcode,
            living_space=listing.living_space,
            rooms=listing.rooms,
            rent=listing.rent,
            year_built=listing.year_built,
            floor=listing.floor
        )
        
        scam_result = scam_detector.analyze(
            description=listing.description or "",
            rent=listing.rent,
            living_space=listing.living_space,
            predicted_rent=price_result["predicted_rent"]
        )
        
        all_reasons = price_reasons + scam_result["reasons"]
        unique_reasons = list(dict.fromkeys(all_reasons))[:6]
        
        pred = FullPrediction(
            predicted_rent=price_result["predicted_rent"],
            predicted_rent_per_sqm=price_result["predicted_rent_per_sqm"],
            actual_rent=price_result["actual_rent"],
            actual_rent_per_sqm=price_result["actual_rent_per_sqm"],
            overpricing_pct=price_result["overpricing_pct"],
            is_overpriced=price_result["is_overpriced"],
            scam_score=scam_result["scam_score"],
            risk_level=RiskLevel(scam_result["risk_level"]),
            reasons=unique_reasons,
            nlp_score=scam_result["nlp_score"],
            anomaly_score=scam_result["anomaly_score"],
            rules_score=scam_result["rules_score"]
        )
        
        predictions.append(pred)
        
        if pred.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            high_risk_count += 1
    
    return BulkPredictionResponse(
        predictions=predictions,
        total_processed=len(predictions),
        high_risk_count=high_risk_count
    )


# Run with: uvicorn backend.app.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host="0.0.0.0", port=8000, reload=True)
