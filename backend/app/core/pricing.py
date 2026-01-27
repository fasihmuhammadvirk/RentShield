"""
Pricing prediction module for RentShield.
Handles rent prediction and overpricing detection.
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from .preprocess import preprocess_features, estimate_fair_rent, FeaturePreprocessor


class PricingPredictor:
    """
    Pricing predictor for German rental listings.
    Uses trained ML model or falls back to heuristics.
    """
    
    def __init__(self, artifacts_path: Optional[Path] = None):
        """
        Initialize the pricing predictor.
        
        Args:
            artifacts_path: Path to model artifacts directory
        """
        self.artifacts_path = artifacts_path or Path(__file__).parent.parent / "artifacts"
        self.model = None
        self.preprocessor = None
        self.model_loaded = False
        
        # Overpricing threshold (15% above predicted = overpriced)
        self.overpricing_threshold = 0.15
    
    def load_model(self) -> bool:
        """
        Load the trained pricing model and preprocessor.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        model_path = self.artifacts_path / "rent_model.joblib"
        preprocessor_path = self.artifacts_path / "rent_preprocessor.joblib"
        
        try:
            if model_path.exists():
                self.model = joblib.load(model_path)
                print(f"✓ Loaded pricing model from {model_path}")
            
            if preprocessor_path.exists():
                self.preprocessor = joblib.load(preprocessor_path)
                print(f"✓ Loaded preprocessor from {preprocessor_path}")
            
            self.model_loaded = self.model is not None
            return self.model_loaded
        except Exception as e:
            print(f"⚠ Could not load pricing model: {e}")
            self.model_loaded = False
            return False
    
    def predict_rent_per_sqm(
        self,
        city: str,
        postcode: str,
        living_space: float,
        rooms: float,
        year_built: Optional[int] = None,
        floor: Optional[int] = None
    ) -> float:
        """
        Predict fair rent per square meter.
        
        Returns:
            Predicted rent per sqm in euros
        """
        if self.model is not None:
            # Use trained model
            features = preprocess_features(
                city=city,
                postcode=postcode,
                living_space=living_space,
                rooms=rooms,
                year_built=year_built,
                floor=floor
            )
            # Model predicts rent per sqm
            rent_per_sqm = float(self.model.predict(features)[0])
            return max(rent_per_sqm, 5.0)  # Minimum 5€/sqm
        else:
            # Fallback to heuristic estimation
            total_rent = estimate_fair_rent(
                city=city,
                postcode=postcode,
                living_space=living_space,
                rooms=rooms,
                year_built=year_built,
                floor=floor
            )
            return total_rent / living_space if living_space > 0 else 10.0
    
    def predict(
        self,
        city: str,
        postcode: str,
        living_space: float,
        rooms: float,
        rent: float,
        year_built: Optional[int] = None,
        floor: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Full price prediction with overpricing analysis.
        
        Returns:
            Dictionary with prediction results
        """
        # Predict fair rent per sqm
        predicted_rent_per_sqm = self.predict_rent_per_sqm(
            city=city,
            postcode=postcode,
            living_space=living_space,
            rooms=rooms,
            year_built=year_built,
            floor=floor
        )
        
        # Calculate total predicted rent
        predicted_rent = predicted_rent_per_sqm * living_space
        
        # Calculate actual rent per sqm
        actual_rent_per_sqm = rent / living_space if living_space > 0 else 0
        
        # Calculate overpricing percentage
        if predicted_rent > 0:
            overpricing_pct = (rent - predicted_rent) / predicted_rent
        else:
            overpricing_pct = 0.0
        
        # Determine if significantly overpriced
        is_overpriced = overpricing_pct > self.overpricing_threshold
        
        return {
            "predicted_rent": round(predicted_rent, 2),
            "predicted_rent_per_sqm": round(predicted_rent_per_sqm, 2),
            "actual_rent": rent,
            "actual_rent_per_sqm": round(actual_rent_per_sqm, 2),
            "overpricing_pct": round(overpricing_pct, 3),
            "is_overpriced": is_overpriced
        }
    
    def get_price_analysis(
        self,
        city: str,
        postcode: str,
        living_space: float,
        rooms: float,
        rent: float,
        year_built: Optional[int] = None,
        floor: Optional[int] = None
    ) -> Tuple[Dict[str, Any], list]:
        """
        Get price analysis with reasons.
        
        Returns:
            Tuple of (prediction_dict, reasons_list)
        """
        prediction = self.predict(
            city=city,
            postcode=postcode,
            living_space=living_space,
            rooms=rooms,
            rent=rent,
            year_built=year_built,
            floor=floor
        )
        
        reasons = []
        
        if prediction["overpricing_pct"] > 0.30:
            reasons.append(f"significantly overpriced ({prediction['overpricing_pct']*100:.0f}% above market)")
        elif prediction["overpricing_pct"] > 0.15:
            reasons.append(f"moderately overpriced ({prediction['overpricing_pct']*100:.0f}% above market)")
        
        # Check for extreme rent per sqm
        if prediction["actual_rent_per_sqm"] > 25:
            reasons.append("rent per sqm is very high (>25€)")
        elif prediction["actual_rent_per_sqm"] > 20:
            reasons.append("rent per sqm is above average (>20€)")
        
        # Check for suspiciously low rent
        if prediction["overpricing_pct"] < -0.30:
            reasons.append("rent is suspiciously low - could be a scam")
        
        return prediction, reasons
