"""
Feature preprocessing module for RentShield.
Handles encoding, scaling, and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import joblib
from pathlib import Path


# German city tiers based on rental market
CITY_TIERS = {
    # Tier 1: Most expensive cities
    "munich": 1, "münchen": 1,
    "frankfurt": 1, "frankfurt am main": 1,
    "stuttgart": 1,
    
    # Tier 2: Major cities
    "berlin": 2,
    "hamburg": 2,
    "düsseldorf": 2, "dusseldorf": 2,
    "cologne": 2, "köln": 2,
    "mainz": 2,
    
    # Tier 3: Large cities
    "hannover": 3, "hanover": 3,
    "nuremberg": 3, "nürnberg": 3,
    "dresden": 3,
    "leipzig": 3,
    "bonn": 3,
    "mannheim": 3,
    "karlsruhe": 3,
    "freiburg": 3,
    "augsburg": 3,
    "wiesbaden": 3,
    
    # Tier 4: Medium cities
    "essen": 4,
    "dortmund": 4,
    "duisburg": 4,
    "bochum": 4,
    "wuppertal": 4,
    "bielefeld": 4,
    "münster": 4, "munster": 4,
    "kassel": 4,
    "kiel": 4,
    "rostock": 4,
    "erfurt": 4,
    "magdeburg": 4,
    "potsdam": 4,
}

# Average rent per sqm by city tier (baseline as of 2024)
TIER_RENT_BASELINE = {
    1: 20.0,  # Munich, Frankfurt, Stuttgart
    2: 15.0,  # Berlin, Hamburg, etc.
    3: 12.0,  # Hannover, Dresden, etc.
    4: 9.0,   # Essen, Dortmund, etc.
    5: 7.5,   # Other smaller cities
}


def get_city_tier(city: str) -> int:
    """Get city tier based on rental market classification."""
    city_lower = city.lower().strip()
    return CITY_TIERS.get(city_lower, 5)  # Default to tier 5


def get_postcode_multiplier(postcode: str) -> float:
    """
    Get rent multiplier based on postcode.
    Prime postcodes in city centers get higher multipliers.
    """
    # Convert to string to handle both int and str postcodes
    postcode = str(postcode) if postcode else ""
    
    if not postcode or len(postcode) < 2:
        return 1.0
    
    try:
        prefix = int(postcode[:2])
        
        # Berlin prime areas (10xxx)
        if 10 <= prefix <= 14:
            return 1.15
        # Munich (80xxx-81xxx)
        elif 80 <= prefix <= 81:
            return 1.20
        # Hamburg central (20xxx-22xxx)
        elif 20 <= prefix <= 22:
            return 1.12
        # Frankfurt central (60xxx)
        elif prefix == 60:
            return 1.18
        # Cologne central (50xxx)
        elif prefix == 50:
            return 1.10
        else:
            return 1.0
    except (ValueError, TypeError):
        return 1.0


def preprocess_features(
    city: str,
    postcode: str,
    living_space: float,
    rooms: float,
    year_built: Optional[int] = None,
    floor: Optional[int] = None
) -> np.ndarray:
    """
    Preprocess input features for the pricing model.
    
    Returns:
        Feature array for model prediction
    """
    # City tier (1-5)
    city_tier = get_city_tier(city)
    
    # Postcode multiplier
    postcode_mult = get_postcode_multiplier(postcode)
    
    # Room density (rooms per sqm)
    room_density = rooms / living_space if living_space > 0 else 0
    
    # Building age (years since construction)
    current_year = 2026
    building_age = current_year - year_built if year_built else 30  # Default 30 years
    
    # Floor feature (normalized)
    floor_normalized = min(floor / 10, 1.0) if floor is not None else 0.3
    
    # Size category
    if living_space < 40:
        size_category = 1  # Studio/small
    elif living_space < 70:
        size_category = 2  # Normal
    elif living_space < 100:
        size_category = 3  # Large
    else:
        size_category = 4  # Very large
    
    # Feature vector
    features = np.array([
        city_tier,
        postcode_mult,
        living_space,
        rooms,
        room_density,
        building_age,
        floor_normalized,
        size_category
    ]).reshape(1, -1)
    
    return features


def estimate_fair_rent(
    city: str,
    postcode: str,
    living_space: float,
    rooms: float,
    year_built: Optional[int] = None,
    floor: Optional[int] = None
) -> float:
    """
    Estimate fair rent using heuristic rules.
    Used as fallback when model is not available.
    
    Returns:
        Estimated fair monthly rent in euros
    """
    city_tier = get_city_tier(city)
    base_rent_per_sqm = TIER_RENT_BASELINE.get(city_tier, 8.0)
    
    # Apply postcode premium
    postcode_mult = get_postcode_multiplier(postcode)
    rent_per_sqm = base_rent_per_sqm * postcode_mult
    
    # Adjust for building age
    current_year = 2026
    age = current_year - year_built if year_built else 30
    if age < 10:
        rent_per_sqm *= 1.15  # New building premium
    elif age > 50:
        rent_per_sqm *= 0.90  # Old building discount
    
    # Adjust for floor
    if floor is not None:
        if floor == 0:
            rent_per_sqm *= 0.95  # Ground floor discount
        elif floor >= 4:
            rent_per_sqm *= 1.05  # High floor premium (views)
    
    # Size discount for larger apartments
    if living_space > 100:
        rent_per_sqm *= 0.92
    elif living_space > 80:
        rent_per_sqm *= 0.96
    
    return rent_per_sqm * living_space


class FeaturePreprocessor:
    """
    Feature preprocessor for ML models.
    Handles encoding and scaling of input features.
    """
    
    def __init__(self):
        self.fitted = False
        self.feature_names = [
            'city_tier', 'postcode_mult', 'living_space', 'rooms',
            'room_density', 'building_age', 'floor_normalized', 'size_category'
        ]
    
    def fit(self, X: pd.DataFrame) -> 'FeaturePreprocessor':
        """Fit the preprocessor (placeholder for future scaling)."""
        self.fitted = True
        return self
    
    def transform(self, data: Dict[str, Any]) -> np.ndarray:
        """Transform input data to feature array."""
        return preprocess_features(
            city=data.get('city', ''),
            postcode=data.get('postcode', ''),
            living_space=data.get('living_space', 50),
            rooms=data.get('rooms', 2),
            year_built=data.get('year_built'),
            floor=data.get('floor')
        )
    
    def save(self, path: str) -> None:
        """Save preprocessor to disk."""
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str) -> 'FeaturePreprocessor':
        """Load preprocessor from disk."""
        return joblib.load(path)
