"""
Scam detection module for RentShield.
Combines NLP, anomaly detection, and rule-based approaches.
"""

import numpy as np
import re
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


# Red flag keywords and phrases (English and German)
SCAM_KEYWORDS = {
    # Payment/deposit red flags
    "high": [
        "western union", "wire transfer", "bank transfer before",
        "deposit before viewing", "kaution vorab", "überweisung vor besichtigung",
        "payment upfront", "zahlung im voraus", "send money",
        "cannot meet", "kann nicht treffen", "out of country",
        "im ausland", "abroad", "urgent", "dringend",
        "lottery winner", "inheritance", "too good to be true"
    ],
    # Moderate red flags
    "medium": [
        "no viewing possible", "keine besichtigung möglich",
        "keys will be sent", "schlüssel werden geschickt",
        "agent fee", "provision", "temporary", "befristet",
        "quick decision", "schnelle entscheidung",
        "many interested", "viele interessenten",
        "first come first serve", "wer zuerst kommt"
    ],
    # Low-level flags (context dependent)
    "low": [
        "renovated", "renoviert", "as is", "wie gesehen",
        "bills included", "nebenkosten inklusive",
        "furnished", "möbliert"
    ]
}

# Suspicious patterns in descriptions
SUSPICIOUS_PATTERNS = [
    r"send.*money.*before",
    r"western\s*union",
    r"wire\s*transfer",
    r"deposit.*(?:before|prior).*viewing",
    r"kaution.*vor.*besichtigung",
    r"(?:i|we)\s*(?:am|are)\s*(?:abroad|overseas|away)",
    r"(?:ich|wir)\s*(?:bin|sind)\s*im\s*ausland",
    r"keys?\s*(?:will|can)\s*be\s*(?:sent|mailed)",
    r"schlüssel.*(?:geschickt|gesendet)",
    r"too\s*good\s*to\s*be\s*true",
    r"urgent.*(?:sale|rent|moving)",
    r"(?:lottery|inheritance|won)",
]


class ScamDetector:
    """
    Scam detection system for rental listings.
    Combines NLP classification, anomaly detection, and rule-based scoring.
    """
    
    def __init__(self, artifacts_path: Optional[Path] = None):
        """
        Initialize the scam detector.
        
        Args:
            artifacts_path: Path to model artifacts directory
        """
        self.artifacts_path = artifacts_path or Path(__file__).parent.parent / "artifacts"
        
        # ML components
        self.tfidf = None
        self.nlp_model = None
        self.anomaly_model = None
        
        # Weights for combining scores
        self.weights = {
            "nlp": 0.5,
            "anomaly": 0.3,
            "rules": 0.2
        }
        
        # Risk thresholds
        self.thresholds = {
            "low": 0.25,
            "medium": 0.50,
            "high": 0.75
        }
        
        self.models_loaded = False
    
    def load_models(self) -> bool:
        """
        Load trained scam detection models.
        
        Returns:
            True if models loaded successfully
        """
        try:
            tfidf_path = self.artifacts_path / "tfidf.joblib"
            nlp_path = self.artifacts_path / "scam_model.joblib"
            anomaly_path = self.artifacts_path / "anomaly_model.joblib"
            thresholds_path = self.artifacts_path / "thresholds.json"
            
            if tfidf_path.exists():
                self.tfidf = joblib.load(tfidf_path)
                print(f"✓ Loaded TF-IDF vectorizer")
            
            if nlp_path.exists():
                self.nlp_model = joblib.load(nlp_path)
                print(f"✓ Loaded NLP scam classifier")
            
            if anomaly_path.exists():
                self.anomaly_model = joblib.load(anomaly_path)
                print(f"✓ Loaded anomaly detector")
            
            if thresholds_path.exists():
                import json
                with open(thresholds_path, 'r') as f:
                    loaded = json.load(f)
                    self.thresholds.update(loaded.get('thresholds', {}))
                    self.weights.update(loaded.get('weights', {}))
            
            self.models_loaded = self.tfidf is not None and self.nlp_model is not None
            return self.models_loaded
        except Exception as e:
            print(f"⚠ Could not load scam models: {e}")
            return False
    
    def analyze_text_rules(self, text: str) -> Tuple[float, List[str]]:
        """
        Rule-based text analysis for scam indicators.
        
        Returns:
            Tuple of (score, list of detected red flags)
        """
        if not text:
            return 0.0, []
        
        text_lower = text.lower()
        detected_flags = []
        score = 0.0
        
        # Check high-risk keywords
        for keyword in SCAM_KEYWORDS["high"]:
            if keyword in text_lower:
                score += 0.25
                detected_flags.append(f"high-risk phrase: '{keyword}'")
        
        # Check medium-risk keywords
        for keyword in SCAM_KEYWORDS["medium"]:
            if keyword in text_lower:
                score += 0.10
                detected_flags.append(f"caution phrase: '{keyword}'")
        
        # Check suspicious patterns
        for pattern in SUSPICIOUS_PATTERNS:
            if re.search(pattern, text_lower):
                score += 0.20
                detected_flags.append("suspicious pattern detected")
                break  # Only count once
        
        # Cap score at 1.0
        return min(score, 1.0), detected_flags
    
    def analyze_text_nlp(self, text: str) -> float:
        """
        NLP-based scam probability using trained model.
        
        Returns:
            Scam probability (0-1)
        """
        if not text or self.tfidf is None or self.nlp_model is None:
            return 0.0
        
        try:
            # Vectorize text
            text_vec = self.tfidf.transform([text])
            
            # Get probability
            proba = self.nlp_model.predict_proba(text_vec)[0]
            # Assuming class 1 is scam
            scam_prob = proba[1] if len(proba) > 1 else proba[0]
            
            return float(scam_prob)
        except Exception as e:
            print(f"NLP analysis error: {e}")
            return 0.0
    
    def detect_price_anomaly(
        self,
        rent: float,
        living_space: float,
        predicted_rent: float
    ) -> Tuple[float, List[str]]:
        """
        Detect price anomalies that might indicate scams.
        
        Returns:
            Tuple of (anomaly_score, reasons)
        """
        reasons = []
        score = 0.0
        
        rent_per_sqm = rent / living_space if living_space > 0 else 0
        predicted_per_sqm = predicted_rent / living_space if living_space > 0 else 0
        
        # Suspiciously low rent (common scam tactic)
        if rent_per_sqm < 5 and predicted_per_sqm > 8:
            score += 0.5
            reasons.append("rent suspiciously below market rate")
        
        # Very low rent for the area
        if predicted_rent > 0 and rent < predicted_rent * 0.5:
            score += 0.4
            reasons.append("rent is less than 50% of market value")
        
        # Extreme values
        if living_space > 200 and rent < 500:
            score += 0.6
            reasons.append("large apartment at unrealistic price")
        
        if living_space < 20 and rent > 1500:
            score += 0.3
            reasons.append("tiny space at premium price - verify legitimacy")
        
        # Use anomaly model if available
        if self.anomaly_model is not None:
            try:
                features = np.array([[rent, living_space, rent_per_sqm]])
                
                # IsolationForest returns -1 for anomalies
                prediction = self.anomaly_model.predict(features)[0]
                anomaly_score = self.anomaly_model.score_samples(features)[0]
                
                if prediction == -1:
                    score += 0.3
                    reasons.append("price anomaly detected by ML model")
                
                # Convert score to 0-1 range (lower is more anomalous)
                normalized = 1 - (anomaly_score + 0.5)  # Rough normalization
                score = max(score, min(normalized, 1.0))
            except Exception:
                pass
        
        return min(score, 1.0), reasons
    
    def get_risk_level(self, scam_score: float) -> str:
        """Convert scam score to risk level."""
        if scam_score >= self.thresholds["high"]:
            return "CRITICAL"
        elif scam_score >= self.thresholds["medium"]:
            return "HIGH"
        elif scam_score >= self.thresholds["low"]:
            return "MEDIUM"
        else:
            return "LOW"
    
    def analyze(
        self,
        description: str,
        rent: float,
        living_space: float,
        predicted_rent: float
    ) -> Dict[str, Any]:
        """
        Full scam analysis combining all methods.
        
        Returns:
            Dictionary with scam analysis results
        """
        all_reasons = []
        
        # 1. Rule-based text analysis
        rules_score, rules_reasons = self.analyze_text_rules(description)
        all_reasons.extend(rules_reasons)
        
        # 2. NLP analysis
        nlp_score = self.analyze_text_nlp(description)
        if nlp_score > 0.5:
            all_reasons.append("suspicious language patterns detected")
        
        # 3. Price anomaly detection
        anomaly_score, anomaly_reasons = self.detect_price_anomaly(
            rent=rent,
            living_space=living_space,
            predicted_rent=predicted_rent
        )
        all_reasons.extend(anomaly_reasons)
        
        # Combine scores with weights
        combined_score = (
            self.weights["nlp"] * nlp_score +
            self.weights["anomaly"] * anomaly_score +
            self.weights["rules"] * rules_score
        )
        
        # Apply boost for multiple red flags
        if len(all_reasons) >= 3:
            combined_score = min(combined_score * 1.2, 1.0)
        
        # Get risk level
        risk_level = self.get_risk_level(combined_score)
        
        # Deduplicate reasons
        unique_reasons = list(dict.fromkeys(all_reasons))
        
        return {
            "scam_score": round(combined_score, 3),
            "nlp_score": round(nlp_score, 3),
            "anomaly_score": round(anomaly_score, 3),
            "rules_score": round(rules_score, 3),
            "risk_level": risk_level,
            "reasons": unique_reasons[:5]  # Top 5 reasons
        }
