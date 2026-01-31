#!/usr/bin/env python3
"""
Retrain models with deployment-compatible paths.
This ensures joblib can unpickle them on Streamlit Cloud.
"""

import sys
from pathlib import Path

# Add project root to path
root = Path(__file__).parent.parent
sys.path.insert(0, str(root))

# Now run the training scripts
print("=" * 60)
print("Retraining models for deployment...")
print("=" * 60)

# Train pricing model
print("\n1. Training pricing model...")
import training.train_price as train_price
train_price.train_pricing_model(
    data_path=root / "data" / "training" / "training_data.csv",
    artifacts_path=root / "backend" / "app" / "artifacts"
)

# Train scam models
print("\n2. Training scam detection models...")
import training.train_scam as train_scam
train_scam.train_scam_models(
    data_path=root / "data" / "training" / "training_data.csv",
    artifacts_path=root / "backend" / "app" / "artifacts"
)

print("\n" + "=" * 60)
print("✅ All models retrained successfully!")
print("=" * 60)
