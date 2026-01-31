"""
Train the pricing (rent prediction) model for RentShield.
Uses GradientBoostingRegressor to predict fair rent per square meter.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.app.core.preprocess import FeaturePreprocessor, preprocess_features


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix from dataframe.
    
    Returns:
        X: Feature matrix
        y: Target variable (rent per sqm)
    """
    features_list = []
    
    for _, row in df.iterrows():
        features = preprocess_features(
            city=row['city'],
            postcode=row['postcode'],
            living_space=row['living_space'],
            rooms=row['rooms'],
            year_built=row.get('year_built'),
            floor=row.get('floor')
        )
        features_list.append(features.flatten())
    
    X = np.array(features_list)
    y = df['rent_per_sqm'].values
    
    return X, y


def train_pricing_model(data_path: Path, artifacts_path: Path) -> dict:
    """
    Train the pricing model and save artifacts.
    
    Args:
        data_path: Path to training data CSV
        artifacts_path: Path to save model artifacts
    
    Returns:
        Dictionary with training metrics
    """
    print("=" * 50)
    print("🏠 Training Rent Prediction Model")
    print("=" * 50)
    
    # Load data
    print("\n📊 Loading training data...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Filter out scam listings for price training
    # Scams have unrealistic prices that would skew the model
    df_clean = df[df['scam_label'] == 0].copy()
    print(f"✓ Using {len(df_clean)} legitimate listings for training")
    
    # Prepare features
    print("\n⚙️ Preparing features...")
    X, y = prepare_features(df_clean)
    print(f"✓ Feature shape: {X.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✓ Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Train model
    print("\n🎯 Training GradientBoostingRegressor...")
    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    print("✓ Model trained")
    
    # Evaluate
    print("\n📈 Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\nTraining Metrics:")
    print(f"  MAE: €{train_mae:.2f}/m²")
    print(f"  R²:  {train_r2:.3f}")
    
    print(f"\nTest Metrics:")
    print(f"  MAE: €{test_mae:.2f}/m²")
    print(f"  R²:  {test_r2:.3f}")
    
    # Cross-validation
    print("\n🔄 Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()
    print(f"  CV MAE: €{cv_mae:.2f}/m² (±€{cv_scores.std():.2f})")
    
    # Feature importance
    print("\n🔍 Feature Importance:")
    feature_names = [
        'city_tier', 'postcode_mult', 'living_space', 'rooms',
        'room_density', 'building_age', 'floor_normalized', 'size_category'
    ]
    for name, importance in sorted(
        zip(feature_names, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {name}: {importance:.3f}")
    
    # Save artifacts
    print(f"\n💾 Saving artifacts to {artifacts_path}...")
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    model_path = artifacts_path / "rent_model.joblib"
    joblib.dump(model, model_path)
    print(f"✓ Saved model to {model_path}")
    
    # Note: We don't save the preprocessor because it causes pickling issues
    # The preprocessing logic is in preprocess.py and doesn't need to be pickled
    
    
    metrics = {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "cv_mae": cv_mae,
        "n_samples": len(df_clean)
    }
    
    print("\n✅ Training complete!")
    return metrics


if __name__ == "__main__":
    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "training" / "training_data.csv"
    artifacts_path = project_root / "backend" / "app" / "artifacts"
    
    if not data_path.exists():
        print("⚠️ Training data not found. Run build_dataset.py first.")
        print(f"Expected path: {data_path}")
        sys.exit(1)
    
    metrics = train_pricing_model(data_path, artifacts_path)
    
    print("\n" + "=" * 50)
    print("📊 Final Metrics Summary")
    print("=" * 50)
    print(f"  Test MAE: €{metrics['test_mae']:.2f}/m²")
    print(f"  Test R²:  {metrics['test_r2']:.3f}")
    print(f"  Samples:  {metrics['n_samples']}")
