"""
Train the scam detection models for RentShield.
Includes TF-IDF vectorizer, text classifier, and anomaly detector.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import sys


def train_text_classifier(df: pd.DataFrame, artifacts_path: Path) -> dict:
    """
    Train TF-IDF + Logistic Regression for scam text detection.
    
    Returns:
        Dictionary with training metrics
    """
    print("\n" + "=" * 50)
    print("📝 Training Text-Based Scam Classifier")
    print("=" * 50)
    
    # Prepare text data
    texts = df['description'].fillna('').values
    labels = df['scam_label'].values
    
    print(f"✓ Total samples: {len(texts)}")
    print(f"✓ Scam samples: {labels.sum()} ({labels.mean()*100:.1f}%)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # TF-IDF Vectorizer
    print("\n⚙️ Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'  # Also handles German stopwords partially
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    print(f"✓ Vocabulary size: {len(tfidf.vocabulary_)}")
    
    # Train classifier
    print("\n🎯 Training Logistic Regression classifier...")
    classifier = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate
    print("\n📈 Evaluation:")
    y_pred = classifier.predict(X_test_tfidf)
    y_proba = classifier.predict_proba(X_test_tfidf)[:, 1]
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Scam']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  TN: {cm[0,0]}, FP: {cm[0,1]}")
    print(f"  FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC-AUC: {roc_auc:.3f}")
    
    # Save models
    print(f"\n💾 Saving TF-IDF and classifier...")
    tfidf_path = artifacts_path / "tfidf.joblib"
    classifier_path = artifacts_path / "scam_model.joblib"
    
    joblib.dump(tfidf, tfidf_path)
    joblib.dump(classifier, classifier_path)
    
    print(f"✓ Saved TF-IDF to {tfidf_path}")
    print(f"✓ Saved classifier to {classifier_path}")
    
    return {
        "roc_auc": roc_auc,
        "precision": cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0,
        "recall": cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
    }


def train_anomaly_detector(df: pd.DataFrame, artifacts_path: Path) -> dict:
    """
    Train IsolationForest for price/size anomaly detection.
    
    Returns:
        Dictionary with training info
    """
    print("\n" + "=" * 50)
    print("🔍 Training Anomaly Detector")
    print("=" * 50)
    
    # Use only legitimate listings for normal pattern learning
    df_normal = df[df['scam_label'] == 0].copy()
    
    # Features for anomaly detection
    features = df_normal[['rent', 'living_space', 'rent_per_sqm']].values
    print(f"✓ Training on {len(features)} normal listings")
    
    # Train IsolationForest
    print("\n🎯 Training IsolationForest...")
    anomaly_model = IsolationForest(
        n_estimators=100,
        contamination=0.1,  # Expect 10% anomalies
        random_state=42,
        n_jobs=-1
    )
    anomaly_model.fit(features)
    
    # Test on all data
    all_features = df[['rent', 'living_space', 'rent_per_sqm']].values
    predictions = anomaly_model.predict(all_features)
    
    # -1 for anomalies, 1 for normal
    detected_anomalies = (predictions == -1).sum()
    print(f"✓ Detected {detected_anomalies} anomalies ({detected_anomalies/len(df)*100:.1f}%)")
    
    # Check how many scams are detected
    scam_anomalies = ((predictions == -1) & (df['scam_label'] == 1)).sum()
    total_scams = df['scam_label'].sum()
    print(f"✓ Scam detection rate: {scam_anomalies}/{total_scams} ({scam_anomalies/total_scams*100:.1f}%)")
    
    # Save model
    model_path = artifacts_path / "anomaly_model.joblib"
    joblib.dump(anomaly_model, model_path)
    print(f"✓ Saved anomaly model to {model_path}")
    
    return {
        "total_anomalies": int(detected_anomalies),
        "scam_detection_rate": scam_anomalies / total_scams if total_scams > 0 else 0
    }


def save_thresholds(artifacts_path: Path):
    """Save threshold configuration for scam scoring."""
    thresholds_config = {
        "thresholds": {
            "low": 0.25,
            "medium": 0.50,
            "high": 0.75
        },
        "weights": {
            "nlp": 0.5,
            "anomaly": 0.3,
            "rules": 0.2
        },
        "version": "1.0.0"
    }
    
    thresholds_path = artifacts_path / "thresholds.json"
    with open(thresholds_path, 'w') as f:
        json.dump(thresholds_config, f, indent=2)
    
    print(f"✓ Saved thresholds to {thresholds_path}")


def main():
    """Main training pipeline for scam detection."""
    print("=" * 50)
    print("🛡️ RentShield Scam Detection Training")
    print("=" * 50)
    
    # Paths
    project_root = Path(__file__).parent.parent
    data_path = project_root / "data" / "training" / "training_data.csv"
    artifacts_path = project_root / "backend" / "app" / "artifacts"
    
    if not data_path.exists():
        print("⚠️ Training data not found. Run build_dataset.py first.")
        print(f"Expected path: {data_path}")
        sys.exit(1)
    
    # Load data
    print("\n📊 Loading training data...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df)} samples")
    
    # Create artifacts directory
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    # Train text classifier
    text_metrics = train_text_classifier(df, artifacts_path)
    
    # Train anomaly detector
    anomaly_metrics = train_anomaly_detector(df, artifacts_path)
    
    # Save thresholds
    print("\n⚙️ Saving threshold configuration...")
    save_thresholds(artifacts_path)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Training Summary")
    print("=" * 50)
    print(f"\nText Classifier:")
    print(f"  ROC-AUC: {text_metrics['roc_auc']:.3f}")
    print(f"  Precision: {text_metrics['precision']:.3f}")
    print(f"  Recall: {text_metrics['recall']:.3f}")
    
    print(f"\nAnomaly Detector:")
    print(f"  Total anomalies: {anomaly_metrics['total_anomalies']}")
    print(f"  Scam detection: {anomaly_metrics['scam_detection_rate']*100:.1f}%")
    
    print("\n✅ All scam detection models trained successfully!")


if __name__ == "__main__":
    main()
