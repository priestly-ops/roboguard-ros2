#!/usr/bin/env python3
"""
train_models.py
===============
Train Guardian ML models on collected metric data.

Usage:
  python scripts/train_models.py \
    --data data/training.parquet \
    --model-dir models/ \
    --model-type isolation_forest \
    --epochs 50

  python scripts/train_models.py \
    --data data/training.parquet \
    --model-dir models/ \
    --model-type lstm \
    --epochs 100 \
    --sequence-length 50
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from guardian.guardian.ml.anomaly_detector import (
    IsolationForestDetector,
    AutoencoderDetector,
    OneClassSVMDetector,
)
from guardian.guardian.ml.failure_predictor import LSTMPredictor, XGBoostPredictor
from guardian.guardian.ml.feature_extractor import FeatureExtractor


ANOMALY_MODELS = {
    'isolation_forest': IsolationForestDetector,
    'autoencoder': AutoencoderDetector,
    'one_class_svm': OneClassSVMDetector,
}

PREDICTOR_MODELS = {
    'lstm': LSTMPredictor,
    'xgboost': XGBoostPredictor,
}


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == '.parquet':
        return pd.read_parquet(path)
    elif p.suffix == '.csv':
        return pd.read_csv(path)
    elif p.suffix == '.json':
        return pd.read_json(path)
    else:
        raise ValueError(f"Unsupported data format: {p.suffix}")


def prepare_anomaly_features(df: pd.DataFrame, extractor: FeatureExtractor) -> np.ndarray:
    """Convert raw metric DataFrame to feature matrix for anomaly detection."""
    features = []
    for _, row in df.iterrows():
        feat = extractor._from_dict(row.to_dict())
        features.append(feat)
    return np.array(features, dtype=np.float32)


def prepare_predictor_sequences(
    df: pd.DataFrame,
    extractor: FeatureExtractor,
    seq_len: int,
    label_col: str = 'label',
):
    """Prepare windowed sequences and labels for LSTM/XGBoost predictor."""
    X, y = [], []
    grouped = df.groupby('node_name') if 'node_name' in df.columns else [('all', df)]

    for _, group in grouped:
        group = group.reset_index(drop=True)
        for i in range(len(group) - seq_len):
            window = group.iloc[i:i + seq_len]
            feats = [extractor._from_dict(row.to_dict()) for _, row in window.iterrows()]
            X.append(np.array(feats, dtype=np.float32))
            label = int(group.iloc[i + seq_len].get(label_col, 0))
            y.append(label)

    return np.array(X), np.array(y, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description='Train Guardian ML models')
    parser.add_argument('--data', required=True, help='Path to training data file')
    parser.add_argument('--model-dir', default='models/', help='Output directory for models')
    parser.add_argument(
        '--model-type',
        choices=[*ANOMALY_MODELS, *PREDICTOR_MODELS],
        default='isolation_forest',
        help='Model type to train',
    )
    parser.add_argument('--epochs', type=int, default=50, help='Training epochs (deep models)')
    parser.add_argument('--sequence-length', type=int, default=50, help='LSTM sequence length')
    parser.add_argument('--contamination', type=float, default=0.05, help='Anomaly fraction')
    parser.add_argument('--output-name', default='', help='Override output file name')
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    print(f"📂 Loading data from {args.data}…")
    df = load_data(args.data)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")

    extractor = FeatureExtractor()
    os.makedirs(args.model_dir, exist_ok=True)

    start = time.time()

    # ── Train anomaly detector ─────────────────────────────────────────────────
    if args.model_type in ANOMALY_MODELS:
        print(f"\n🧠 Training anomaly detector: {args.model_type}")
        X = prepare_anomaly_features(df, extractor)
        print(f"   Feature matrix: {X.shape}")

        cls = ANOMALY_MODELS[args.model_type]
        model = cls(contamination=args.contamination)
        model.fit(X)

        out_name = args.output_name or f"anomaly_{args.model_type}.pkl"
        out_path = os.path.join(args.model_dir, out_name)
        model.save(out_path)
        print(f"   ✅ Saved to {out_path}")

    # ── Train failure predictor ────────────────────────────────────────────────
    elif args.model_type in PREDICTOR_MODELS:
        print(f"\n🧠 Training failure predictor: {args.model_type}")
        X, y = prepare_predictor_sequences(df, extractor, args.sequence_length)
        print(f"   Sequences: {X.shape}, Labels: {y.shape} (pos={y.sum():.0f})")

        cls = PREDICTOR_MODELS[args.model_type]
        if args.model_type == 'lstm':
            model = cls(sequence_length=args.sequence_length, epochs=args.epochs)
        else:
            model = cls(sequence_length=args.sequence_length)
        model.fit(X, y)

        ext = 'pt' if args.model_type == 'lstm' else 'pkl'
        out_name = args.output_name or f"predictor_{args.model_type}.{ext}"
        out_path = os.path.join(args.model_dir, out_name)
        model.save(out_path)
        print(f"   ✅ Saved to {out_path}")

    elapsed = time.time() - start
    print(f"\n⏱  Training completed in {elapsed:.1f}s")

    # ── Save metadata ──────────────────────────────────────────────────────────
    meta = {
        'model_type': args.model_type,
        'training_samples': len(df),
        'sequence_length': args.sequence_length,
        'epochs': args.epochs,
        'contamination': args.contamination,
        'elapsed_seconds': elapsed,
        'output_path': out_path,
    }
    meta_path = os.path.join(args.model_dir, 'training_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"   Metadata saved to {meta_path}")


if __name__ == '__main__':
    main()
