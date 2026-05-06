#!/usr/bin/env python3
"""
generate_test_data.py
=====================
Generates synthetic robot metric data for model training and testing.

Simulates:
  - Normal operating patterns (multi-modal, noisy)
  - CPU spikes (gradual increase)
  - Memory leaks (monotonic increase)
  - Topic drops (rate → 0)
  - Node crashes (metrics → 0 then absent)

Usage:
  python scripts/generate_test_data.py --output data/synthetic.parquet --n-nodes 5
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


LIFECYCLE_STATES = ['active', 'inactive', 'active', 'active', 'active']


def generate_node_data(
    node_name: str,
    n_samples: int = 3600,
    failure_rate: float = 0.1,
    rng: np.random.Generator = None,
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng()

    records = []
    failure_mode = None
    failure_start = None
    memory_base = rng.uniform(100, 800)  # MB
    cpu_base = rng.uniform(5, 60)         # %

    for i in range(n_samples):
        t = i / 10.0  # 10 Hz → seconds

        # Randomly inject failures
        if failure_mode is None and rng.random() < failure_rate / n_samples * 20:
            failure_mode = rng.choice(['cpu_spike', 'memory_leak', 'topic_drop', 'crash'])
            failure_start = i

        # Compute metrics based on state
        if failure_mode == 'cpu_spike' and failure_start is not None:
            progress = min(1.0, (i - failure_start) / 100)
            cpu = cpu_base + progress * 80.0 + rng.normal(0, 2)
            mem = memory_base + rng.normal(0, 5)
        elif failure_mode == 'memory_leak' and failure_start is not None:
            progress = (i - failure_start) / 200
            cpu = cpu_base + rng.normal(0, 3)
            mem = memory_base + progress * 500 + rng.normal(0, 10)
        elif failure_mode == 'topic_drop' and failure_start is not None:
            progress = min(1.0, (i - failure_start) / 50)
            cpu = cpu_base * (1 - progress * 0.5) + rng.normal(0, 2)
            mem = memory_base + rng.normal(0, 5)
        elif failure_mode == 'crash' and failure_start is not None:
            if i - failure_start > 20:
                cpu = 0.0
                mem = 0.0
            else:
                cpu = cpu_base + rng.normal(0, 5)
                mem = memory_base + rng.normal(0, 5)
        else:
            cpu = cpu_base + rng.normal(0, 3) + 5 * np.sin(t / 30)
            mem = memory_base + rng.normal(0, 10)

        # Topic rates
        n_topics = 3
        if failure_mode == 'topic_drop' and failure_start and i > failure_start + 30:
            base_rate = 0.5
        else:
            base_rate = rng.uniform(5, 25)
        topic_rates = [max(0.0, base_rate + rng.normal(0, 1)) for _ in range(n_topics)]

        # Latency
        latency_ms = max(0.1, 2.0 + rng.normal(0, 0.5) + cpu / 50)

        # Label: 1 if within 30s before a failure peak
        is_pre_failure = 0
        if failure_start is not None:
            time_to_failure = failure_start + 100 - i
            if 0 < time_to_failure < 30 * 10:  # 30s window at 10Hz
                is_pre_failure = 1

        records.append({
            'node_name': node_name,
            'timestamp': t,
            'cpu_percent': max(0.0, min(100.0, cpu)),
            'memory_mb': max(0.0, mem),
            'memory_percent': max(0.0, min(100.0, mem / 8000 * 100)),
            'thread_count': int(rng.integers(5, 20)),
            'open_files': int(rng.integers(3, 15)),
            'avg_latency_ms': latency_ms,
            'max_latency_ms': latency_ms * rng.uniform(1.5, 4.0),
            'topic_rates_hz': topic_rates,
            'is_alive': int(not (failure_mode == 'crash' and failure_start and i > failure_start + 25)),
            'lifecycle_state': 'active' if cpu > 0 else 'error',
            'label': is_pre_failure,
            'failure_mode': failure_mode or 'none',
        })

        # Reset failure after some time
        if failure_mode == 'crash' and failure_start and i > failure_start + 50:
            failure_mode = None
            failure_start = None
            memory_base = rng.uniform(100, 800)
            cpu_base = rng.uniform(5, 60)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic robot metric data')
    parser.add_argument('--output', default='data/synthetic.parquet', help='Output file path')
    parser.add_argument('--n-nodes', type=int, default=5, help='Number of simulated nodes')
    parser.add_argument('--n-samples', type=int, default=3600, help='Samples per node')
    parser.add_argument('--failure-rate', type=float, default=0.15, help='Failure injection rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    node_names = [
        '/robot/navigation',
        '/robot/slam',
        '/robot/arm_controller',
        '/robot/camera_driver',
        '/robot/lidar_driver',
    ][:args.n_nodes]

    print(f"🔬 Generating synthetic data for {args.n_nodes} nodes × {args.n_samples} samples…")

    dfs = []
    for node_name in node_names:
        print(f"   {node_name}…")
        df = generate_node_data(node_name, args.n_samples, args.failure_rate, rng)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n📊 Dataset: {len(combined)} rows")
    print(f"   Labels: {combined['label'].sum()} pre-failure samples "
          f"({combined['label'].mean():.1%})")
    print(f"   Failure modes: {combined['failure_mode'].value_counts().to_dict()}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    if args.output.endswith('.parquet'):
        combined.to_parquet(args.output, index=False)
    else:
        combined.to_csv(args.output, index=False)

    print(f"\n✅ Saved to {args.output}")


if __name__ == '__main__':
    main()
