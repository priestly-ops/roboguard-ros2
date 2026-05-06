# 🤖 ROS2 Guardian — AI-Powered Predictive Failure Detection & Self-Healing Platform

<div align="center">

![ROS2 Guardian](https://img.shields.io/badge/ROS2-Humble%20%7C%20Iron%20%7C%20Jazzy-blue?style=for-the-badge&logo=ros)
![Python](https://img.shields.io/badge/Python-3.10%2B-green?style=for-the-badge&logo=python)
![License](https://img.shields.io/badge/License-Apache%202.0-orange?style=for-the-badge)
![CI](https://img.shields.io/github/actions/workflow/status/your-org/ros2-guardian/ci.yml?style=for-the-badge&label=CI)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker)

**Predict failures before they happen. Heal automatically when they do.**

[Architecture](#architecture) · [Quick Start](#quick-start) · [Configuration](#configuration) · [Dashboard](#dashboard) · [Contributing](#contributing)

</div>

---

## Overview

**ROS2 Guardian** is an intelligent runtime platform that continuously monitors your ROS2 robot system, predicts component failures using machine learning, and autonomously executes recovery strategies — all without human intervention.

### Key Features

| Feature | Description |
|---|---|
| 🔍 **Real-time Health Monitoring** | Tracks CPU, memory, latency, topic throughput, TF tree integrity, and custom metrics for every node |
| 🧠 **ML Anomaly Detection** | Isolation Forest + LSTM-based time-series models detect anomalous behavior before failures occur |
| ⚡ **Predictive Failure Scoring** | Assigns a 0–100 risk score per node with configurable alert thresholds |
| 🔧 **Self-Healing Engine** | Automated recovery: node restart, parameter reset, topic remapping, hardware reinitialization |
| 📊 **Live Dashboard** | React-based web UI with real-time charts, alert history, and manual override controls |
| 🔔 **Multi-Channel Alerts** | ROS2 topics, email, Slack, and PagerDuty integrations |
| 📝 **Audit Trail** | Full event log of all detections, predictions, and healing actions |
| 🐳 **Docker Ready** | Single-command deployment with `docker compose up` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ROS2 Robot System                           │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐           │
│  │ Nav Node │  │SLAM Node │  │Arm Ctrl  │  │ Sensors  │  ...       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘           │
│       │              │              │              │                 │
│       └──────────────┴──────────────┴──────────────┘                │
│                              │                                      │
│                    ┌─────────▼──────────┐                          │
│                    │  Health Monitor    │  ← /guardian/metrics      │
│                    │      Node          │                           │
│                    └─────────┬──────────┘                          │
│                              │                                      │
│               ┌──────────────┼──────────────┐                      │
│               ▼              ▼              ▼                       │
│    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
│    │   Anomaly    │  │   Failure    │  │  Dashboard   │           │
│    │  Detector    │  │  Predictor   │  │   Bridge     │           │
│    │    Node      │  │    Node      │  │    Node      │           │
│    └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
│           │                  │                  │                   │
│           └──────────────────┼──────────────────┘                  │
│                              ▼                                      │
│                    ┌─────────────────────┐                         │
│                    │   Self-Healer Node  │                         │
│                    │  (Action Server)    │                         │
│                    └─────────────────────┘                         │
│                              │                                      │
│           ┌──────────────────┼──────────────────┐                  │
│           ▼                  ▼                  ▼                   │
│    ┌────────────┐   ┌─────────────┐   ┌──────────────┐            │
│    │   Restart  │   │  Remap /    │   │  Parameter   │            │
│    │   Node     │   │  Reconnect  │   │    Reset     │            │
│    └────────────┘   └─────────────┘   └──────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

### Components

- **`health_monitor_node`** — Collects system-wide metrics via `rclpy`, `/rosout`, and custom probes
- **`anomaly_detector_node`** — Runs Isolation Forest model on sliding windows of metric data
- **`failure_predictor_node`** — LSTM model predicts node failure probability over next N seconds
- **`self_healer_node`** — ROS2 Action Server that executes recovery workflows
- **`dashboard_bridge_node`** — WebSocket bridge for the React dashboard

---

## Quick Start

### Prerequisites

- ROS2 Humble / Iron / Jazzy
- Python 3.10+
- Docker & Docker Compose (optional)

### Option 1: Docker (Recommended)

```bash
git clone https://github.com/your-org/ros2-guardian.git
cd ros2-guardian
docker compose up
```

The dashboard will be available at `http://localhost:3000`.

### Option 2: Native ROS2

```bash
# Clone into your ROS2 workspace
cd ~/ros2_ws/src
git clone https://github.com/your-org/ros2-guardian.git

# Install Python dependencies
pip install -r requirements.txt

# Build
cd ~/ros2_ws
colcon build --packages-select guardian guardian_msgs
source install/setup.bash

# Launch
ros2 launch guardian guardian.launch.py
```

### Option 3: Quick Demo (No Robot Required)

```bash
# Simulate a robot system with injected failures
ros2 launch guardian guardian_sim.launch.py inject_failures:=true
```

---

## Configuration

Edit `config/default_params.yaml` to tune Guardian for your robot:

```yaml
guardian:
  # Monitoring
  monitor_rate_hz: 10.0
  history_window_seconds: 60
  
  # ML Models
  anomaly_detector:
    model: isolation_forest        # isolation_forest | autoencoder | one_class_svm
    contamination: 0.05
    retrain_interval_minutes: 30
  
  failure_predictor:
    model: lstm                    # lstm | transformer | xgboost
    prediction_horizon_seconds: 30
    alert_threshold: 0.75          # 0.0 - 1.0 risk score
    critical_threshold: 0.90
  
  # Self-Healing
  healing:
    max_restart_attempts: 3
    restart_cooldown_seconds: 10
    escalation_policy: notify_human  # restart | escalate | notify_human
    strategies:
      - name: soft_restart
        trigger_score: 0.75
      - name: hard_restart
        trigger_score: 0.90
      - name: parameter_reset
        trigger_score: 0.80
      - name: topic_remap
        trigger_score: 0.85
  
  # Notifications
  notifications:
    ros_topic: true
    slack_webhook: ""
    email: ""
    pagerduty_key: ""
```

See `config/healing_strategies.yaml` for per-node recovery overrides.

---

## Dashboard

The React dashboard provides:

- **Live System Map** — Visual graph of all nodes with health color coding
- **Metrics Charts** — Real-time time-series for CPU, memory, latency, throughput
- **Alert Feed** — Chronological list of anomalies and healing events
- **Risk Scores** — Per-node failure probability gauges
- **Manual Override** — Trigger or suppress healing actions
- **Model Performance** — Precision/recall metrics for the ML models

```bash
cd dashboard
npm install
npm run dev        # Development
npm run build      # Production build
```

---

## Custom Healing Strategies

Implement `BaseHealingStrategy` to add your own recovery logic:

```python
from guardian.healing.recovery_strategies import BaseHealingStrategy

class MyCustomRecovery(BaseHealingStrategy):
    name = "my_custom_recovery"
    
    async def can_handle(self, alert: SystemAlert) -> bool:
        return alert.node_name.startswith("/my_robot/arm")
    
    async def execute(self, alert: SystemAlert) -> HealingResult:
        # Your recovery logic here
        self.logger.info(f"Running custom recovery for {alert.node_name}")
        await self.restart_node(alert.node_name)
        return HealingResult(success=True, action_taken="custom_restart")
```

Register it in `config/default_params.yaml` under `custom_strategies`.

---

## Custom Metrics

Add robot-specific metrics via the `MetricProbe` interface:

```python
from guardian.utils.metrics import MetricProbe, MetricSample

class BatteryProbe(MetricProbe):
    probe_name = "battery_voltage"
    
    def collect(self) -> MetricSample:
        voltage = self.read_battery_voltage()
        return MetricSample(value=voltage, unit="V", tags={"cell": "main"})
```

---

## Training Custom Models

```bash
# Collect training data from a running robot
python scripts/collect_training_data.py --duration 3600 --output data/training.parquet

# Train models
python scripts/train_models.py \
  --data data/training.parquet \
  --model-dir models/ \
  --model-type lstm \
  --epochs 50

# Evaluate
python scripts/evaluate_models.py --model-dir models/ --test-data data/test.parquet
```

---

## ROS2 Interface

### Topics

| Topic | Type | Description |
|---|---|---|
| `/guardian/metrics` | `guardian_msgs/NodeHealth` | Per-node health metrics |
| `/guardian/alerts` | `guardian_msgs/SystemAlert` | Anomaly & failure alerts |
| `/guardian/healing_events` | `guardian_msgs/HealingAction` | Healing action log |
| `/guardian/system_status` | `std_msgs/String` (JSON) | Overall system health |

### Services

| Service | Type | Description |
|---|---|---|
| `/guardian/trigger_healing` | `guardian_msgs/TriggerHealing` | Manually trigger healing |
| `/guardian/get_health_report` | `guardian_msgs/GetHealthReport` | Full health snapshot |
| `/guardian/suppress_alerts` | `guardian_msgs/SuppressAlerts` | Silence alerts for node |

### Actions

| Action | Type | Description |
|---|---|---|
| `/guardian/execute_recovery` | `guardian_msgs/ExecuteRecovery` | Long-running recovery workflow |

---

## Project Structure

```
ros2-guardian/
├── guardian/                   # Main ROS2 Python package
│   ├── guardian/
│   │   ├── nodes/              # ROS2 node implementations
│   │   ├── ml/                 # ML models (anomaly, predictor)
│   │   ├── healing/            # Self-healing engine
│   │   └── utils/              # Shared utilities
│   ├── test/                   # Unit & integration tests
│   ├── package.xml
│   └── setup.py
├── guardian_msgs/              # Custom ROS2 message definitions
│   ├── msg/
│   ├── srv/
│   └── action/
├── dashboard/                  # React web dashboard
├── config/                     # YAML configuration files
├── launch/                     # ROS2 launch files
├── docker/                     # Dockerfile & compose
├── scripts/                    # Training & utility scripts
└── docs/                       # Architecture & API docs
```

---

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](docs/CONTRIBUTING.md) and open a PR.

```bash
# Run tests
colcon test --packages-select guardian
pytest guardian/test/ -v

# Lint
ament_flake8 guardian
ament_mypy guardian
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<div align="center">
Built with ❤️ for the robotics community
</div>
