"""
guardian.launch.py
==================
Launches all Guardian nodes: health monitor, anomaly detector,
failure predictor, self-healer, and dashboard bridge.

Usage:
  ros2 launch guardian guardian.launch.py
  ros2 launch guardian guardian.launch.py config_file:=/path/to/my_params.yaml
  ros2 launch guardian guardian.launch.py dry_run:=true
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    GroupAction,
    LogInfo,
    OpaqueFunction,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    # ── Launch arguments ──────────────────────────────────────────────────────
    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=PathJoinSubstitution([
            FindPackageShare('guardian'), 'config', 'default_params.yaml'
        ]),
        description='Path to Guardian YAML configuration file',
    )

    dry_run_arg = DeclareLaunchArgument(
        'dry_run',
        default_value='false',
        description='Run in dry-run mode (log actions without executing)',
    )

    namespace_arg = DeclareLaunchArgument(
        'namespace',
        default_value='guardian',
        description='ROS2 namespace for all Guardian nodes',
    )

    model_type_arg = DeclareLaunchArgument(
        'anomaly_model',
        default_value='isolation_forest',
        description='Anomaly detection model: isolation_forest | autoencoder | one_class_svm',
    )

    predictor_model_arg = DeclareLaunchArgument(
        'predictor_model',
        default_value='lstm',
        description='Failure prediction model: lstm | xgboost',
    )

    # ── Shared parameters ─────────────────────────────────────────────────────
    config_file = LaunchConfiguration('config_file')
    dry_run = LaunchConfiguration('dry_run')
    namespace = LaunchConfiguration('namespace')
    anomaly_model = LaunchConfiguration('anomaly_model')
    predictor_model = LaunchConfiguration('predictor_model')

    # ── Health Monitor Node ───────────────────────────────────────────────────
    health_monitor = Node(
        package='guardian',
        executable='health_monitor',
        name='health_monitor',
        namespace=namespace,
        parameters=[
            config_file,
            {
                'monitor_rate_hz': 10.0,
                'history_window_seconds': 60.0,
            }
        ],
        output='screen',
        emulate_tty=True,
    )

    # ── Anomaly Detector Node ─────────────────────────────────────────────────
    anomaly_detector = Node(
        package='guardian',
        executable='anomaly_detector',
        name='anomaly_detector',
        namespace=namespace,
        parameters=[
            config_file,
            {
                'model_type': anomaly_model,
                'window_size': 30,
                'contamination': 0.05,
                'alert_cooldown_seconds': 30.0,
            }
        ],
        output='screen',
        emulate_tty=True,
    )

    # ── Failure Predictor Node ────────────────────────────────────────────────
    failure_predictor = Node(
        package='guardian',
        executable='failure_predictor',
        name='failure_predictor',
        namespace=namespace,
        parameters=[
            config_file,
            {
                'model_type': predictor_model,
                'sequence_length': 50,
                'prediction_horizon_seconds': 30.0,
                'alert_threshold': 0.75,
                'critical_threshold': 0.90,
            }
        ],
        output='screen',
        emulate_tty=True,
    )

    # ── Self-Healer Node ──────────────────────────────────────────────────────
    self_healer = Node(
        package='guardian',
        executable='self_healer',
        name='self_healer',
        namespace=namespace,
        parameters=[
            config_file,
            {
                'max_restart_attempts': 3,
                'restart_cooldown_seconds': 10.0,
                'escalation_policy': 'notify_human',
                'verify_timeout_seconds': 15.0,
                'dry_run': dry_run,
            }
        ],
        output='screen',
        emulate_tty=True,
    )

    # ── Dashboard Bridge Node ─────────────────────────────────────────────────
    dashboard_bridge = Node(
        package='guardian',
        executable='dashboard_bridge',
        name='dashboard_bridge',
        namespace=namespace,
        parameters=[
            config_file,
            {'websocket_port': 8765}
        ],
        output='screen',
        emulate_tty=True,
    )

    return LaunchDescription([
        config_arg,
        dry_run_arg,
        namespace_arg,
        model_type_arg,
        predictor_model_arg,

        LogInfo(msg='🤖 Launching ROS2 Guardian — AI-Powered Self-Healing Platform'),

        GroupAction(actions=[
            health_monitor,
            anomaly_detector,
            failure_predictor,
            self_healer,
            dashboard_bridge,
        ]),
    ])
