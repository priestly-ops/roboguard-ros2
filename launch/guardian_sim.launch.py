"""
guardian_sim.launch.py
======================
Launches Guardian with a simulated robot and optional failure injection.
Use this to demo Guardian or test healing strategies without real hardware.

Usage:
  ros2 launch guardian guardian_sim.launch.py
  ros2 launch guardian guardian_sim.launch.py inject_failures:=true
  ros2 launch guardian guardian_sim.launch.py failure_type:=memory_leak
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    inject_failures_arg = DeclareLaunchArgument(
        'inject_failures',
        default_value='false',
        description='Randomly inject simulated failures into mock nodes',
    )
    failure_type_arg = DeclareLaunchArgument(
        'failure_type',
        default_value='random',
        description='Type of failure to inject: random | cpu_spike | memory_leak | topic_drop | crash',
    )
    failure_interval_arg = DeclareLaunchArgument(
        'failure_interval_seconds',
        default_value='60.0',
        description='Average interval between injected failures (seconds)',
    )

    inject_failures = LaunchConfiguration('inject_failures')
    failure_type = LaunchConfiguration('failure_type')
    failure_interval = LaunchConfiguration('failure_interval_seconds')

    # ── Mock robot nodes (minimal publishers) ──────────────────────────────
    mock_nav = Node(
        package='guardian',
        executable='mock_node',
        name='navigation',
        namespace='robot',
        parameters=[{'mock_topics': ['/robot/cmd_vel', '/robot/odom'], 'publish_rate_hz': 20.0}],
        output='screen',
    )

    mock_slam = Node(
        package='guardian',
        executable='mock_node',
        name='slam',
        namespace='robot',
        parameters=[{'mock_topics': ['/robot/map', '/robot/scan'], 'publish_rate_hz': 5.0}],
        output='screen',
    )

    mock_arm = Node(
        package='guardian',
        executable='mock_node',
        name='arm_controller',
        namespace='robot',
        parameters=[{'mock_topics': ['/robot/joint_states'], 'publish_rate_hz': 50.0}],
        output='screen',
    )

    # ── Failure injector (optional) ────────────────────────────────────────
    failure_injector = Node(
        package='guardian',
        executable='failure_injector',
        name='failure_injector',
        parameters=[{
            'inject_failures': inject_failures,
            'failure_type': failure_type,
            'interval_seconds': failure_interval,
            'target_nodes': ['/robot/navigation', '/robot/slam', '/robot/arm_controller'],
        }],
        output='screen',
    )

    # ── Include main Guardian launch ────────────────────────────────────────
    guardian_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare('guardian'), 'launch', 'guardian.launch.py'])
        ),
        launch_arguments={
            'dry_run': 'false',
            'anomaly_model': 'isolation_forest',
            'predictor_model': 'lstm',
        }.items(),
    )

    return LaunchDescription([
        inject_failures_arg,
        failure_type_arg,
        failure_interval_arg,

        LogInfo(msg='🧪 Launching Guardian Simulation with mock robot nodes'),

        mock_nav,
        mock_slam,
        mock_arm,
        failure_injector,
        guardian_launch,
    ])
