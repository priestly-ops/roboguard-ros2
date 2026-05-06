from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'guardian'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'scikit-learn>=1.3.0',
        'pandas>=2.0.0',
        'torch>=2.0.0',
        'websockets>=11.0',
        'aiohttp>=3.8.0',
        'pyyaml>=6.0',
        'psutil>=5.9.0',
    ],
    zip_safe=True,
    maintainer='ROS2 Guardian Team',
    maintainer_email='guardian@your-org.com',
    description='AI-powered predictive failure detection and self-healing platform for ROS2',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'health_monitor     = guardian.nodes.health_monitor_node:main',
            'anomaly_detector   = guardian.nodes.anomaly_detector_node:main',
            'failure_predictor  = guardian.nodes.failure_predictor_node:main',
            'self_healer        = guardian.nodes.self_healer_node:main',
            'dashboard_bridge   = guardian.nodes.dashboard_bridge_node:main',
            'mock_robot         = guardian.nodes.mock_robot_node:main',
            'failure_injector   = guardian.nodes.failure_injector_node:main',
        ],
    },
)
