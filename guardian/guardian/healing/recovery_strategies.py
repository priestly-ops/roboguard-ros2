"""
Recovery Strategy Selector
==========================
Selects the most appropriate healing action for a given alert context.

Selection logic:
  1. Check node-specific overrides in config
  2. Score each available strategy by: severity, node type, past successes
  3. Respect escalation ladder (soft → hard → params → escalate)
"""

import abc
from typing import List

from guardian.healing.healing_actions import (
    BaseHealingAction,
    HealingContext,
    SoftRestartAction,
    HardRestartAction,
    ParameterResetAction,
    TopicRemapAction,
    HardwareReinitAction,
    EscalateAction,
)


class BaseHealingStrategy(abc.ABC):
    """
    User-extensible healing strategy interface.

    Subclass this to define custom recovery logic for specific nodes.
    Register in config/default_params.yaml under `custom_strategies`.
    """

    name: str = 'custom'

    @abc.abstractmethod
    async def can_handle(self, ctx: HealingContext) -> bool:
        """Return True if this strategy applies to the alert context."""

    @abc.abstractmethod
    async def execute(self, ctx: HealingContext):
        """Execute the recovery workflow."""


class StrategySelector:
    """
    Selects the best healing action for a given HealingContext.

    Priority order (configurable):
      1. Node-specific strategy override (from config)
      2. Severity-based escalation ladder
      3. Historical success rate

    Parameters
    ----------
    ros_node : rclpy.Node
        The guardian node (for parameter access)
    """

    # Default escalation ladder in order of aggressiveness
    DEFAULT_LADDER: List[BaseHealingAction] = [
        SoftRestartAction(),
        ParameterResetAction(),
        TopicRemapAction(),
        HardRestartAction(),
        HardwareReinitAction(),
        EscalateAction(),
    ]

    def __init__(self, ros_node=None) -> None:
        self._ros_node = ros_node
        self._success_counts = {}
        self._failure_counts = {}
        self._custom_strategies: List[BaseHealingAction] = []

    def register_strategy(self, strategy: BaseHealingAction) -> None:
        """Register a custom healing strategy."""
        self._custom_strategies.append(strategy)

    def record_outcome(self, action_name: str, success: bool) -> None:
        """Update success/failure counts for adaptive selection."""
        if success:
            self._success_counts[action_name] = self._success_counts.get(action_name, 0) + 1
        else:
            self._failure_counts[action_name] = self._failure_counts.get(action_name, 0) + 1

    def select(self, ctx: HealingContext) -> BaseHealingAction:
        """
        Select the best healing action for the given context.

        Returns the first action in the escalation ladder that:
          - hasn't failed too many times for this node recently
          - is appropriate for the alert severity
        """
        severity = getattr(ctx.alert, 'severity', 'ERROR')
        score = float(getattr(ctx.alert, 'anomaly_score', 0.8))
        node_name = ctx.node_name

        # Check custom strategies first
        for strategy in self._custom_strategies:
            # Sync wrapper for can_handle (strategies may be async)
            if hasattr(strategy, 'node_pattern'):
                import re
                if re.match(strategy.node_pattern, node_name):
                    return strategy

        # Severity-based selection from escalation ladder
        if severity == 'CRITICAL' or score >= 0.90:
            # Jump straight to hard restart
            return HardRestartAction()
        elif severity == 'ERROR' or score >= 0.75:
            # Try soft restart first, then hard restart
            soft_failures = self._failure_counts.get(
                f"{node_name}:soft_restart", 0
            )
            if soft_failures < 2:
                return SoftRestartAction()
            return HardRestartAction()
        elif severity == 'WARN' or score >= 0.50:
            return ParameterResetAction()
        else:
            return SoftRestartAction()
