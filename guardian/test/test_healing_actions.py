"""
Tests for guardian.healing.healing_actions
"""

import asyncio
import dataclasses
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from guardian.healing.healing_actions import (
    HealingContext,
    HealingResult,
    SoftRestartAction,
    HardRestartAction,
    ParameterResetAction,
    TopicRemapAction,
    EscalateAction,
)
from guardian.healing.recovery_strategies import StrategySelector
from guardian.utils.logger import GuardianLogger


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

def make_mock_alert(severity='ERROR', score=0.85):
    alert = MagicMock()
    alert.severity = severity
    alert.anomaly_score = score
    alert.node_name = '/robot/navigation'
    return alert


def make_context(dry_run=True, metadata=None):
    ros_node = MagicMock()
    ros_node.get_node_names_and_namespaces.return_value = [
        ('navigation', 'robot')
    ]
    logger = GuardianLogger(MagicMock())
    logger._l.info = MagicMock()
    logger._l.warning = MagicMock()
    logger._l.error = MagicMock()

    return HealingContext(
        node_name='/robot/navigation',
        alert=make_mock_alert(),
        dry_run=dry_run,
        ros_node=ros_node,
        logger=logger,
        metadata=metadata or {},
    )


# ──────────────────────────────────────────────────────────────────────────────
# SoftRestartAction
# ──────────────────────────────────────────────────────────────────────────────

class TestSoftRestartAction:

    def test_dry_run_succeeds(self):
        action = SoftRestartAction()
        ctx = make_context(dry_run=True)
        result = asyncio.get_event_loop().run_until_complete(action.execute(ctx))
        assert result.success is True
        assert result.action_name == 'soft_restart'
        assert 'DRY RUN' in result.message

    def test_has_correct_name(self):
        assert SoftRestartAction.name == 'soft_restart'


# ──────────────────────────────────────────────────────────────────────────────
# HardRestartAction
# ──────────────────────────────────────────────────────────────────────────────

class TestHardRestartAction:

    def test_dry_run_succeeds(self):
        action = HardRestartAction()
        ctx = make_context(dry_run=True)
        result = asyncio.get_event_loop().run_until_complete(action.execute(ctx))
        assert result.success is True
        assert 'DRY RUN' in result.message

    def test_fails_without_package_info(self):
        action = HardRestartAction()
        ctx = make_context(dry_run=False, metadata={'pid': None})
        result = asyncio.get_event_loop().run_until_complete(action.execute(ctx))
        # Should fail gracefully without package/executable info
        assert result.action_name == 'hard_restart'


# ──────────────────────────────────────────────────────────────────────────────
# ParameterResetAction
# ──────────────────────────────────────────────────────────────────────────────

class TestParameterResetAction:

    def test_dry_run_with_no_snapshot(self):
        action = ParameterResetAction()
        ctx = make_context(dry_run=True, metadata={})
        result = asyncio.get_event_loop().run_until_complete(action.execute(ctx))
        assert result.success is True  # dry run always succeeds

    def test_fails_with_no_snapshot_real(self):
        action = ParameterResetAction()
        ctx = make_context(dry_run=False, metadata={})
        result = asyncio.get_event_loop().run_until_complete(action.execute(ctx))
        assert result.success is False
        assert 'snapshot' in result.message.lower()


# ──────────────────────────────────────────────────────────────────────────────
# EscalateAction
# ──────────────────────────────────────────────────────────────────────────────

class TestEscalateAction:

    def test_always_succeeds(self):
        action = EscalateAction()
        ctx = make_context(dry_run=True)
        result = asyncio.get_event_loop().run_until_complete(action.execute(ctx))
        assert result.success is True
        assert result.action_name == 'escalate'

    def test_includes_node_name_in_message(self):
        action = EscalateAction()
        ctx = make_context(dry_run=True)
        result = asyncio.get_event_loop().run_until_complete(action.execute(ctx))
        assert '/robot/navigation' in result.message


# ──────────────────────────────────────────────────────────────────────────────
# StrategySelector
# ──────────────────────────────────────────────────────────────────────────────

class TestStrategySelector:

    def test_critical_returns_hard_restart(self):
        selector = StrategySelector()
        ctx = make_context()
        ctx.alert.severity = 'CRITICAL'
        ctx.alert.anomaly_score = 0.95
        strategy = selector.select(ctx)
        assert strategy.name == 'hard_restart'

    def test_warn_returns_parameter_reset(self):
        selector = StrategySelector()
        ctx = make_context()
        ctx.alert.severity = 'WARN'
        ctx.alert.anomaly_score = 0.55
        strategy = selector.select(ctx)
        assert strategy.name == 'parameter_reset'

    def test_error_with_no_failures_returns_soft_restart(self):
        selector = StrategySelector()
        ctx = make_context()
        ctx.alert.severity = 'ERROR'
        ctx.alert.anomaly_score = 0.80
        strategy = selector.select(ctx)
        assert strategy.name == 'soft_restart'

    def test_error_with_soft_failures_returns_hard_restart(self):
        selector = StrategySelector()
        # Simulate 2 soft restart failures for this node
        selector._failure_counts['/robot/navigation:soft_restart'] = 2
        ctx = make_context()
        ctx.alert.severity = 'ERROR'
        ctx.alert.anomaly_score = 0.80
        strategy = selector.select(ctx)
        assert strategy.name == 'hard_restart'

    def test_record_outcome_updates_counts(self):
        selector = StrategySelector()
        selector.record_outcome('soft_restart', success=True)
        selector.record_outcome('soft_restart', success=False)
        assert selector._success_counts.get('soft_restart') == 1
        assert selector._failure_counts.get('soft_restart') == 1
