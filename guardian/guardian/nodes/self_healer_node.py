"""
Self-Healer Node
================
ROS2 Action Server that receives SystemAlert messages and executes
recovery workflows autonomously.

Healing pipeline:
  1. Receive alert (via subscription or action goal)
  2. Select best healing strategy for the failing node
  3. Execute strategy with feedback streaming
  4. Verify recovery success
  5. Escalate or retry on failure
  6. Log all events to /guardian/healing_events

Healing strategies (in order of escalation):
  soft_restart     → lifecycle_node.deactivate + reactivate
  parameter_reset  → reset params to last-known-good values
  hard_restart     → kill PID + re-launch via launch service
  topic_remap      → reconnect to alternative topic remappings
  hw_reinit        → send hardware re-initialization command
  escalate         → notify human via topic + external webhooks
"""

import asyncio
import threading
import time
from typing import Dict, List

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rcl_interfaces.msg import ParameterDescriptor

from guardian_msgs.msg import SystemAlert, HealingAction
from guardian_msgs.action import ExecuteRecovery
from guardian_msgs.srv import TriggerHealing
from guardian.healing.healing_actions import (
    HealingContext,
    EscalateAction,
)
from guardian.healing.recovery_strategies import StrategySelector
from guardian.utils.logger import GuardianLogger


class SelfHealerNode(Node):
    """
    Autonomous self-healing engine for ROS2 nodes.

    Parameters
    ----------
    max_restart_attempts : int
        Max consecutive restarts before escalating to human.
    restart_cooldown_seconds : float
        Minimum wait between successive restarts of the same node.
    escalation_policy : str
        What to do after max retries: notify_human | stop_robot | ignore
    verify_timeout_seconds : float
        Seconds to wait for a node to recover before declaring failure.
    dry_run : bool
        If True, log actions but do not execute them (safe mode).
    """

    NODE_NAME = 'guardian_self_healer'

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self._declare_params()

        self.log = GuardianLogger(self.get_logger())
        self.max_retries: int = self.get_parameter('max_restart_attempts').value
        self.cooldown: float = self.get_parameter('restart_cooldown_seconds').value
        self.escalation_policy: str = self.get_parameter('escalation_policy').value
        self.verify_timeout: float = self.get_parameter('verify_timeout_seconds').value
        self.dry_run: bool = self.get_parameter('dry_run').value

        if self.dry_run:
            self.log.warn("Self-Healer running in DRY RUN mode — no actions will be executed")

        self.log.info(
            f"SelfHealerNode starting — max_retries={self.max_retries}, "
            f"cooldown={self.cooldown}s, escalation={self.escalation_policy}"
        )

        # ── Healing actions ───────────────────────────────────────────────────
        self._strategy_selector = StrategySelector(self)
        self._restart_tracker: Dict[str, List[float]] = {}
        self._active_healings: Dict[str, bool] = {}
        self._lock = threading.Lock()

        # ── ROS2 I/O ──────────────────────────────────────────────────────────
        cb_group = ReentrantCallbackGroup()

        self._alert_sub = self.create_subscription(
            SystemAlert, '/guardian/alerts', self._on_alert, 10,
            callback_group=cb_group
        )
        self._event_pub = self.create_publisher(
            HealingAction, '/guardian/healing_events', 10
        )
        self._status_pub = self.create_publisher(
            SystemAlert, '/guardian/healing_status', 10
        )

        # Services
        self._trigger_srv = self.create_service(
            TriggerHealing, '/guardian/trigger_healing',
            self._handle_trigger_healing,
            callback_group=cb_group
        )

        # Action server for long-running recovery workflows
        self._action_server = ActionServer(
            self,
            ExecuteRecovery,
            '/guardian/execute_recovery',
            execute_callback=self._execute_recovery,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
            callback_group=cb_group,
        )

        self.log.info("SelfHealerNode ready")

    # ──────────────────────────────────────────────────────────────────────────
    # Parameter declarations
    # ──────────────────────────────────────────────────────────────────────────

    def _declare_params(self) -> None:
        self.declare_parameter(
            'max_restart_attempts', 3,
            ParameterDescriptor(description='Max restarts before escalation')
        )
        self.declare_parameter(
            'restart_cooldown_seconds', 10.0,
            ParameterDescriptor(description='Cooldown between restarts of same node')
        )
        self.declare_parameter(
            'escalation_policy', 'notify_human',
            ParameterDescriptor(description='notify_human | stop_robot | ignore')
        )
        self.declare_parameter(
            'verify_timeout_seconds', 15.0,
            ParameterDescriptor(description='Timeout for post-healing verification')
        )
        self.declare_parameter(
            'dry_run', False,
            ParameterDescriptor(description='Log but do not execute healing actions')
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Alert subscription callback
    # ──────────────────────────────────────────────────────────────────────────

    def _on_alert(self, alert: SystemAlert) -> None:
        if alert.severity not in ('ERROR', 'CRITICAL'):
            return

        node_name = alert.node_name
        with self._lock:
            if self._active_healings.get(node_name):
                self.log.debug(f"Healing already in progress for {node_name}, skipping")
                return
            self._active_healings[node_name] = True

        threading.Thread(
            target=self._run_healing_sync,
            args=(alert,),
            daemon=True
        ).start()

    def _run_healing_sync(self, alert: SystemAlert) -> None:
        """Synchronous wrapper for the async healing workflow."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._heal(alert))
        finally:
            loop.close()
            with self._lock:
                self._active_healings[alert.node_name] = False

    # ──────────────────────────────────────────────────────────────────────────
    # Core healing workflow
    # ──────────────────────────────────────────────────────────────────────────

    async def _heal(self, alert: SystemAlert, goal_handle=None) -> bool:
        node_name = alert.node_name
        start_time = time.monotonic()

        self.log.info(f"🔧 Starting healing for {node_name} (score={alert.anomaly_score:.3f})")

        # ── Cooldown check ────────────────────────────────────────────────────
        restarts = self._restart_tracker.get(node_name, [])
        recent = [t for t in restarts if time.monotonic() - t < self.cooldown]
        if len(recent) >= self.max_retries:
            self.log.warn(
                f"Max retries ({self.max_retries}) reached for {node_name}. "
                f"Applying escalation policy: {self.escalation_policy}"
            )
            await self._escalate(alert)
            return False

        # ── Strategy selection ────────────────────────────────────────────────
        ctx = HealingContext(
            node_name=node_name,
            alert=alert,
            dry_run=self.dry_run,
            ros_node=self,
            logger=self.log,
        )
        strategy = self._strategy_selector.select(ctx)
        self.log.info(f"Selected strategy: {strategy.name} for {node_name}")

        # ── Publish start event ───────────────────────────────────────────────
        self._publish_healing_event(
            node_name=node_name,
            action_name=strategy.name,
            status='STARTED',
            message=f"Executing {strategy.name}",
        )

        # ── Execute ───────────────────────────────────────────────────────────
        success = False
        try:
            if goal_handle:
                feedback = ExecuteRecovery.Feedback()
                feedback.current_action = strategy.name
                feedback.progress_percent = 0.0
                goal_handle.publish_feedback(feedback)

            result = await strategy.execute(ctx)
            success = result.success

            if goal_handle:
                feedback.progress_percent = 75.0
                feedback.current_action = 'verifying'
                goal_handle.publish_feedback(feedback)

        except Exception as exc:  # noqa: BLE001
            self.log.error(f"Strategy {strategy.name} raised exception: {exc}")
            success = False

        # ── Verify recovery ───────────────────────────────────────────────────
        if success:
            success = await self._verify_recovery(node_name)

        # ── Log outcome ───────────────────────────────────────────────────────
        elapsed = time.monotonic() - start_time
        status = 'SUCCEEDED' if success else 'FAILED'
        self._publish_healing_event(
            node_name=node_name,
            action_name=strategy.name,
            status=status,
            message=f"Healing {status} in {elapsed:.1f}s",
        )

        if success:
            self.log.info(f"✅ Healing succeeded for {node_name} via {strategy.name}")
        else:
            self.log.warn(f"❌ Healing failed for {node_name}. Tracking restart attempt.")
            restarts.append(time.monotonic())
            self._restart_tracker[node_name] = restarts

        return success

    async def _verify_recovery(self, node_name: str) -> bool:
        """Poll for node health to verify recovery."""
        deadline = time.monotonic() + self.verify_timeout
        self.log.info(f"Verifying recovery for {node_name} (timeout={self.verify_timeout}s)")

        while time.monotonic() < deadline:
            await asyncio.sleep(1.0)
            # In real implementation: check /guardian/metrics for node_name
            # For now, check if the ROS2 node is listed as alive
            known_nodes = [
                f"/{ns}/{n}".replace('//', '/')
                for n, ns in self.get_node_names_and_namespaces()
            ]
            if any(node_name in kn for kn in known_nodes):
                return True

        self.log.warn(f"Recovery verification timed out for {node_name}")
        return False

    async def _escalate(self, alert: SystemAlert) -> None:
        if self.escalation_policy == 'notify_human':
            self._publish_healing_event(
                node_name=alert.node_name,
                action_name='escalate',
                status='ESCALATED',
                message=f"⚠️ Human intervention required for {alert.node_name}",
            )
            ctx = HealingContext(
                node_name=alert.node_name,
                alert=alert,
                dry_run=self.dry_run,
                ros_node=self,
                logger=self.log,
            )
            escalator = EscalateAction()
            await escalator.execute(ctx)
        elif self.escalation_policy == 'stop_robot':
            self.log.error(f"STOPPING ROBOT due to unrecoverable failure in {alert.node_name}")
            # Emergency stop topic
            from std_msgs.msg import Bool
            estop_pub = self.create_publisher(Bool, '/emergency_stop', 1)
            msg = Bool()
            msg.data = True
            estop_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────────────
    # Action server callbacks
    # ──────────────────────────────────────────────────────────────────────────

    def _goal_callback(self, goal_request):
        self.log.info(f"Received healing goal for {goal_request.node_name}")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle):
        self.log.info("Healing goal cancelled")
        return CancelResponse.ACCEPT

    async def _execute_recovery(self, goal_handle) -> ExecuteRecovery.Result:
        request = goal_handle.request
        alert = SystemAlert()
        alert.node_name = request.node_name
        alert.severity = request.severity or 'ERROR'
        alert.anomaly_score = request.anomaly_score

        success = await self._heal(alert, goal_handle=goal_handle)

        result = ExecuteRecovery.Result()
        result.success = success
        result.actions_taken = [request.node_name]

        if success:
            goal_handle.succeed()
        else:
            goal_handle.abort()

        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Service handler
    # ──────────────────────────────────────────────────────────────────────────

    def _handle_trigger_healing(
        self,
        request: TriggerHealing.Request,
        response: TriggerHealing.Response
    ) -> TriggerHealing.Response:
        alert = SystemAlert()
        alert.node_name = request.node_name
        alert.severity = 'ERROR'
        alert.anomaly_score = 1.0
        alert.message = 'Manually triggered healing'

        self._on_alert(alert)
        response.accepted = True
        response.message = f"Healing triggered for {request.node_name}"
        return response

    # ──────────────────────────────────────────────────────────────────────────
    # Event publishing
    # ──────────────────────────────────────────────────────────────────────────

    def _publish_healing_event(
        self,
        node_name: str,
        action_name: str,
        status: str,
        message: str,
    ) -> None:
        evt = HealingAction()
        evt.header.stamp = self.get_clock().now().to_msg()
        evt.node_name = node_name
        evt.action_name = action_name
        evt.status = status
        evt.message = message
        evt.timestamp = self.get_clock().now().to_msg()
        self._event_pub.publish(evt)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main(args=None) -> None:
    rclpy.init(args=args)
    node = SelfHealerNode()
    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
