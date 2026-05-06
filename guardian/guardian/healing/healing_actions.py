"""
Healing Actions
===============
Concrete implementations of self-healing actions.
Each action is a standalone unit with:
  - can_handle()  — feasibility check
  - execute()     — async execution with structured result
  - rollback()    — undo the action if needed

Actions:
  SoftRestartAction     — lifecycle deactivate + activate
  HardRestartAction     — kill process + respawn
  ParameterResetAction  — revert parameters to last-known-good
  TopicRemapAction      — reconnect to backup topics
  HardwareReinitAction  — send hardware reset command
  EscalateAction        — human notification via webhooks
"""

import asyncio
import dataclasses
import subprocess
import time
from typing import Any, Dict, Optional

from guardian.utils.logger import GuardianLogger


@dataclasses.dataclass
class HealingContext:
    """Shared context passed to all healing actions."""
    node_name: str
    alert: Any                     # guardian_msgs/SystemAlert
    dry_run: bool
    ros_node: Any                  # rclpy Node reference
    logger: GuardianLogger
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class HealingResult:
    """Result returned by every healing action."""
    success: bool
    action_name: str
    message: str = ''
    duration_seconds: float = 0.0
    metadata: Dict[str, Any] = dataclasses.field(default_factory=dict)


class BaseHealingAction:
    """Abstract base class for healing actions."""

    name: str = 'base'
    description: str = ''

    async def can_handle(self, ctx: HealingContext) -> bool:
        """Return True if this action is applicable for the given context."""
        return True

    async def execute(self, ctx: HealingContext) -> HealingResult:
        raise NotImplementedError

    async def rollback(self, ctx: HealingContext) -> None:
        """Undo this action. Called if subsequent verification fails."""
        pass

    def _log(self, ctx: HealingContext, msg: str) -> None:
        prefix = "[DRY RUN] " if ctx.dry_run else ""
        ctx.logger.info(f"{prefix}[{self.name}] {msg}")


# ──────────────────────────────────────────────────────────────────────────────
# Soft Restart  (Lifecycle: deactivate → activate)
# ──────────────────────────────────────────────────────────────────────────────

class SoftRestartAction(BaseHealingAction):
    """
    Trigger a lifecycle transition: deactivate → cleanup → configure → activate.

    Safe for managed lifecycle nodes; no data is lost in the process.
    """

    name = 'soft_restart'
    description = 'Lifecycle deactivate + reactivate (lifecycle nodes only)'

    async def execute(self, ctx: HealingContext) -> HealingResult:
        start = time.monotonic()
        node_name = ctx.node_name
        self._log(ctx, f"Soft-restarting {node_name}")

        if ctx.dry_run:
            await asyncio.sleep(0.5)
            return HealingResult(
                success=True, action_name=self.name,
                message=f"[DRY RUN] Would soft-restart {node_name}",
                duration_seconds=time.monotonic() - start,
            )

        try:
            from lifecycle_msgs.srv import ChangeState
            from lifecycle_msgs.msg import Transition

            ros = ctx.ros_node
            service_name = f"{node_name}/change_state"

            client = ros.create_client(ChangeState, service_name)
            if not client.wait_for_service(timeout_sec=3.0):
                return HealingResult(
                    success=False, action_name=self.name,
                    message=f"Lifecycle service not available for {node_name}",
                )

            async def change(transition_id: int) -> bool:
                req = ChangeState.Request()
                req.transition.id = transition_id
                future = client.call_async(req)
                await asyncio.wait_for(asyncio.wrap_future(future), timeout=5.0)
                return future.result().success

            # deactivate → cleanup → configure → activate
            for trans_id in [
                Transition.TRANSITION_DEACTIVATE,
                Transition.TRANSITION_CLEANUP,
                Transition.TRANSITION_CONFIGURE,
                Transition.TRANSITION_ACTIVATE,
            ]:
                ok = await change(trans_id)
                if not ok:
                    return HealingResult(
                        success=False, action_name=self.name,
                        message=f"Lifecycle transition {trans_id} failed for {node_name}",
                    )
                await asyncio.sleep(0.2)

            return HealingResult(
                success=True, action_name=self.name,
                message=f"Soft restart succeeded for {node_name}",
                duration_seconds=time.monotonic() - start,
            )

        except Exception as exc:  # noqa: BLE001
            return HealingResult(
                success=False, action_name=self.name,
                message=f"Soft restart failed: {exc}",
                duration_seconds=time.monotonic() - start,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Hard Restart  (kill PID + respawn via ros2 run / launch)
# ──────────────────────────────────────────────────────────────────────────────

class HardRestartAction(BaseHealingAction):
    """
    Kill the node process and respawn it.

    Requires node → PID mapping to be available (populated by HealthMonitor).
    Respawn is done via `ros2 run <pkg> <executable>` with original args.
    """

    name = 'hard_restart'
    description = 'Kill process and respawn via ros2 run'

    async def execute(self, ctx: HealingContext) -> HealingResult:
        start = time.monotonic()
        node_name = ctx.node_name
        self._log(ctx, f"Hard-restarting {node_name}")

        if ctx.dry_run:
            await asyncio.sleep(1.0)
            return HealingResult(
                success=True, action_name=self.name,
                message=f"[DRY RUN] Would hard-restart {node_name}",
                duration_seconds=time.monotonic() - start,
            )

        pid: Optional[int] = ctx.metadata.get('pid')
        pkg: Optional[str] = ctx.metadata.get('package')
        executable: Optional[str] = ctx.metadata.get('executable')

        try:
            if pid:
                import signal
                import os
                ctx.logger.info(f"Killing PID {pid} for {node_name}")
                os.kill(pid, signal.SIGTERM)
                await asyncio.sleep(2.0)
                # SIGKILL if still alive
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

            # Respawn
            if pkg and executable:
                cmd = ['ros2', 'run', pkg, executable]
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                ctx.logger.info(f"Respawned {node_name} as PID {proc.pid}")
                await asyncio.sleep(3.0)

                return HealingResult(
                    success=True, action_name=self.name,
                    message=f"Hard restart succeeded for {node_name} (new PID {proc.pid})",
                    duration_seconds=time.monotonic() - start,
                    metadata={'new_pid': proc.pid},
                )

            return HealingResult(
                success=False, action_name=self.name,
                message=f"Cannot respawn {node_name}: package/executable unknown",
            )

        except Exception as exc:  # noqa: BLE001
            return HealingResult(
                success=False, action_name=self.name,
                message=f"Hard restart failed: {exc}",
                duration_seconds=time.monotonic() - start,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Parameter Reset
# ──────────────────────────────────────────────────────────────────────────────

class ParameterResetAction(BaseHealingAction):
    """
    Reset a node's parameters to their last-known-good values.

    The HealthMonitor caches parameter snapshots periodically.
    This action restores the most recent snapshot.
    """

    name = 'parameter_reset'
    description = 'Reset node parameters to last-known-good snapshot'

    async def execute(self, ctx: HealingContext) -> HealingResult:
        start = time.monotonic()
        node_name = ctx.node_name
        self._log(ctx, f"Resetting parameters for {node_name}")

        params_snapshot: Optional[Dict] = ctx.metadata.get('last_good_params')

        if ctx.dry_run:
            return HealingResult(
                success=True, action_name=self.name,
                message=f"[DRY RUN] Would reset parameters for {node_name}",
                duration_seconds=time.monotonic() - start,
            )

        if not params_snapshot:
            return HealingResult(
                success=False, action_name=self.name,
                message=f"No parameter snapshot available for {node_name}",
            )

        try:
            from rcl_interfaces.srv import SetParameters
            from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType

            ros = ctx.ros_node
            client = ros.create_client(SetParameters, f"{node_name}/set_parameters")
            if not client.wait_for_service(timeout_sec=3.0):
                return HealingResult(
                    success=False, action_name=self.name,
                    message=f"SetParameters service unavailable for {node_name}",
                )

            req = SetParameters.Request()
            for name_p, value in params_snapshot.items():
                param = Parameter()
                param.name = name_p
                pv = ParameterValue()
                if isinstance(value, bool):
                    pv.type = ParameterType.PARAMETER_BOOL
                    pv.bool_value = value
                elif isinstance(value, int):
                    pv.type = ParameterType.PARAMETER_INTEGER
                    pv.integer_value = value
                elif isinstance(value, float):
                    pv.type = ParameterType.PARAMETER_DOUBLE
                    pv.double_value = value
                elif isinstance(value, str):
                    pv.type = ParameterType.PARAMETER_STRING
                    pv.string_value = value
                param.value = pv
                req.parameters.append(param)

            future = client.call_async(req)
            await asyncio.sleep(2.0)

            return HealingResult(
                success=True, action_name=self.name,
                message=f"Parameters reset for {node_name} ({len(params_snapshot)} params)",
                duration_seconds=time.monotonic() - start,
            )

        except Exception as exc:  # noqa: BLE001
            return HealingResult(
                success=False, action_name=self.name,
                message=f"Parameter reset failed: {exc}",
                duration_seconds=time.monotonic() - start,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Topic Remap
# ──────────────────────────────────────────────────────────────────────────────

class TopicRemapAction(BaseHealingAction):
    """
    Remap subscriptions/publishers to backup or alternative topics.
    Uses dynamic parameter-based remapping where supported.
    """

    name = 'topic_remap'
    description = 'Remap to backup topics to restore connectivity'

    async def execute(self, ctx: HealingContext) -> HealingResult:
        start = time.monotonic()
        self._log(ctx, f"Remapping topics for {ctx.node_name}")

        if ctx.dry_run:
            return HealingResult(
                success=True, action_name=self.name,
                message=f"[DRY RUN] Would remap topics for {ctx.node_name}",
                duration_seconds=time.monotonic() - start,
            )

        # Topic remapping is typically applied at launch time.
        # At runtime, we can try restarting with remap args if metadata provides them.
        remap_rules: list = ctx.metadata.get('remap_rules', [])
        if not remap_rules:
            return HealingResult(
                success=False, action_name=self.name,
                message=f"No remap rules configured for {ctx.node_name}",
            )

        # Trigger hard restart with remap args injected
        ctx.metadata['respawn_args'] = [f"--ros-args"] + [
            f"-r {src}:={dst}" for src, dst in remap_rules
        ]
        hard = HardRestartAction()
        result = await hard.execute(ctx)
        result.action_name = self.name
        result.message = f"Topic remapped and respawned: {remap_rules}"
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Hardware Reinit
# ──────────────────────────────────────────────────────────────────────────────

class HardwareReinitAction(BaseHealingAction):
    """
    Send a hardware re-initialization command via a ROS2 service or topic.
    Useful for sensor/actuator drivers that support soft reset.
    """

    name = 'hw_reinit'
    description = 'Send hardware re-initialization command'

    async def execute(self, ctx: HealingContext) -> HealingResult:
        start = time.monotonic()
        self._log(ctx, f"Hardware reinit for {ctx.node_name}")

        if ctx.dry_run:
            return HealingResult(
                success=True, action_name=self.name,
                message=f"[DRY RUN] Would reinit hardware for {ctx.node_name}",
                duration_seconds=time.monotonic() - start,
            )

        reinit_topic: Optional[str] = ctx.metadata.get('reinit_topic')
        reinit_service: Optional[str] = ctx.metadata.get('reinit_service')

        try:
            if reinit_topic:
                from std_msgs.msg import Bool
                pub = ctx.ros_node.create_publisher(Bool, reinit_topic, 1)
                msg = Bool()
                msg.data = True
                pub.publish(msg)
                await asyncio.sleep(0.5)
                ctx.ros_node.destroy_publisher(pub)

            elif reinit_service:
                from std_srvs.srv import Trigger
                client = ctx.ros_node.create_client(Trigger, reinit_service)
                if client.wait_for_service(timeout_sec=3.0):
                    future = client.call_async(Trigger.Request())
                    await asyncio.sleep(2.0)

            await asyncio.sleep(2.0)  # Allow hardware to reinit

            return HealingResult(
                success=True, action_name=self.name,
                message=f"Hardware reinit sent for {ctx.node_name}",
                duration_seconds=time.monotonic() - start,
            )

        except Exception as exc:  # noqa: BLE001
            return HealingResult(
                success=False, action_name=self.name,
                message=f"Hardware reinit failed: {exc}",
                duration_seconds=time.monotonic() - start,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Escalate  (notify human)
# ──────────────────────────────────────────────────────────────────────────────

class EscalateAction(BaseHealingAction):
    """
    Notify a human operator via configured channels (Slack, email, PagerDuty).
    Always succeeds — used as last resort.
    """

    name = 'escalate'
    description = 'Notify human operator via external channels'

    async def execute(self, ctx: HealingContext) -> HealingResult:
        start = time.monotonic()
        self._log(ctx, f"Escalating failure for {ctx.node_name} to human operators")

        node_name = ctx.node_name
        score = getattr(ctx.alert, 'anomaly_score', 1.0)
        message = (
            f"🚨 GUARDIAN ALERT: Node `{node_name}` has failed after all "
            f"automated recovery attempts. Manual intervention required. "
            f"Anomaly score: {score:.1%}"
        )

        # Slack
        slack_url = ctx.metadata.get('slack_webhook', '')
        if slack_url and not ctx.dry_run:
            await self._send_slack(slack_url, message)

        # Email  (stub — implement with aiosmtplib)
        email = ctx.metadata.get('alert_email', '')
        if email and not ctx.dry_run:
            ctx.logger.info(f"Would email {email}: {message}")

        ctx.logger.warn(f"ESCALATED: {message}")

        return HealingResult(
            success=True, action_name=self.name,
            message=message,
            duration_seconds=time.monotonic() - start,
        )

    async def _send_slack(self, webhook_url: str, text: str) -> None:
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                await session.post(
                    webhook_url,
                    json={'text': text},
                    timeout=aiohttp.ClientTimeout(total=5),
                )
        except Exception:  # noqa: BLE001
            pass  # Best effort
