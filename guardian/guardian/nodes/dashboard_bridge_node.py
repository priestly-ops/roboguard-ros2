"""
Dashboard Bridge Node
=====================
Bridges ROS2 topics to a WebSocket server that the React dashboard connects to.
Forwards NodeHealth, SystemAlert, and HealingAction messages as JSON.
Also accepts commands from the dashboard (trigger healing, suppress alerts).
"""

import asyncio
import json
import threading
from typing import Set

import websockets
import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor

from guardian_msgs.msg import NodeHealth, SystemAlert, HealingAction
from guardian.utils.logger import GuardianLogger


def _msg_to_dict(msg) -> dict:
    """Best-effort conversion of a ROS2 message to a dict for JSON serialisation."""
    result = {}
    for slot in msg.get_fields_and_field_types().keys():
        val = getattr(msg, slot, None)
        if hasattr(val, 'get_fields_and_field_types'):
            result[slot] = _msg_to_dict(val)
        elif isinstance(val, (list, tuple)):
            result[slot] = list(val)
        else:
            result[slot] = val
    return result


class DashboardBridgeNode(Node):
    """WebSocket bridge between ROS2 topics and the React dashboard."""

    NODE_NAME = 'guardian_dashboard_bridge'

    def __init__(self) -> None:
        super().__init__(self.NODE_NAME)
        self.declare_parameter('websocket_port', 8765)
        self.declare_parameter('websocket_host', '0.0.0.0')

        self.log = GuardianLogger(self.get_logger())
        self._port: int = self.get_parameter('websocket_port').value
        self._host: str = self.get_parameter('websocket_host').value

        self._clients: Set[websockets.WebSocketServerProtocol] = set()
        self._lock = threading.Lock()

        # Subscriptions
        self.create_subscription(NodeHealth, '/guardian/metrics', self._on_metrics, 10)
        self.create_subscription(SystemAlert, '/guardian/alerts', self._on_alert, 10)
        self.create_subscription(HealingAction, '/guardian/healing_events', self._on_healing, 10)
        self.create_subscription(SystemAlert, '/guardian/risk_scores', self._on_score, 10)

        # Start WebSocket server in background thread
        self._ws_thread = threading.Thread(target=self._run_ws_server, daemon=True)
        self._ws_thread.start()
        self.log.info(f"Dashboard bridge WebSocket server on ws://{self._host}:{self._port}")

    def _run_ws_server(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        server = websockets.serve(self._ws_handler, self._host, self._port)
        loop.run_until_complete(server)
        loop.run_forever()

    async def _ws_handler(self, ws, path) -> None:
        with self._lock:
            self._clients.add(ws)
        self.log.info(f"Dashboard client connected: {ws.remote_address}")
        try:
            async for message in ws:
                await self._handle_command(message)
        finally:
            with self._lock:
                self._clients.discard(ws)
            self.log.info(f"Dashboard client disconnected: {ws.remote_address}")

    async def _handle_command(self, message: str) -> None:
        try:
            cmd = json.loads(message)
            if cmd.get('type') == 'trigger_healing':
                node_name = cmd.get('node_name', '')
                self.log.info(f"Dashboard requested healing for {node_name}")
                # TODO: call /guardian/trigger_healing service
        except (json.JSONDecodeError, KeyError):
            pass

    def _broadcast(self, payload: dict) -> None:
        """Broadcast a JSON payload to all connected dashboard clients."""
        msg_str = json.dumps(payload)
        with self._lock:
            clients = list(self._clients)
        if not clients:
            return
        # Fire-and-forget broadcast on the WS event loop
        for ws in clients:
            asyncio.run_coroutine_threadsafe(
                ws.send(msg_str), asyncio.get_event_loop()
            )

    def _on_metrics(self, msg: NodeHealth) -> None:
        self._broadcast({'type': 'metrics', 'data': _msg_to_dict(msg)})

    def _on_alert(self, msg: SystemAlert) -> None:
        self._broadcast({'type': 'alert', 'data': _msg_to_dict(msg)})

    def _on_healing(self, msg: HealingAction) -> None:
        self._broadcast({'type': 'healing_event', 'data': _msg_to_dict(msg)})

    def _on_score(self, msg: SystemAlert) -> None:
        self._broadcast({'type': 'risk_score', 'data': _msg_to_dict(msg)})


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DashboardBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
