"""Guardian Logger — thin wrapper around rclpy logger with emoji prefixes."""


class GuardianLogger:
    """Wraps an rclpy logger with consistent formatting for Guardian messages."""

    def __init__(self, rclpy_logger) -> None:
        self._l = rclpy_logger

    def info(self, msg: str) -> None:
        self._l.info(f"[Guardian] {msg}")

    def warn(self, msg: str) -> None:
        self._l.warning(f"[Guardian] ⚠️  {msg}")

    def error(self, msg: str) -> None:
        self._l.error(f"[Guardian] ❌ {msg}")

    def debug(self, msg: str) -> None:
        self._l.debug(f"[Guardian] {msg}")

    def critical(self, msg: str) -> None:
        self._l.fatal(f"[Guardian] 🚨 {msg}")
