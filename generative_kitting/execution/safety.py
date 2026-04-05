"""
Execution Safety Layer for the Generative Kitting System.

Provides workspace boundary enforcement, gripper force limits,
emergency stop, and action timeout mechanisms.
"""

import signal
import threading
import time
from typing import Optional

from utils.logger import log


class WorkspaceBoundary:
    """
    Validates that all target poses are within the configured
    workspace bounds.  Prevents the robot from moving to
    unreachable or unsafe positions.
    """

    def __init__(self, bounds: dict):
        """
        Parameters
        ----------
        bounds : dict
            Keys: x_min, x_max, y_min, y_max, z_min, z_max
        """
        self.x_min = bounds.get("x_min", -0.2)
        self.x_max = bounds.get("x_max", 0.8)
        self.y_min = bounds.get("y_min", -0.5)
        self.y_max = bounds.get("y_max", 0.5)
        self.z_min = bounds.get("z_min", 0.0)
        self.z_max = bounds.get("z_max", 0.4)

        log.info(
            f"WorkspaceBoundary: X=[{self.x_min}, {self.x_max}], "
            f"Y=[{self.y_min}, {self.y_max}], "
            f"Z=[{self.z_min}, {self.z_max}]"
        )

    def check(self, x: float, y: float, z: float) -> bool:
        """
        Check if a position is within workspace bounds.

        Parameters
        ----------
        x, y, z : float
            Target position coordinates.

        Returns
        -------
        bool
            True if within bounds, False otherwise.
        """
        if not (self.x_min <= x <= self.x_max):
            log.warning(f"X={x:.4f} outside bounds [{self.x_min}, {self.x_max}]")
            return False
        if not (self.y_min <= y <= self.y_max):
            log.warning(f"Y={y:.4f} outside bounds [{self.y_min}, {self.y_max}]")
            return False
        if not (self.z_min <= z <= self.z_max):
            log.warning(f"Z={z:.4f} outside bounds [{self.z_min}, {self.z_max}]")
            return False
        return True

    def clamp(self, x: float, y: float, z: float) -> tuple:
        """
        Clamp a position to workspace bounds.

        Returns
        -------
        tuple of (float, float, float)
            Clamped (x, y, z).
        """
        cx = max(self.x_min, min(self.x_max, x))
        cy = max(self.y_min, min(self.y_max, y))
        cz = max(self.z_min, min(self.z_max, z))
        if (cx, cy, cz) != (x, y, z):
            log.warning(
                f"Position clamped: ({x:.4f},{y:.4f},{z:.4f}) → "
                f"({cx:.4f},{cy:.4f},{cz:.4f})"
            )
        return cx, cy, cz


def check_gripper_force(force: float, max_force: float) -> bool:
    """
    Validate that a gripper force value is within safe limits.

    Parameters
    ----------
    force : float
        Requested grip force in Newtons.
    max_force : float
        Maximum allowed force from config.

    Returns
    -------
    bool
        True if force is within limits.
    """
    if force < 0:
        log.error(f"Negative gripper force: {force}N — rejected")
        return False
    if force > max_force:
        log.warning(f"Gripper force {force}N exceeds limit {max_force}N — clamped")
        return False
    return True


def emergency_stop(robot_controller) -> bool:
    """
    Execute an emergency stop: immediately move the robot to home position.

    Parameters
    ----------
    robot_controller : RobotController
        The active robot controller instance.

    Returns
    -------
    bool
        True if emergency stop succeeded.
    """
    log.critical("🚨 EMERGENCY STOP TRIGGERED — moving to home position")
    try:
        result = robot_controller.move_home()
        if result:
            log.info("Emergency stop complete — robot at home position")
        else:
            log.error("Emergency stop: move_home returned False")
        return result
    except Exception as e:
        log.critical(f"Emergency stop FAILED: {e}")
        return False


class ActionTimeout:
    """
    Context manager that enforces a timeout on action execution.

    Usage
    -----
    with ActionTimeout(30, "pick_object"):
        # execute action ...

    If the action takes longer than the timeout, a TimeoutError is raised.

    Note: On Windows, signal-based timeouts don't work in all contexts,
    so this uses a threading-based approach.
    """

    def __init__(self, timeout_seconds: float = 30.0, action_name: str = "action"):
        self.timeout = timeout_seconds
        self.action_name = action_name
        self._timer: Optional[threading.Timer] = None
        self._timed_out = False

    def __enter__(self):
        self._timed_out = False
        if self.timeout > 0:
            self._timer = threading.Timer(self.timeout, self._on_timeout)
            self._timer.daemon = True
            self._timer.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._timer:
            self._timer.cancel()
        if self._timed_out and exc_type is None:
            raise TimeoutError(
                f"Action '{self.action_name}' timed out after {self.timeout}s"
            )
        return False

    def _on_timeout(self):
        self._timed_out = True
        log.error(
            f"TIMEOUT: Action '{self.action_name}' exceeded {self.timeout}s limit"
        )

    @property
    def is_timed_out(self) -> bool:
        return self._timed_out


class SafetyMonitor:
    """
    Aggregates all safety checks into a single monitor.
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            The ``execution`` section of config.yaml.
        """
        self.boundary = WorkspaceBoundary(config.get("workspace_bounds", {}))
        self.max_gripper_force = config.get("gripper_force", 40.0)
        self.action_timeout = config.get("action_timeout_seconds", 30.0)
        self._estop_requested = False

        log.info("SafetyMonitor initialised")

    def validate_pose(self, x: float, y: float, z: float) -> bool:
        """Check if a target pose is safe."""
        return self.boundary.check(x, y, z)

    def validate_force(self, force: float) -> bool:
        """Check if a gripper force is safe."""
        return check_gripper_force(force, self.max_gripper_force)

    def get_timeout_context(self, action_name: str) -> ActionTimeout:
        """Get an ActionTimeout context manager for an action."""
        return ActionTimeout(self.action_timeout, action_name)

    def request_estop(self):
        """Flag an emergency stop request."""
        self._estop_requested = True
        log.critical("Emergency stop requested via SafetyMonitor")

    @property
    def estop_requested(self) -> bool:
        return self._estop_requested

    def clear_estop(self):
        """Clear the emergency stop flag."""
        self._estop_requested = False
        log.info("Emergency stop flag cleared")
