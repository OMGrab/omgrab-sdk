"""Centralized device status and system information manager.

This module provides a single source of truth for all device state, system metrics,
and operational status. Can be queried by a display module or external consumers.
"""

from typing import TYPE_CHECKING
from typing import Optional
from typing import Protocol

import dataclasses
import datetime
import json
import logging
import pathlib
import threading
import time

from omgrab.runtime import battery_monitor

if TYPE_CHECKING:
    from omgrab.runtime import network_monitor as network_monitor_module
    from omgrab.runtime import recording_manager as recording_manager_module

logger = logging.getLogger(__name__)


class StateProvider(Protocol):
    """Minimal interface for querying workflow state.

    Used by DeviceStatusManager to decouple from the full StateMachine.
    """

    def get_current_state(self) -> str:
        """Get the current workflow state as a string."""
        ...


@dataclasses.dataclass
class StorageInfo:
    """Storage space information.

    Attributes:
        total_bytes: Total storage capacity in bytes.
        used_bytes: Used storage in bytes.
        available_bytes: Available storage in bytes.
        used_percent: Percentage of storage used (0-100).
    """

    total_bytes: int
    used_bytes: int
    available_bytes: int
    used_percent: float


@dataclasses.dataclass
class CPUInfo:
    """CPU metrics.

    Attributes:
        temperature_celsius: CPU temperature in Celsius, or None if unavailable.
        usage_percent: Overall CPU usage percentage (0-100).
        usage_per_core: Per-core usage percentages (0-100).
    """

    temperature_celsius: Optional[float]
    usage_percent: float
    usage_per_core: list[float]

    def __str__(self) -> str:
        """Return human-readable CPU status."""
        core_tmps = (f'{core}: {pct:.1f}%' for core, pct in enumerate(self.usage_per_core))
        return f'CPU: {self.temperature_celsius:.1f}C | [{", ".join(core_tmps)}]'


@dataclasses.dataclass
class MemoryInfo:
    """Memory usage information.

    Attributes:
        total_bytes: Total RAM in bytes.
        used_bytes: Used RAM in bytes.
        available_bytes: Available RAM in bytes.
        used_percent: Percentage of RAM used (0-100).
    """

    total_bytes: int
    used_bytes: int
    available_bytes: int
    used_percent: float

    def __str__(self) -> str:
        """Return human-readable memory status."""
        return (
            f'Memory: {self.used_percent:.1f}% '
            f'({self.used_bytes / 2**30:.2f}GB / {self.total_bytes / 2**30:.2f}GB)'
        )


@dataclasses.dataclass
class NetworkInfo:
    """Network connectivity information.

    Attributes:
        status: Connection state, one of 'offline', 'network_only', or 'online'.
        wifi_ssid: Connected WiFi network name, or None if not on WiFi.
        wifi_signal_strength: Signal strength in dBm (e.g. -62), or None if unavailable.
    """

    status: str
    wifi_ssid: Optional[str]
    wifi_signal_strength: Optional[int]


@dataclasses.dataclass
class RecordingInfo:
    """Active recording information.

    Attributes:
        is_recording: Whether a recording is currently in progress.
        recording_id: Timestamp-based identifier for the active recording, or None.
        duration_seconds: Elapsed recording time in seconds, or None if not recording.
    """

    is_recording: bool
    recording_id: Optional[str]
    duration_seconds: Optional[float]


@dataclasses.dataclass
class BatteryInfo:
    """Battery status information.

    Attributes:
        percent: Battery percentage (0-100).
        voltage_v: Battery voltage in volts.
        current_a: Current draw in amps (positive when charging).
        power_w: Power consumption in watts.
        is_charging: Whether the battery is currently charging.
    """

    percent: float
    voltage_v: float
    current_a: float
    power_w: float
    is_charging: bool

    def __str__(self) -> str:
        """Return human-readable battery status."""
        return (
            f'Battery: {self.percent:.1f}% | '
            f'{self.voltage_v:.2f}V | '
            f'{self.power_w:.2f}W | '
            f'{self.current_a:.3f}A '
            f'({"charging" if self.is_charging else "discharging"})'
        )


@dataclasses.dataclass
class DeviceStatus:
    """Complete device status snapshot.

    Aggregates all subsystem statuses into a single point-in-time reading,
    produced by DeviceStatusManager on each update cycle.

    Attributes:
        device_id: Unique identifier for this device.
        software_version: Currently running software version string.
        uptime_seconds: Seconds since the status manager started.
        state_machine_state: Current workflow state (e.g. 'idle', 'recording').
        storage: Disk space metrics for the spool directory.
        cpu: CPU temperature and usage metrics.
        memory: RAM usage metrics.
        network: Network connectivity and WiFi details.
        recording: Active recording state and duration.
        device_healthy: Whether the device is operating normally.
        device_error: Human-readable error description, or None if healthy.
        battery: Battery metrics, or None if no battery hardware is present.
    """

    device_id: str
    software_version: str
    uptime_seconds: float
    state_machine_state: str
    storage: StorageInfo
    cpu: CPUInfo
    memory: MemoryInfo
    network: NetworkInfo
    recording: RecordingInfo
    device_healthy: bool
    device_error: Optional[str]
    battery: Optional[BatteryInfo]


class SystemMetricsProvider(Protocol):
    """Interface for reading system-level hardware and OS metrics.

    The default implementation reads from Linux /proc and /sys filesystems.
    Inject an alternative for testing or non-Linux development hosts.
    """

    def get_cpu_info(self) -> CPUInfo:
        """Get CPU temperature and usage information."""
        ...

    def get_memory_info(self) -> MemoryInfo:
        """Get memory usage information."""
        ...

    def get_storage_info(self, path: pathlib.Path) -> StorageInfo:
        """Get storage space information for the given path."""
        ...

    def get_wifi_info(self) -> tuple[Optional[str], Optional[int]]:
        """Get WiFi SSID and signal strength in dBm."""
        ...


class DeviceStatusManager:
    """Manages and aggregates device status information from multiple sources."""

    # Default path for status JSON file.
    DEFAULT_STATUS_FILE = pathlib.Path('/data/device_status.json')

    def __init__(
        self,
        device_id: str,
        software_version: str,
        spool_base_dir: pathlib.Path,
        system_metrics: SystemMetricsProvider,
        enable_battery_monitor: bool = True,
        status_file: Optional[pathlib.Path] = None,
    ):
        """Initialize the device status manager.

        Args:
            device_id: Unique device identifier.
            software_version: Current software version.
            spool_base_dir: Base directory for storage calculations.
            system_metrics: Provider for system-level metrics (CPU, memory,
                storage, WiFi).
            enable_battery_monitor: Whether to enable battery monitoring.
            status_file: Path to write status JSON file for external readers.
        """
        self._device_id = device_id
        self._software_version = software_version
        self._spool_base_dir = spool_base_dir
        self._status_file = status_file or self.DEFAULT_STATUS_FILE
        self._system_metrics = system_metrics

        # Component references (set after construction via setters).
        self._state_machine: Optional[StateProvider] = None
        self._recording_manager: Optional[recording_manager_module.RecordingManager] = None
        self._network_monitor: Optional[network_monitor_module.NetworkMonitor] = None

        # Boot time for uptime calculation.
        self._boot_time = time.monotonic()

        # Thread safety.
        self._lock = threading.Lock()

        # Cached status for quick access.
        self._cached_status_dict: dict = {}

        # Worker thread for periodic status updates.
        self._update_interval_s = 1.0
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Battery monitor (optional).
        self._battery_monitor: Optional[battery_monitor.BatteryMonitor] = None
        if enable_battery_monitor:
            try:
                self._battery_monitor = battery_monitor.BatteryMonitor()
                if self._battery_monitor.available:
                    logger.info('Battery monitoring enabled')
                else:
                    logger.info('Battery hardware not detected')
                    self._battery_monitor = None
            except Exception as e:
                logger.warning('Failed to initialize battery monitor: %s', e)
                self._battery_monitor = None

    def set_state_machine(self, state_machine: StateProvider):
        """Set the state machine reference."""
        self._state_machine = state_machine

    def set_recording_manager(self, recording_manager: 'recording_manager_module.RecordingManager'):
        """Set the recording manager reference."""
        self._recording_manager = recording_manager

    def set_network_monitor(self, network_monitor: 'network_monitor_module.NetworkMonitor'):
        """Set the network monitor reference."""
        self._network_monitor = network_monitor

    def start(self):
        """Start the background worker that updates status periodically."""
        if self._worker_thread is not None:
            return

        self._stop_event.clear()
        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name='device-status-worker'
        )
        self._worker_thread.start()
        logger.info('Device status worker started (interval=%.1fs)', self._update_interval_s)

    def shutdown(self):
        """Stop the background worker."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
        logger.info('Device status worker stopped')

    def _worker_loop(self):
        """Periodically compute status and write to disk."""
        while not self._stop_event.is_set():
            try:
                self._update_status()
            except Exception as e:
                logger.warning('Error updating device status: %s', e)

            self._stop_event.wait(timeout=self._update_interval_s)

    def _update_status(self):
        """Compute status, cache it, and write to disk."""
        status = self.get_status()
        status_dict = dataclasses.asdict(status)

        with self._lock:
            self._cached_status_dict = status_dict

        self._write_status_file(status_dict)

    def get_status(self) -> DeviceStatus:
        """Get a complete snapshot of device status.

        Returns:
            DeviceStatus dataclass with all current information.
        """
        with self._lock:
            device_healthy, device_error = self._get_device_health()
            device_status = DeviceStatus(
                device_id=self._device_id,
                software_version=self._software_version,
                uptime_seconds=time.monotonic() - self._boot_time,
                state_machine_state=self._get_state_machine_state(),
                storage=self._system_metrics.get_storage_info(self._spool_base_dir),
                cpu=self._system_metrics.get_cpu_info(),
                memory=self._system_metrics.get_memory_info(),
                network=self._get_network_info(),
                recording=self._get_recording_info(),
                device_healthy=device_healthy,
                device_error=device_error,
                battery=self._get_battery_info(),
            )
            return device_status

    def get_status_dict(self) -> dict:
        """Get the cached status dictionary.

        The status is updated every second by the background worker.
        This method returns the most recent cached value for fast access.

        Returns:
            Dictionary representation of device status.
        """
        with self._lock:
            return self._cached_status_dict.copy()

    def _write_status_file(self, status_dict: dict):
        """Write status to JSON file for external readers (e.g., TUI dashboard).

        Uses atomic write (write to temp, then rename) to prevent partial reads.
        """
        try:
            # Add timestamp for freshness checking.
            status_dict['_updated_at'] = time.time()

            tmp_path = self._status_file.with_suffix('.tmp')
            with open(tmp_path, 'w') as f:
                json.dump(status_dict, f, indent=2)
            tmp_path.rename(self._status_file)
        except Exception as e:
            logger.debug('Failed to write status file: %s', e)

    def _get_state_machine_state(self) -> str:
        """Get current state machine state."""
        if self._state_machine:
            return self._state_machine.get_current_state()
        return 'unknown'

    def _get_network_info(self) -> NetworkInfo:
        """Get network connectivity information.

        Returns:
            NetworkInfo with network status and WiFi details.
        """
        status = 'unknown'
        if self._network_monitor:
            snapshot = self._network_monitor.snapshot
            status = snapshot.status.value

        wifi_ssid, wifi_signal = self._system_metrics.get_wifi_info()

        return NetworkInfo(
            status=status,
            wifi_ssid=wifi_ssid,
            wifi_signal_strength=wifi_signal,
        )

    def _get_recording_info(self) -> RecordingInfo:
        """Get active recording information.

        Returns:
            RecordingInfo with current recording details.
        """
        if self._recording_manager is None:
            return RecordingInfo(
                is_recording=False,
                recording_id=None,
                duration_seconds=None,
            )

        is_recording = self._recording_manager.is_recording
        recording_id = self._recording_manager.active_recording_id

        duration_seconds = None
        started_at = self._recording_manager.recording_started_at
        if is_recording and started_at is not None:
            delta = datetime.datetime.now(datetime.UTC) - started_at
            duration_seconds = delta.total_seconds()

        return RecordingInfo(
            is_recording=is_recording,
            recording_id=recording_id,
            duration_seconds=duration_seconds,
        )

    def _get_device_health(self) -> tuple[bool, Optional[str]]:
        """Get device health status.

        Returns:
            Tuple of (is_healthy, error_message).
        """
        return True, None

    def _get_battery_info(self) -> Optional[BatteryInfo]:
        """Get battery information from battery monitor.

        Returns:
            BatteryInfo with current battery status, or None if unavailable.
        """
        if self._battery_monitor is None:
            return None

        try:
            status = self._battery_monitor.get_status()
            if status is None:
                return None

            return BatteryInfo(
                percent=status.percent,
                voltage_v=status.voltage_v,
                current_a=status.current_a,
                power_w=status.power_w,
                is_charging=status.current_a > 0,
            )
        except Exception as e:
            logger.debug('Failed to get battery info: %s', e)
            return None
