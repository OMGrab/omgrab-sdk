"""Recording manager that owns capture device and recording sessions."""
from typing import Optional
from typing import Protocol
from typing import runtime_checkable

import dataclasses
import datetime
import logging
import pathlib
import threading
from collections.abc import Callable
from contextlib import ExitStack

from omgrab.cameras import cameras
from omgrab.cameras import queue_reader_camera
from omgrab.devices import capture_device as capture_device_module
from omgrab.recording import chunked_writer
from omgrab.recording import manifest
from omgrab.recording import py_av_writer
from omgrab.runtime import recording_session
from omgrab.sensors import queue_reader_sensor
from omgrab.sensors import sensor as sensor_module

logger = logging.getLogger(__name__)


# Type aliases
RecordingID = str

# Callback type for device unhealthy notification
OnDeviceUnhealthyCallback = Callable[[], None]

# Callback type for recording complete notification
OnRecordingCompleteCallback = Callable[[pathlib.Path], None]

@dataclasses.dataclass(frozen=True)
class RecordingConfig:
    """Configuration for the recording manager.

    Attributes:
        device_ready_timeout_s: Max time to wait for the capture device
            to become ready before starting a recording.
        health_check_interval_s: Interval between device health checks
            during recording.
        session_join_timeout_s: Timeout for joining finished recording
            session threads during shutdown.
        chunk_length_s: Duration of each recording chunk in seconds.
        max_queue_size: Maximum size per encoder queue.
    """

    device_ready_timeout_s: float = 7.5
    health_check_interval_s: float = 1.0
    session_join_timeout_s: float = 10.0
    chunk_length_s: float = 60.0
    max_queue_size: int = 400


@runtime_checkable
class RecordingController(Protocol):
    """Minimal interface for controlling recording start/stop.

    Used by StateMachine to decouple from the full RecordingManager.
    """

    def start_recording(self) -> bool:
        """Start a recording. Returns True on success."""
        ...

    def stop_recording(self):
        """Stop the active recording."""
        ...


def _make_recording_name() -> str:
    """Generate a recording name from the current UTC timestamp."""
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H-%M-%SZ')


class RecordingManager:
    """Manages recording sessions and capture device lifecycles.

    Owns:
    - Capture devices (opened/closed per recording)
    - Active recording session (at most one at a time)
    - Finished sessions pending cleanup

    The StateMachine calls start_recording/stop_recording; this class
    handles all the threading and device management internally.
    """

    def __init__(
            self,
            devices: list[capture_device_module.CaptureDevice],
            target_cameras: list[cameras.Camera],
            stream_names: list[str],
            stream_configs: dict[str, chunked_writer.VideoStreamConfig],
            spool_dir: pathlib.Path,
            output_dir: pathlib.Path,
            config: Optional[RecordingConfig] = None,
            on_device_unhealthy: Optional[OnDeviceUnhealthyCallback] = None,
            on_recording_complete: Optional[OnRecordingCompleteCallback] = None,
            sensors: list[sensor_module.Sensor] | None = None,
            sensor_stream_names: list[str] | None = None,
            sensor_stream_configs: dict[str, chunked_writer.DataStreamConfig] | None = None):
        """Initialize the recording manager.

        Args:
            devices: Capture devices to use for recording. All must be
                connected for recording to start, and all are monitored
                for health during recording.
            target_cameras: List of cameras to capture from.
            stream_names: List of stream names, one per camera.
            stream_configs: Configuration for each video stream.
            spool_dir: Directory to write temporary video chunks.
            output_dir: Directory to write merged final recordings.
            config: Recording configuration.
            on_device_unhealthy: Optional callback invoked when any device
                becomes unhealthy during recording.
            on_recording_complete: Optional callback invoked when a recording
                has been merged. Called with the path to the merged MKV file.
            sensors: Optional list of sensors (e.g., IMU).
            sensor_stream_names: Optional list of sensor stream names, one per sensor.
            sensor_stream_configs: Optional configuration for each sensor stream.
        """
        self._config = config or RecordingConfig()
        self._devices = devices
        self._cameras = target_cameras
        self._stream_names = stream_names
        self._stream_configs = stream_configs
        self._sensors = sensors or []
        self._sensor_stream_names = sensor_stream_names or []
        self._sensor_stream_configs = sensor_stream_configs or {}
        self._spool_dir = spool_dir
        self._output_dir = output_dir
        self._on_device_unhealthy = on_device_unhealthy
        self._on_recording_complete = on_recording_complete

        self._lock = threading.Lock()
        self._active_session: Optional[recording_session.RecordingSession] = None
        self._active_recording_id: Optional[RecordingID] = None
        self._recording_started_at: Optional[datetime.datetime] = None
        self._finished_sessions: list[recording_session.RecordingSession] = []

        self._device_stack: Optional[ExitStack] = None

        self._health_monitor_stop = threading.Event()
        self._health_monitor_thread: Optional[threading.Thread] = None

    @property
    def is_recording(self) -> bool:
        """True if there is an active session currently capturing."""
        with self._lock:
            return self._active_session is not None and not self._active_session.stopped

    @property
    def active_recording_id(self) -> Optional[RecordingID]:
        """Get the ID of the active recording, or None if not recording."""
        with self._lock:
            return self._active_recording_id

    @property
    def recording_started_at(self) -> Optional[datetime.datetime]:
        """Get the start time of the active recording, or None if not recording."""
        with self._lock:
            return self._recording_started_at

    def set_on_device_unhealthy(self, callback: Optional[OnDeviceUnhealthyCallback]):
        """Set the callback for device unhealthy events.

        This allows setting the callback after construction, which is needed
        when there are circular dependencies (e.g., StateMachine needs
        RecordingManager and vice versa).

        Args:
            callback: Callback to invoke when device becomes unhealthy during
                recording, or None to clear the callback.
        """
        self._on_device_unhealthy = callback

    def start_recording(self) -> bool:
        """Start a new recording session.

        Opens the capture device, creates a new session with timestamp-based
        recording name, and starts threads.

        Returns:
            True if recording started successfully, False otherwise.
        """
        self._cleanup_finished_sessions()

        with self._lock:
            if self._active_session is not None:
                logger.warning('Cannot start recording: session already active')
                return False

        # Check all devices are connected before attempting to open
        for device in self._devices:
            if not device.connected:
                logger.warning(
                    'Cannot start recording: device %s not connected',
                    device.label)
                return False

        # Open all capture devices
        try:
            logger.debug('Opening capture devices')
            self._device_stack = ExitStack()
            for device in self._devices:
                self._device_stack.enter_context(device)
        except Exception as e:
            logger.error('Failed to open capture device: %s', e)
            self._close_device()
            return False

        # Wait for all devices to be ready
        try:
            for device in self._devices:
                device_ready = device.wait_until_ready(
                    timeout=self._config.device_ready_timeout_s)
                if not device_ready:
                    logger.warning('Device %s not ready after %.1fs',
                                   device.label,
                                   self._config.device_ready_timeout_s)
                    self._close_device()
                    return False
        except Exception as e:
            logger.error('Error waiting for device ready: %s', e)
            self._close_device()
            return False

        self._flush_device_queues()

        # Generate timestamp-based recording name
        recording_name = _make_recording_name()
        actual_start_time = datetime.datetime.now(datetime.UTC)

        # Create recording subdirectory for chunks
        recording_dir = self._spool_dir / recording_name
        recording_dir.mkdir(parents=True, exist_ok=True)

        # Create and start session
        session = recording_session.RecordingSession(
            recording_id=recording_name,
            target_cameras=self._cameras,
            stream_names=self._stream_names,
            stream_configs=self._stream_configs,
            spool_dir=recording_dir,
            sensors=self._sensors,
            sensor_stream_names=self._sensor_stream_names,
            sensor_stream_configs=self._sensor_stream_configs,
            chunk_length_s=self._config.chunk_length_s,
            max_queue_size=self._config.max_queue_size,
        )

        try:
            session.start()
        except Exception as e:
            logger.error('Failed to start recording session: %s', e)
            self._close_device()
            return False

        with self._lock:
            self._active_session = session
            self._active_recording_id = recording_name
            self._recording_started_at = actual_start_time

        self._start_health_monitor()

        logger.info('Recording started: %s', recording_name)
        return True

    def stop_recording(self):
        """Stop the current recording session.

        Stops capture threads immediately, then merges chunks into a final
        MKV file. The session is moved to finished_sessions for later cleanup.
        """
        # Stop health monitor first
        self._stop_health_monitor()

        with self._lock:
            session = self._active_session
            recording_id = self._active_recording_id
            started_at = self._recording_started_at
            if session is None:
                logger.warning('No active session to stop')
                return
            self._active_session = None
            self._active_recording_id = None
            self._recording_started_at = None
            self._finished_sessions.append(session)

        # Stop capture threads (fast)
        session.stop()
        stopped_at = datetime.datetime.now(datetime.UTC)
        self._close_device()

        if recording_id:
            recording_dir = self._spool_dir / recording_id
            output_path = self._output_dir / f'{recording_id}.mkv'
            try:
                py_av_writer.merge_recording_chunks(recording_dir, output_path)
                logger.info('Recording merged: %s', output_path)
                self._write_manifest(
                    recording_id, started_at, stopped_at, output_path)
                if self._on_recording_complete is not None:
                    self._on_recording_complete(output_path)
            except Exception:
                logger.error('Failed to merge recording %s (dir=%s, output=%s)',
                             recording_id, recording_dir, output_path,
                             exc_info=True)

        logger.info('Recording stopped: %s', recording_id)

    def shutdown(self):
        """Shutdown the manager: stop active session and join all threads."""
        logger.info('Shutting down recording manager')

        # Stop active session if any
        with self._lock:
            has_active = self._active_session is not None

        if has_active:
            self.stop_recording()

        # Join all finished sessions
        with self._lock:
            sessions_to_join = self._finished_sessions[:]
            self._finished_sessions.clear()

        for s in sessions_to_join:
            logger.debug('Joining session %s', s.recording_id)
            s.join(timeout=self._config.session_join_timeout_s)
            if s.is_alive:
                logger.warning('Session %s did not fully terminate', s.recording_id)

        logger.info('Recording manager shutdown complete')

    def _close_device(self):
        """Close the capture device if open."""
        if self._device_stack is not None:
            try:
                self._device_stack.close()
            except Exception as e:
                logger.error('Error closing capture device: %s', e)
            finally:
                self._device_stack = None

    def _flush_device_queues(self):
        """Flush any stale frames/data from device queues.

        This prevents data from a previous recording from appearing
        in the new recording's queue.
        """
        for cam in self._cameras:
            if isinstance(cam, queue_reader_camera.QueueReaderCamera):
                flushed = cam.flush_queue()
                if flushed > 0:
                    logger.debug('Flushed %d stale frames from camera queue', flushed)
        for source in self._sensors:
            if isinstance(source, queue_reader_sensor.QueueReaderSensor):
                flushed = source.flush_queue()
                if flushed > 0:
                    logger.debug('Flushed %d stale data items from data source queue', flushed)

    def _write_manifest(
            self,
            recording_id: str,
            started_at: Optional[datetime.datetime],
            stopped_at: Optional[datetime.datetime],
            output_path: pathlib.Path):
        """Write a recording manifest JSON alongside the merged MKV.

        Args:
            recording_id: Timestamp-based recording identifier.
            started_at: UTC datetime when capture began.
            stopped_at: UTC datetime when capture ended.
            output_path: Path to the merged MKV file.
        """
        duration_s = None
        if started_at is not None and stopped_at is not None:
            duration_s = (stopped_at - started_at).total_seconds()

        devices = [
            manifest.DeviceInfo(
                label=device.label,
                device_type=device.device_type,
            )
            for device in self._devices
        ]

        recording_manifest = manifest.build_manifest(
            recording_id=recording_id,
            started_at=started_at,
            stopped_at=stopped_at,
            duration_s=duration_s,
            devices=devices,
            stream_configs_map=self._stream_configs,
            sensor_stream_configs_map=self._sensor_stream_configs,
            output_file=output_path.name,
        )

        try:
            manifest.write_manifest(recording_manifest, self._output_dir)
        except Exception:
            logger.error('Failed to write manifest for %s',
                         recording_id, exc_info=True)

    def _cleanup_finished_sessions(self):
        """Clean up finished sessions that have fully terminated.

        This is called lazily before starting a new recording.
        """
        with self._lock:
            still_alive = []
            for session in self._finished_sessions:
                if session.is_alive:
                    # Try a non-blocking join
                    session.join(timeout=0.0)
                    if session.is_alive:
                        still_alive.append(session)
                    else:
                        logger.debug('Cleaned up finished session %s', session.recording_id)
                else:
                    logger.debug('Cleaned up finished session %s', session.recording_id)
            self._finished_sessions = still_alive

        if still_alive:
            logger.debug('%d sessions still draining', len(still_alive))

    def _start_health_monitor(self):
        """Start the device health monitor thread."""
        self._health_monitor_stop.clear()
        self._health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True,
            name='device-health-monitor'
        )
        self._health_monitor_thread.start()

    def _stop_health_monitor(self):
        """Stop the device health monitor thread."""
        self._health_monitor_stop.set()
        if self._health_monitor_thread is not None:
            self._health_monitor_thread.join(timeout=2.0)
            self._health_monitor_thread = None

    def _health_monitor_loop(self):
        """Monitor device health and stop recording if device becomes unhealthy.

        Uses the cheap `ready` check for polling, not the expensive `connected` check.
        """
        logger.info('Health monitor started')
        while not self._health_monitor_stop.wait(timeout=self._config.health_check_interval_s):
            with self._lock:
                session = self._active_session
            if session is None:
                # Recording stopped normally
                break

            all_ready = all(device.ready for device in self._devices)
            if not all_ready:
                logger.warning('Capture device became unhealthy (not ready)')
                self._health_monitor_stop.set()

                # Run callback in a separate thread to avoid self-join
                # deadlock: the callback calls stop_recording() which
                # joins this health monitor thread.
                if self._on_device_unhealthy is not None:
                    callback = self._on_device_unhealthy

                    def _safe_callback(cb=callback):
                        try:
                            cb()
                        except Exception:
                            logger.error(
                                'Device-unhealthy callback raised',
                                exc_info=True)

                    threading.Thread(
                        target=_safe_callback,
                        daemon=True,
                        name='device-unhealthy-callback'
                    ).start()
                break

        logger.info('Health monitor stopped')
