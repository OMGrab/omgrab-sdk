"""Recording session that owns capture threads and a parallel writer."""

from typing import Any
from typing import Optional

import datetime
import logging
import pathlib
import queue
import threading
import time
from collections.abc import Callable

from omgrab.cameras import cameras
from omgrab.recording import chunked_writer
from omgrab.sensors import sensor as sensor_module

logger = logging.getLogger(__name__)

# Constants
CAPTURE_THREAD_SHUTDOWN_TIMEOUT_S = 2.0
WRITER_DRAIN_QUIET_PERIOD_S = 0.5
FRAME_NOT_AVAILABLE_LOG_INTERVAL_S = 5.0
FRAME_NOT_AVAILABLE_MIN_MISSES_TO_LOG = 5
WRITER_STATUS_LOG_INTERVAL_S = 5.0


class RecordingSession:
    """A single recording session with isolated threads and parallel writer.

    Owns:
    - Per-stream encoder queues (via ChunkedWriter)
    - Capture threads (one per camera/data source)
    - Encoder threads + mux thread (via ChunkedWriter)

    Lifecycle:
    - start(): Start capture threads and parallel writer
    - stop(): Signal capture threads to stop; writer continues draining
    - join(): Wait for all threads to finish
    """

    def __init__(
        self,
        recording_id: str,
        target_cameras: list[cameras.Camera],
        stream_names: list[str],
        stream_configs: dict[str, chunked_writer.VideoStreamConfig],
        spool_dir: pathlib.Path,
        start_chunk_callback: Optional[chunked_writer.StartChunkCallback] = None,
        sensors: list[sensor_module.Sensor] | None = None,
        sensor_stream_names: list[str] | None = None,
        sensor_stream_configs: dict[str, chunked_writer.DataStreamConfig] | None = None,
        chunk_length_s: float = 60.0,
        max_queue_size: int = 400,
    ):
        """Initialize the recording session.

        Args:
            recording_id: Unique ID for this recording (fixed for session lifetime).
            target_cameras: List of cameras to capture from.
            stream_names: List of stream names, one per camera.
            stream_configs: Configuration for each video stream.
            spool_dir: Directory to write video files.
            start_chunk_callback: Optional callback to generate chunk IDs.
                When None, ChunkedWriter generates sequential 5-digit IDs.
            sensors: Optional list of sensors (e.g., IMU).
            sensor_stream_names: Optional list of sensor stream names, one per sensor.
            sensor_stream_configs: Optional configuration for each sensor stream.
            chunk_length_s: Duration of each recording chunk in seconds.
            max_queue_size: Maximum size per encoder queue.
        """
        if len(target_cameras) != len(stream_names):
            raise ValueError(
                f'Mismatch: {len(target_cameras)} cameras, {len(stream_names)} stream names'
            )

        sensors = sensors or []
        sensor_stream_names = sensor_stream_names or []
        if len(sensors) != len(sensor_stream_names):
            raise ValueError(
                f'Mismatch: {len(sensors)} sensors, {len(sensor_stream_names)} sensor stream names'
            )

        self._recording_id = recording_id
        self._cameras = target_cameras
        self._stream_names = stream_names
        self._stream_configs = stream_configs
        self._sensors = sensors
        self._sensor_stream_names = sensor_stream_names
        self._sensor_stream_configs = sensor_stream_configs or {}
        self._spool_dir = spool_dir
        self._start_chunk_callback = start_chunk_callback
        self._chunk_length_s = chunk_length_s
        self._max_queue_size = max_queue_size

        # Thread management
        self._capture_threads: list[threading.Thread] = []
        self._parallel_writer: Optional[chunked_writer.ChunkedWriter] = None
        self._stop_event = threading.Event()
        self._started = False
        self._stopped = False

        # Timing for dropped frame detection
        self._capture_started_at: Optional[datetime.datetime] = None
        self._capture_stopped_at: Optional[datetime.datetime] = None

    @property
    def stopped(self) -> bool:
        """True if the session has been stopped."""
        return self._stopped

    @property
    def recording_id(self) -> str:
        """Get the recording ID for this session."""
        return self._recording_id

    @property
    def recording_length_s(self) -> Optional[float]:
        """Intended recording duration in seconds (capture start to stop signal)."""
        if self._capture_started_at is None or self._capture_stopped_at is None:
            return None
        return (self._capture_stopped_at - self._capture_started_at).total_seconds()

    @property
    def is_alive(self) -> bool:
        """True if any thread is still running."""
        return any(t.is_alive() for t in self._capture_threads)

    def start(self):
        """Start capture threads and parallel writer.

        Raises:
            RuntimeError: If already started.
        """
        if self._started:
            raise RuntimeError('Session already started')
        self._started = True

        logger.info('Starting recording session %s', self._recording_id)

        self._parallel_writer = chunked_writer.ChunkedWriter(
            name=self._recording_id,
            output_directory=self._spool_dir,
            stream_configs=self._stream_configs,
            start_chunk_callback=self._start_chunk_callback,
            sensor_stream_configs=self._sensor_stream_configs,
            chunk_length_s=self._chunk_length_s,
            max_encoder_queue_size=self._max_queue_size,
        )
        self._parallel_writer.start()

        # Start capture threads for cameras (producers)
        for camera, stream_name in zip(self._cameras, self._stream_names, strict=True):
            timeout_s = 5.0 / camera.config.fps
            thread = threading.Thread(
                target=self._run_capture_loop,
                args=(
                    camera,
                    camera.get_next_frame,
                    cameras.FrameUnavailableError,
                    stream_name,
                    timeout_s,
                    'frame',
                ),
                name=f'capture-{stream_name}-{self._recording_id}',
                daemon=True,
            )
            thread.start()
            self._capture_threads.append(thread)

        # Start capture threads for other sensors
        for source, stream_name in zip(self._sensors, self._sensor_stream_names, strict=True):
            thread = threading.Thread(
                target=self._run_capture_loop,
                args=(
                    source,
                    source.get_next_item,
                    sensor_module.SensorDataUnavailableError,
                    stream_name,
                    0.5,
                    'data item',
                ),
                name=f'capture-{stream_name}-{self._recording_id}',
                daemon=True,
            )
            thread.start()
            self._capture_threads.append(thread)

        self._capture_started_at = datetime.datetime.now(datetime.UTC)
        logger.info(
            'Started %d capture threads and parallel writer for session %s',
            len(self._capture_threads),
            self._recording_id,
        )

    def stop(self):
        """Stop the recording session.

        Joins capture threads, then stops the parallel writer (which joins
        encoder threads and merges the final chunk). Blocks until complete.
        """
        if self._stopped:
            return
        self._stopped = True
        self._capture_stopped_at = datetime.datetime.now(datetime.UTC)

        logger.info('Stopping recording session %s', self._recording_id)
        self._stop_event.set()

        # Join capture threads (they should exit quickly once stop_event is set)
        for thread in self._capture_threads:
            if thread.is_alive():
                thread.join(timeout=CAPTURE_THREAD_SHUTDOWN_TIMEOUT_S)
                if thread.is_alive():
                    logger.warning('Capture thread %s did not exit in time', thread.name)

        # Stop the parallel writer (encoder + mux threads drain and shut down)
        if self._parallel_writer:
            self._parallel_writer.stop()

        logger.debug('Session stopped for %s', self._recording_id)

    def join(self, timeout: float | None = None) -> bool:
        """Wait for all threads to finish (including writer drain).

        Args:
            timeout: Maximum time to wait in seconds, or None for no timeout.

        Returns:
            True if all threads finished, False if timeout expired.
        """
        if not self._started:
            return True

        # Ensure stop was called
        if not self._stopped:
            self.stop()

        return not self.is_alive

    def _run_capture_loop(
        self,
        source: cameras.Camera | sensor_module.Sensor,
        read_fn: Callable[..., tuple[Any, datetime.datetime]],
        exception_type: type[Exception],
        stream_name: str,
        timeout_s: float,
        item_label: str,
    ):
        """Capture data from a source and enqueue it for encoding.

        Args:
            source: Context manager that provides the data (camera or sensor).
            read_fn: Callable that reads the next item, accepting timeout_s.
            exception_type: Exception raised when no data is available.
            stream_name: Name of the stream.
            timeout_s: Timeout for each read attempt.
            item_label: Label for log messages (e.g. 'frame', 'data item').
        """
        assert self._parallel_writer is not None
        encoder_queue = self._parallel_writer.get_encoder_queue(stream_name)

        try:
            with source:
                missing_count = 0
                missing_since_log = 0
                last_missing_log_s = time.monotonic()
                last_missing_error: Optional[str] = None

                while not self._stop_event.is_set():
                    try:
                        data, timestamp = read_fn(timeout_s=timeout_s)
                    except exception_type as e:
                        missing_count += 1
                        missing_since_log += 1
                        last_missing_error = str(e)
                        now_s = time.monotonic()
                        if (
                            missing_since_log >= FRAME_NOT_AVAILABLE_MIN_MISSES_TO_LOG
                            and (now_s - last_missing_log_s) >= FRAME_NOT_AVAILABLE_LOG_INTERVAL_S
                        ):
                            logger.warning(
                                'Session %s: missed %d %ss from %s (total: %d)\nLast error: %s',
                                self._recording_id,
                                missing_since_log,
                                item_label,
                                stream_name,
                                missing_count,
                                last_missing_error,
                            )
                            missing_since_log = 0
                            last_missing_log_s = now_s
                            last_missing_error = None
                        continue

                    try:
                        encoder_queue.put((data, timestamp), block=False)
                    except queue.Full:
                        logger.warning(
                            'Session %s: encoder queue full for %s, dropping %s',
                            self._recording_id,
                            stream_name,
                            item_label,
                        )
        except Exception:
            logger.error(
                'Session %s: capture thread crashed for stream %s',
                self._recording_id, stream_name, exc_info=True)

        logger.debug(
            'Capture loop exiting for stream %s in session %s',
            stream_name, self._recording_id)
