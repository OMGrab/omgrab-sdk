"""OAK-D capture device implementation.

The capture device represents the physical OAK-D hardware and owns the
DepthAI pipeline lifecycle. It can be opened and closed independently of
the cameras that read from its frame queues. Cameras are lightweight
wrappers that represent a particular recording session; the device may
outlive many camera instances across multiple recordings.
"""

from typing import Optional

import abc
import contextlib
import dataclasses
import datetime
import json
import logging
import queue
import threading
import time
from collections.abc import Callable
from collections.abc import Iterator

import cv2
import depthai as dai
import numpy as np

from omgrab.cameras import cameras
from omgrab.cameras import oakd_camera
from omgrab.devices import capture_device
from omgrab.devices import oakd_device_type
from omgrab.sensors import queue_reader_sensor

logger = logging.getLogger(__name__)


_IDLE_LOOP_SLEEP_S = 0.01
_IDLE_READY_CHECK_INTERVAL_S = 0.1
_MIN_STALE_QUEUE_TIMEOUT_S = 0.1
_STARTUP_MIN_STALE_QUEUE_TIMEOUT_S = 1.0


@dataclasses.dataclass(frozen=True)
class PipelineConfig:
    """Configuration for the OAK-D pipeline runner.

    Attributes:
        warmup_frames: Number of initial frames to discard before marking
            the device as ready.
        startup_grace_s: Grace period after pipeline start during which
            stale-frame detection is relaxed.
        shutdown_timeout_s: Timeout for joining the capture thread during
            device shutdown.
    """

    warmup_frames: int = 10
    startup_grace_s: float = 3.0
    shutdown_timeout_s: float = 2.0


_DEFAULT_PIPELINE_CONFIG = PipelineConfig()


def _flush_queue(q: queue.Queue) -> int:
    """Drain all pending items from a queue.

    Returns:
        Number of items flushed.
    """
    flushed = 0
    while True:
        try:
            q.get_nowait()
            flushed += 1
        except queue.Empty:
            break
    return flushed


class _PipelineRunner(abc.ABC):
    """Abstract base class for DepthAI pipeline runners.

    Manages the shared lifecycle of a DepthAI pipeline: thread creation,
    start/stop, readiness tracking, and error handling.  Subclasses only
    need to implement ``_setup_pipeline()`` (which node graph to build)
    and ``_capture_once()`` (how to read and route frames each tick).
    """

    def __init__(self, fps: int, config: PipelineConfig = _DEFAULT_PIPELINE_CONFIG):
        """Initialize shared pipeline runner state.

        Args:
            fps: Frame rate used to compute the stale-frame timeout for
                readiness tracking.
            config: Pipeline configuration.
        """
        self._config = config
        self._pipeline: Optional[dai.Pipeline] = None
        self._capture_thread: Optional[threading.Thread] = None

        self._stop_capture_event = threading.Event()
        self._pipeline_running_event = threading.Event()
        self._capture_thread_error: Optional[BaseException] = None

        self._frame_count_lock = threading.Lock()
        self._ready_condition = threading.Condition(self._frame_count_lock)
        self._ready = False
        self._dropped_frames = 0
        self._total_frames = 0
        self._last_frame_monotonic_s: Optional[float] = None
        self._last_ready_check_monotonic_s: float = 0.0
        self._pipeline_started_monotonic_s: Optional[float] = None
        self._pipeline_start_reference_time: Optional[datetime.datetime] = None

        self._stale_timeout_s = max(_MIN_STALE_QUEUE_TIMEOUT_S, 5.0 / max(1.0, float(fps)))

    @property
    def is_running(self) -> bool:
        """Whether the pipeline is currently running."""
        return self._pipeline_running_event.is_set()

    @property
    def is_ready(self) -> bool:
        """Whether the pipeline is ready (receiving frames recently)."""
        with self._frame_count_lock:
            return self._ready

    @property
    def last_error(self) -> Optional[BaseException]:
        """Last exception raised by the capture thread (if any)."""
        return self._capture_thread_error

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Block until the pipeline is ready to capture frames."""
        with self._ready_condition:
            return self._ready_condition.wait_for(lambda: self._ready, timeout=timeout)

    def start(self):
        """Start the pipeline and capture thread."""
        if self._capture_thread is not None or self._pipeline is not None:
            raise RuntimeError('Pipeline already running.')
        self._stop_capture_event.clear()
        self._pipeline_running_event.clear()
        self._capture_thread_error = None
        try:
            self._setup_pipeline()
            self._reset_counters()
        except Exception:
            self._pipeline = None
            raise
        self._capture_thread = threading.Thread(target=self._capture_thread_target, daemon=True)
        self._capture_thread.start()

    def stop(self):
        """Stop the pipeline and join the capture thread.

        Subclasses should call ``super().stop()`` and then clear any
        DepthAI queue references they own.
        """
        self._stop_capture_event.set()
        capture_thread = self._capture_thread
        if capture_thread is not None:
            capture_thread.join(timeout=self._config.shutdown_timeout_s)
            if capture_thread.is_alive():
                logger.warning(
                    'Capture thread did not exit within %.1fs', self._config.shutdown_timeout_s
                )
        self._capture_thread = None
        self._pipeline = None
        self._pipeline_running_event.clear()
        self._stop_capture_event.clear()
        with self._ready_condition:
            self._ready = False
            self._ready_condition.notify_all()

    @abc.abstractmethod
    def _setup_pipeline(self):
        """Build and store a ``dai.Pipeline`` in ``self._pipeline``."""

    @abc.abstractmethod
    def _capture_once(self):
        """Read one frame (or frame group) from the pipeline and route it."""

    @abc.abstractmethod
    def _on_pipeline_started(self, pipeline):
        """Hook called once the pipeline is running inside the device context.

        Override to perform one-time actions such as reading device EEPROM.

        Args:
            pipeline: The running ``dai.Pipeline`` instance.
        """

    def _got_frame(self):
        """Signal that a frame was successfully received.

        Subclasses should call this from ``_capture_once()`` whenever they
        receive a valid frame so that readiness tracking stays up to date.
        """
        now_monotonic_s = time.monotonic()
        with self._ready_condition:
            self._total_frames += 1
            self._last_frame_monotonic_s = now_monotonic_s
            self._update_ready_locked(now_monotonic_s)

    def _update_ready_locked(self, now_monotonic_s: float):
        """Update readiness based on recent frame activity.

        NOTE: Must be called with the ready condition held.
        """
        effective_stale_timeout_s = self._stale_timeout_s
        if self._pipeline_started_monotonic_s is not None:
            time_since_start_s = now_monotonic_s - self._pipeline_started_monotonic_s
            if time_since_start_s < self._config.startup_grace_s:
                effective_stale_timeout_s = max(
                    effective_stale_timeout_s,
                    _STARTUP_MIN_STALE_QUEUE_TIMEOUT_S,
                )

        frames_recent = (
            self._last_frame_monotonic_s is not None
            and (now_monotonic_s - self._last_frame_monotonic_s) <= effective_stale_timeout_s
        )
        warmed_up = self._total_frames >= self._config.warmup_frames
        new_ready = bool(frames_recent and warmed_up and self.is_running)
        if self._ready == new_ready:
            return
        self._ready = new_ready
        self._ready_condition.notify_all()
        if not new_ready:
            logger.warning(
                'OakD streams stale (>%.2fs), marking device unready', effective_stale_timeout_s
            )
        else:
            logger.info('OakD streams active, marking device ready')

    def _reset_counters(self):
        """Reset frame counters and timing state."""
        self._dropped_frames = 0
        self._total_frames = 0
        self._ready = False
        self._last_frame_monotonic_s = None
        self._last_ready_check_monotonic_s = 0.0
        self._pipeline_started_monotonic_s = None
        self._pipeline_start_reference_time = None

    def _capture_thread_target(self):
        """Top-level target for the capture thread."""
        assert self._pipeline is not None, 'Pipeline not initialized.'
        try:
            self._capture_loop()
        except Exception as e:
            self._capture_thread_error = e
            logger.exception('Capture thread error: %s', e)
        finally:
            self._pipeline_running_event.clear()
            with self._ready_condition:
                self._ready = False
                self._ready_condition.notify_all()
            try:
                self._pipeline.stop()
            except Exception:
                logger.warning('Error stopping pipeline', exc_info=True)

    def _capture_loop(self):
        """Run the pipeline and call ``_capture_once()`` in a loop."""
        assert self._pipeline is not None, 'Pipeline not initialized.'
        with self._pipeline as pipeline:
            pipeline.start()
            self._pipeline_running_event.set()
            with self._ready_condition:
                self._pipeline_started_monotonic_s = time.monotonic()
                self._pipeline_start_reference_time = datetime.datetime.now()

            self._on_pipeline_started(pipeline)

            try:
                while pipeline.isRunning() and not self._stop_capture_event.is_set():
                    self._capture_once()
            finally:
                self._pipeline_running_event.clear()
                with self._ready_condition:
                    self._ready = False
                    self._ready_condition.notify_all()
                try:
                    pipeline.stop()
                except Exception:
                    pass


class _RecordingPipelineRunner(_PipelineRunner):
    """Pipeline runner for synchronized RGB + depth + optional IMU recording."""

    _RGB_STREAM_NAME = 'rgb'
    _DEPTH_STREAM_NAME = 'depth'
    _IMU_STREAM_NAME = 'imu'

    def __init__(
        self,
        rgb_config: cameras.CameraConfig,
        depth_config: cameras.CameraConfig,
        output_rgb_queue: queue.Queue[oakd_camera.TimestampedRGBFrame],
        output_depth_queue: queue.Queue[oakd_camera.TimestampedDepthFrame],
        on_device_type_detected: Optional[Callable] = None,
        imu_rate_hz: int = 0,
        output_imu_queue: Optional[queue.Queue[oakd_camera.TimestampedIMUData]] = None,
        config: PipelineConfig = _DEFAULT_PIPELINE_CONFIG,
    ):
        """Initialize the recording pipeline runner.

        Args:
            rgb_config: Configuration for the RGB camera.
            depth_config: Configuration for the depth camera.
            output_rgb_queue: Queue to push RGB frames to.
            output_depth_queue: Queue to push depth frames to.
            on_device_type_detected: Optional callback invoked with the detected
                CaptureDeviceType after pipeline starts.
            imu_rate_hz: IMU sampling rate in Hz. 0 to disable IMU.
            output_imu_queue: Queue to push serialized IMU data to. Required
                if imu_rate_hz > 0.
            config: Pipeline configuration.
        """
        if rgb_config.fps != depth_config.fps:
            raise ValueError(
                f'RGB and depth framerates must match for synchronized capture. '
                f'Got RGB fps={rgb_config.fps}, depth fps={depth_config.fps}'
            )
        if imu_rate_hz > 0 and output_imu_queue is None:
            raise ValueError('output_imu_queue is required when imu_rate_hz > 0')

        super().__init__(fps=rgb_config.fps, config=config)

        self._rgb_config = rgb_config
        self._depth_config = depth_config
        self._output_rgb_queue = output_rgb_queue
        self._output_depth_queue = output_depth_queue
        self._on_device_type_detected = on_device_type_detected
        self._imu_rate_hz = imu_rate_hz
        self._output_imu_queue = output_imu_queue
        self._imu_enabled = imu_rate_hz > 0

        self._oakd_synced_queue: Optional[dai.MessageQueue] = None

    def stop(self):
        """Stop the recording pipeline and clear synced queue reference."""
        super().stop()
        self._oakd_synced_queue = None

    def _setup_pipeline(self):
        """Setup the DepthAI pipeline with Sync node for temporal alignment."""
        self._pipeline = dai.Pipeline()
        fps = self._rgb_config.fps  # Same fps for both (validated in __init__)

        # RGB width must be multiple of 16 for depth alignment (hardware requirement)
        # Scale height proportionally to maintain aspect ratio
        rgb_width = (self._rgb_config.width // 16) * 16
        rgb_height = int(self._rgb_config.height * rgb_width / self._rgb_config.width)

        # Create RGB camera
        rgb_cam = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = rgb_cam.requestOutput(size=(rgb_width, rgb_height), fps=fps)

        # Create stereo depth node (aligned to RGB camera)
        depth_width = (self._depth_config.width // 16) * 16
        depth_height = int(self._depth_config.height * depth_width / self._depth_config.width)

        # Create stereo depth node with auto-created cameras
        stereo = self._pipeline.create(dai.node.StereoDepth).build(
            autoCreateCameras=True,
            size=(depth_width, depth_height),
            fps=fps,
        )
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        # Explicitly set output size to match the 16-aligned RGB dimensions.
        # This is required because setDepthAlign() uses the RGB camera's native
        # sensor resolution (e.g., 1352 on OAK-D Wide), which may not be a
        # multiple of 16. By explicitly setting the output size, we override
        # the automatic size selection to use our aligned dimensions.
        stereo.setOutputSize(rgb_width, rgb_height)
        stereo.initialConfig.setDepthUnit(
            dai.StereoDepthConfig.AlgorithmControl.DepthUnit.MILLIMETER
        )

        # Create Sync node for temporal alignment
        sync = self._pipeline.create(dai.node.Sync)
        # Set sync threshold based on frame period to allow temporal skew
        # between RGB and depth. Use 1.5x frame period as threshold (e.g.,
        # 62.5ms @ 24fps, 50ms @ 30fps).
        frame_period_ms = 1000.0 / fps
        sync_threshold_ms = int(frame_period_ms * 1.5)
        sync.setSyncThreshold(datetime.timedelta(milliseconds=sync_threshold_ms))
        rgb_out.link(sync.inputs[self._RGB_STREAM_NAME])
        stereo.depth.link(sync.inputs[self._DEPTH_STREAM_NAME])

        # Add IMU node if enabled.
        #
        # The IMUPacket struct only exposes four fields -- acceleroMeter,
        # gyroscope, magneticField, rotationVector -- so enabling multiple
        # sensors in the same category (e.g. ACCELEROMETER_RAW *and*
        # LINEAR_ACCELERATION) would just overwrite the same field.
        #
        # Not all OAK-D variants have the same IMU chip:
        #   - BNO085/086 (OAK-D, OAK-D Pro, OAK-D S2): supports all
        #     sensor types including calibrated, magnetometer, and
        #     rotation vectors.
        #   - BMI270 (OAK-D Wide, OAK-D Lite): only supports
        #     ACCELEROMETER_RAW and GYROSCOPE_RAW.
        #
        # We use the universally supported raw sensors to ensure
        # compatibility across all devices.
        if self._imu_enabled:
            imu = self._pipeline.create(dai.node.IMU)
            imu.enableIMUSensor(
                [
                    dai.IMUSensor.ACCELEROMETER_RAW,
                    dai.IMUSensor.GYROSCOPE_RAW,
                ],
                self._imu_rate_hz,
            )
            # Batch ~(imu_rate / video_fps) readings per report so each sync
            # group gets one IMU message containing all intermediate readings.
            batch_size = max(1, self._imu_rate_hz // fps)
            imu.setBatchReportThreshold(batch_size)
            imu.setMaxBatchReports(10)
            imu.out.link(sync.inputs[self._IMU_STREAM_NAME])

        # Create single output queue for synchronized frames
        self._oakd_synced_queue = sync.out.createOutputQueue()

    def _capture_once(self):
        """Capture synchronized RGB+depth frames from the OakD camera."""

        def log_drop_rate(dropped_frames: int, total_frames: int):
            """Log the drop rate."""
            drop_rate = (dropped_frames / total_frames) * 100
            logger.warning(
                'Dropped %d/%d frames (%.1f%%) - queue full',
                dropped_frames,
                total_frames,
                drop_rate,
            )

        if self._oakd_synced_queue is None:
            raise RuntimeError('OakD synced queue not initialized.')

        if self._oakd_synced_queue.isClosed():
            raise RuntimeError('OakD synced queue closed.')

        now_monotonic_s = time.monotonic()

        # Non-blocking read from synchronized queue
        synced_msg = self._oakd_synced_queue.tryGet()

        if synced_msg is None:
            # Rate-limit readiness checks while idle to reduce CPU churn.
            with self._ready_condition:
                if (
                    now_monotonic_s - self._last_ready_check_monotonic_s
                ) >= _IDLE_READY_CHECK_INTERVAL_S:
                    self._update_ready_locked(now_monotonic_s)
                    self._last_ready_check_monotonic_s = now_monotonic_s
            time.sleep(_IDLE_LOOP_SLEEP_S)
            return

        # Extract synchronized RGB and depth frames from MessageGroup
        rgb_capture = synced_msg[self._RGB_STREAM_NAME]
        depth_capture = synced_msg[self._DEPTH_STREAM_NAME]

        if rgb_capture is None or depth_capture is None:
            logger.warning('Synced message missing RGB or depth frame')
            return

        self._got_frame()

        # Discard early frames during warmup (underexposed, auto-exposure
        # settling, etc.). The device won't be marked ready until warmup
        # completes either, so RecordingManager won't start capture threads
        # until after this period.
        if self._total_frames <= self._config.warmup_frames:
            return

        # Use hardware timestamps from each camera (Sync node aligns them
        # temporally). getTimestamp() returns a timedelta from device boot;
        # convert to absolute datetime via pipeline start reference time.
        rgb_device_timedelta = rgb_capture.getTimestamp()
        depth_device_timedelta = depth_capture.getTimestamp()

        if self._pipeline_start_reference_time is not None:
            rgb_timestamp = self._pipeline_start_reference_time + rgb_device_timedelta
            depth_timestamp = self._pipeline_start_reference_time + depth_device_timedelta
        else:
            rgb_timestamp = datetime.datetime.now()
            depth_timestamp = rgb_timestamp

        # Process RGB frame (flip -1 = 180-degree rotation for upside-down mount)
        rgb_frame = rgb_capture.getCvFrame()
        rgb_frame = cv2.flip(rgb_frame, -1)
        rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)

        # Process depth frame
        depth_frame = depth_capture.getCvFrame()
        depth_frame = cv2.flip(depth_frame, -1)
        depth_frame = depth_frame.astype(np.uint16)

        # Put frames into output queues with their individual timestamps
        rgb_dropped = False
        depth_dropped = False

        try:
            self._output_rgb_queue.put((rgb_frame, rgb_timestamp), block=False)
        except queue.Full:
            rgb_dropped = True

        try:
            self._output_depth_queue.put((depth_frame, depth_timestamp), block=False)
        except queue.Full:
            depth_dropped = True

        # Extract and enqueue IMU data if enabled
        if self._imu_enabled and self._output_imu_queue is not None:
            imu_message = synced_msg[self._IMU_STREAM_NAME]
            if imu_message is not None:
                self._enqueue_imu_data(imu_message)

        # Track drops (count as single frame drop if either was dropped)
        if rgb_dropped or depth_dropped:
            self._dropped_frames += 1
            if self._dropped_frames % 30 == 0:
                log_drop_rate(self._dropped_frames, self._total_frames)

    def _on_pipeline_started(self, pipeline):
        """Read device type from EEPROM once the pipeline is running."""
        if self._on_device_type_detected is None:
            return
        device_type = oakd_device_type.detect_from_pipeline(pipeline)
        if device_type is not None:
            logger.info('Detected OAK-D device type: %s', device_type.value)
            self._on_device_type_detected(device_type)

    def _enqueue_imu_data(self, imu_message: dai.IMUData):
        """Batch IMU readings from a sync group and enqueue as a single item.

        All readings in the message are serialized as a JSON array and pushed
        to the IMU queue as one item. Each reading includes a 'dt' field with
        the millisecond offset from the first reading's timestamp, preserving
        per-reading timing while drastically reducing queue throughput.

        The camera is physically installed upside-down (180-degree rotation
        around the optical axis) for cable management. The video frames are
        corrected with cv2.flip(); here we apply the equivalent correction
        to IMU vectors: negate X and Y, keep Z unchanged.

        Args:
            imu_message: DepthAI IMUData message containing batched readings.
        """
        readings = []
        first_timestamp: Optional[datetime.datetime] = None

        for packet in imu_message.packets:
            accel = packet.acceleroMeter
            gyro = packet.gyroscope

            # Convert device timedelta to absolute datetime.
            # The timestamp lives on the individual report objects, not on
            # IMUPacket itself. Use the accelerometer report's host-synced
            # timestamp (getTimestamp, not getTimestampDevice) to match the
            # same clock domain as video frame timestamps.
            imu_timedelta = accel.getTimestamp()
            if self._pipeline_start_reference_time is not None:
                imu_timestamp = self._pipeline_start_reference_time + imu_timedelta
            else:
                imu_timestamp = datetime.datetime.now()

            if first_timestamp is None:
                first_timestamp = imu_timestamp

            dt_ms = (imu_timestamp - first_timestamp).total_seconds() * 1000

            # Negate X and Y for 180-degree physical rotation
            # (matches the cv2.flip(0) + cv2.flip(1) applied to video).
            readings.append(
                {
                    'dt': round(dt_ms, 3),
                    'a': [round(-accel.x, 6), round(-accel.y, 6), round(accel.z, 6)],
                    'g': [round(-gyro.x, 6), round(-gyro.y, 6), round(gyro.z, 6)],
                }
            )

        if not readings:
            return

        payload = json.dumps(readings, separators=(',', ':')).encode('utf-8')
        if self._output_imu_queue is None or first_timestamp is None:
            return
        try:
            self._output_imu_queue.put((payload, first_timestamp), block=False)
        except queue.Full:
            pass  # IMU drops are silent; video drops already track the issue


class _PreviewPipelineRunner(_PipelineRunner):
    """Lightweight pipeline runner for RGB-only camera preview.

    Creates only an RGB camera node (no stereo depth, sync, or IMU) and
    pushes frames to a single output queue.  Intended for low-latency
    OLED preview / alignment checks.
    """

    def __init__(
        self,
        rgb_config: cameras.CameraConfig,
        output_rgb_queue: queue.Queue[oakd_camera.TimestampedRGBFrame],
        config: PipelineConfig = _DEFAULT_PIPELINE_CONFIG,
    ):
        """Initialize the preview pipeline runner.

        Args:
            rgb_config: Configuration for the RGB camera (fps, width, height).
            output_rgb_queue: Queue to push processed RGB frames to.
            config: Pipeline configuration.
        """
        super().__init__(fps=rgb_config.fps, config=config)
        self._rgb_config = rgb_config
        self._output_rgb_queue = output_rgb_queue
        self._rgb_output_queue: Optional[dai.MessageQueue] = None

    def stop(self):
        """Stop the preview pipeline and clear DepthAI queue reference."""
        super().stop()
        self._rgb_output_queue = None

    def _on_pipeline_started(self, pipeline):
        """No-op for preview pipeline."""
        pass

    def _setup_pipeline(self):
        """Setup a minimal DepthAI pipeline with only the RGB camera."""
        self._pipeline = dai.Pipeline()
        fps = self._rgb_config.fps
        width = (self._rgb_config.width // 16) * 16
        height = int(self._rgb_config.height * width / self._rgb_config.width)

        rgb_cam = self._pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        rgb_out = rgb_cam.requestOutput(size=(width, height), fps=fps)
        self._rgb_output_queue = rgb_out.createOutputQueue()

    def _capture_once(self):
        """Read one RGB frame and push it to the output queue."""
        if self._rgb_output_queue is None:
            raise RuntimeError('Preview output queue not initialized.')
        if self._rgb_output_queue.isClosed():
            raise RuntimeError('Preview output queue closed.')

        msg = self._rgb_output_queue.tryGet()
        if msg is None:
            time.sleep(_IDLE_LOOP_SLEEP_S)
            return

        if self._pipeline_start_reference_time is not None:
            timestamp = self._pipeline_start_reference_time + msg.getTimestamp()
        else:
            timestamp = datetime.datetime.now()

        frame = msg.getCvFrame()
        frame = cv2.flip(frame, -1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            self._output_rgb_queue.put((frame, timestamp), block=False)
        except queue.Full:
            pass  # Drop silently; consumer grabs freshest frame anyway.


class OakDCaptureDevice(capture_device.CaptureDevice):
    """OakD device used to populate two camera queues from one source."""

    def __init__(
        self,
        rgb_config: cameras.CameraConfig,
        depth_config: cameras.CameraConfig,
        max_queue_size: int = 200,
        imu_rate_hz: int = 0,
        preview_config: Optional[cameras.CameraConfig] = None,
        pipeline_config: PipelineConfig = _DEFAULT_PIPELINE_CONFIG,
    ):
        """Initialize the OakD device.

        Args:
            rgb_config: Configuration for the RGB camera.
            depth_config: Configuration for the depth camera.
            max_queue_size: Maximum queue size for frame/data buffers.
            imu_rate_hz: IMU sampling rate in Hz. 0 to disable IMU.
            preview_config: Optional configuration for the preview camera.
                If provided, enables the preview() context manager and
                get_preview_camera(). Uses a lightweight RGB-only pipeline.
            pipeline_config: Configuration for the pipeline runner.
        """
        self._rgb_config = rgb_config
        self._depth_config = depth_config
        self._max_queue_size = max_queue_size
        self._imu_rate_hz = imu_rate_hz
        self._output_rgb_queue = queue.Queue[oakd_camera.TimestampedRGBFrame](
            maxsize=max_queue_size
        )
        self._output_depth_queue = queue.Queue[oakd_camera.TimestampedDepthFrame](
            maxsize=max_queue_size
        )
        self._output_imu_queue: Optional[queue.Queue[oakd_camera.TimestampedIMUData]] = None
        if imu_rate_hz > 0:
            self._output_imu_queue = queue.Queue[oakd_camera.TimestampedIMUData](
                maxsize=max_queue_size
            )
        self._device_type: Optional[oakd_device_type.OakDDeviceType] = None
        self._device_type_lock = threading.Lock()
        self._runner = _RecordingPipelineRunner(
            rgb_config=self._rgb_config,
            depth_config=self._depth_config,
            output_rgb_queue=self._output_rgb_queue,
            output_depth_queue=self._output_depth_queue,
            on_device_type_detected=self._set_device_type,
            imu_rate_hz=imu_rate_hz,
            output_imu_queue=self._output_imu_queue,
            config=pipeline_config,
        )

        # Preview support (optional, enabled when preview_config is provided).
        self._preview_config = preview_config
        self._preview_runner: Optional[_PreviewPipelineRunner] = None
        self._output_preview_queue: Optional[queue.Queue[oakd_camera.TimestampedRGBFrame]] = None
        if preview_config is not None:
            self._output_preview_queue = queue.Queue[oakd_camera.TimestampedRGBFrame](
                maxsize=max_queue_size
            )
            self._preview_runner = _PreviewPipelineRunner(
                rgb_config=preview_config,
                output_rgb_queue=self._output_preview_queue,
                config=pipeline_config,
            )

    def _set_device_type(self, device_type: oakd_device_type.OakDDeviceType):
        """Callback invoked by the pipeline runner when device type is detected."""
        with self._device_type_lock:
            self._device_type = device_type

    @property
    def label(self) -> str:
        """Human-readable label for this device."""
        return 'oakd'

    @property
    def connected(self) -> bool:
        """Check if the OakD camera is connected."""
        if self._runner.is_running:
            return self._runner.is_ready
        try:
            devices = dai.Device.getAllAvailableDevices()
            return len(devices) > 0
        except Exception as e:
            logger.error('Error checking connection: %s', e)
            return False

    @property
    def ready(self) -> bool:
        """Check if the OakD camera is ready."""
        return self._runner.is_ready

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until the OakD camera is ready to capture frames."""
        return self._runner.wait_until_ready(timeout=timeout)

    @property
    def rgb_queue(self) -> queue.Queue[oakd_camera.TimestampedRGBFrame]:
        """Get the rgb queue."""
        return self._output_rgb_queue

    @property
    def depth_queue(self) -> queue.Queue[oakd_camera.TimestampedDepthFrame]:
        """Get the depth queue."""
        return self._output_depth_queue

    def get_rgb_camera(self) -> oakd_camera.OakDRGBCamera:
        """Get the RGB camera."""
        return oakd_camera.OakDRGBCamera(self._rgb_config, self._output_rgb_queue)

    def get_depth_camera(self) -> oakd_camera.OakDDepthCamera:
        """Get the depth camera."""
        return oakd_camera.OakDDepthCamera(self._depth_config, self._output_depth_queue)

    @property
    def imu_queue(self) -> Optional[queue.Queue[oakd_camera.TimestampedIMUData]]:
        """Get the IMU data queue, or None if IMU is disabled."""
        return self._output_imu_queue

    @property
    def imu_enabled(self) -> bool:
        """Whether IMU data collection is enabled."""
        return self._imu_rate_hz > 0

    def get_imu_source(self) -> queue_reader_sensor.QueueReaderSensor:
        """Get a data source for reading IMU data.

        Returns:
            QueueReaderSensor wrapping the IMU queue.

        Raises:
            RuntimeError: If IMU is not enabled.
        """
        if self._output_imu_queue is None:
            raise RuntimeError('IMU is not enabled (imu_rate_hz=0)')
        return queue_reader_sensor.QueueReaderSensor(self._output_imu_queue)

    def get_preview_camera(self) -> oakd_camera.OakDRGBCamera:
        """Get the preview RGB camera.

        The returned camera reads from the preview queue and only produces
        frames while the ``preview()`` context manager is active.

        Returns:
            An OakDRGBCamera reading from the preview queue.

        Raises:
            RuntimeError: If preview_config was not provided at construction.
        """
        if self._preview_config is None or self._output_preview_queue is None:
            raise RuntimeError('Preview is not enabled (preview_config was not provided)')
        return oakd_camera.OakDRGBCamera(self._preview_config, self._output_preview_queue)

    @contextlib.contextmanager
    def preview(self) -> Iterator['OakDCaptureDevice']:
        """Context manager that runs a lightweight RGB-only preview pipeline.

        The preview pipeline is mutually exclusive with the recording
        pipeline (``__enter__``). Starting preview while recording is
        active (or vice-versa) raises ``RuntimeError``.

        Yields:
            OakDCaptureDevice: This device instance.

        Raises:
            PreviewUnavailableError: If preview_config was not provided.
            RuntimeError: If the recording pipeline is already running.
        """
        if self._preview_runner is None:
            raise capture_device.PreviewUnavailableError(
                'Preview is not enabled (preview_config was not provided)'
            )
        if self._runner.is_running:
            raise RuntimeError('Cannot start preview while recording pipeline is running.')
        self._preview_runner.start()
        try:
            assert self._output_preview_queue is not None
            _flush_queue(self._output_preview_queue)
            yield self
        finally:
            self._preview_runner.stop()

    @property
    def device_type(self) -> Optional[str]:
        """Return the detected OAK-D device type as a string.

        The device type is read from EEPROM during pipeline startup. Returns None
        if the device has not been started or if type detection failed.

        Returns:
            One of the OakDDeviceType values (e.g. 'oakd_pro_wide'), or None.
        """
        with self._device_type_lock:
            dt = self._device_type
        if dt is None:
            return None
        return dt.value

    def __enter__(self):
        """Setup the oakd capture device."""
        if self._runner.is_running:
            raise RuntimeError('OakD camera already in use.')
        if self._preview_runner is not None and self._preview_runner.is_running:
            raise RuntimeError('Cannot start recording while preview pipeline is running.')
        self._runner.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the oakd capture device."""
        self._runner.stop()
