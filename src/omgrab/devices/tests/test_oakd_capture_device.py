"""Tests for OAK-D pipeline runners and capture device.

Heavily mocks depthai (dai) to avoid requiring real hardware.
"""
from typing import Optional

import datetime
import json
import queue
import threading
import time

import numpy as np
import pytest

from omgrab.cameras import cameras
from omgrab.cameras import oakd_camera
from omgrab.devices import capture_device
from omgrab.devices import oakd_capture_device
from omgrab.devices import oakd_device_type
from omgrab.sensors import queue_reader_sensor


class _FakeDaiCapture:
    """Fake DepthAI capture message (replaces dai.ImgFrame)."""

    def __init__(
            self,
            frame: np.ndarray,
            timestamp: datetime.timedelta = datetime.timedelta(seconds=1)):
        self._frame = frame
        self._timestamp = timestamp

    def getCvFrame(self) -> np.ndarray:
        return self._frame.copy()

    def getTimestamp(self) -> datetime.timedelta:
        return self._timestamp


class _FakeIMUReport:
    """Minimal fake for a single IMU accelerometer/gyroscope report."""

    def __init__(self, x: float, y: float, z: float, ts: datetime.timedelta):
        self.x = x
        self.y = y
        self.z = z
        self._ts = ts

    def getTimestamp(self) -> datetime.timedelta:
        return self._ts


class _FakeIMUPacket:
    """Minimal fake for dai.IMUPacket."""

    def __init__(self, accel: _FakeIMUReport, gyro: _FakeIMUReport):
        self.acceleroMeter = accel
        self.gyroscope = gyro


class _FakeIMUData:
    """Minimal fake for dai.IMUData containing a list of packets."""

    def __init__(self, packets: list[_FakeIMUPacket]):
        self.packets = packets


class _FakeMessageGroup(dict):
    """dict-like stand-in for dai.MessageGroup returned by Sync node."""


class _FakeSyncedQueue:
    """Fake synced queue that returns a message once, then None."""

    def __init__(self, msg):
        self._msg = msg
        self._returned = False

    def tryGet(self):
        if not self._returned:
            self._returned = True
            return self._msg
        return None

    def isClosed(self):
        return False


@pytest.fixture(autouse=True)
def stub_cv2_resize(monkeypatch):
    """Replace cv2.resize with numpy-based resize."""
    resize_calls: list[tuple[int, int]] = []

    def fake_resize(frame, size):
        w, h = size
        resize_calls.append((w, h))
        if frame.ndim == 3:
            return np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)
        return np.zeros((h, w), dtype=frame.dtype)

    monkeypatch.setattr(oakd_capture_device.cv2, 'resize', fake_resize)
    return resize_calls


class TestRecordingPipelineRunnerFrameRouting:
    """Test _capture_once() directly by manually calling it with a fake queue."""

    def _make_runner(
            self,
            rgb_queue: Optional[queue.Queue] = None,
            depth_queue: Optional[queue.Queue] = None,
            imu_rate_hz: int = 0,
            imu_queue: Optional[queue.Queue] = None,
            skip_warmup: bool = True,
            ) -> oakd_capture_device._RecordingPipelineRunner:
        """Create a runner with dummy pipeline state so _capture_once() works.

        Args:
            rgb_queue: Output queue for RGB frames.
            depth_queue: Output queue for depth frames.
            imu_rate_hz: IMU sampling rate.
            imu_queue: Output queue for IMU data.
            skip_warmup: If True, pre-advance the frame counter past the
                warmup period so frames are immediately enqueued.
        """
        cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        rq = rgb_queue or queue.Queue(maxsize=10)
        dq = depth_queue or queue.Queue(maxsize=10)
        iq = imu_queue

        runner = oakd_capture_device._RecordingPipelineRunner(
            rgb_config=cfg,
            depth_config=cfg,
            output_rgb_queue=rq,
            output_depth_queue=dq,
            imu_rate_hz=imu_rate_hz,
            output_imu_queue=iq,
        )
        # Simulate pipeline running so readiness tracking can fire.
        runner._pipeline_running_event.set()
        runner._pipeline_started_monotonic_s = time.monotonic()
        runner._pipeline_start_reference_time = datetime.datetime(
            2025, 7, 1, 12, 0, 0)
        if skip_warmup:
            runner._total_frames = oakd_capture_device.PipelineConfig().warmup_frames
        return runner

    def test_rgb_and_depth_pushed_to_queues(self, monkeypatch):
        rgb_q: queue.Queue = queue.Queue(maxsize=10)
        depth_q: queue.Queue = queue.Queue(maxsize=10)
        runner = self._make_runner(
            rgb_queue=rgb_q, depth_queue=depth_q)

        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_frame = np.zeros((480, 640), dtype=np.uint16)

        fake_msg = _FakeMessageGroup({
            oakd_capture_device._RecordingPipelineRunner._RGB_STREAM_NAME:
                _FakeDaiCapture(rgb_frame, datetime.timedelta(seconds=1)),
            oakd_capture_device._RecordingPipelineRunner._DEPTH_STREAM_NAME:
                _FakeDaiCapture(depth_frame, datetime.timedelta(seconds=1)),
        })

        # Stub cv2 operations used in _capture_once.
        monkeypatch.setattr(oakd_capture_device.cv2, 'flip', lambda f, code: f)
        monkeypatch.setattr(
            oakd_capture_device.cv2, 'cvtColor', lambda f, code: f)

        runner._oakd_synced_queue = _FakeSyncedQueue(fake_msg)
        runner._capture_once()

        assert not rgb_q.empty()
        assert not depth_q.empty()

    def test_drops_frame_when_queue_full(self, monkeypatch):
        rgb_q: queue.Queue = queue.Queue(maxsize=1)
        depth_q: queue.Queue = queue.Queue(maxsize=1)
        runner = self._make_runner(
            rgb_queue=rgb_q, depth_queue=depth_q)

        monkeypatch.setattr(oakd_capture_device.cv2, 'flip', lambda f, code: f)
        monkeypatch.setattr(oakd_capture_device.cv2, 'cvtColor', lambda f, code: f)

        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_frame = np.zeros((480, 640), dtype=np.uint16)

        # Pre-fill queues.
        rgb_q.put(('filler', datetime.datetime.now()))
        depth_q.put(('filler', datetime.datetime.now()))

        fake_msg = _FakeMessageGroup({
            oakd_capture_device._RecordingPipelineRunner._RGB_STREAM_NAME:
                _FakeDaiCapture(rgb_frame),
            oakd_capture_device._RecordingPipelineRunner._DEPTH_STREAM_NAME:
                _FakeDaiCapture(depth_frame),
        })

        runner._oakd_synced_queue = _FakeSyncedQueue(fake_msg)
        runner._capture_once()  # Should not raise despite full queues.

        assert runner._dropped_frames == 1

    def test_none_synced_message_does_not_crash(self, monkeypatch):
        runner = self._make_runner()

        class _EmptySyncedQueue:
            def tryGet(self):
                return None

            def isClosed(self):
                return False

        sleep_calls: list[float] = []

        def fake_sleep(duration: float):
            sleep_calls.append(duration)

        monkeypatch.setattr(oakd_capture_device.time, 'sleep', fake_sleep)

        runner._oakd_synced_queue = _EmptySyncedQueue()
        runner._capture_once()  # Should not raise.

        assert sleep_calls  # Should have slept during idle.

    def test_closed_queue_raises_runtime_error(self):
        runner = self._make_runner()

        class _ClosedSyncedQueue:
            def isClosed(self):
                return True

        runner._oakd_synced_queue = _ClosedSyncedQueue()

        with pytest.raises(RuntimeError, match='closed'):
            runner._capture_once()

    def test_none_queue_raises_runtime_error(self):
        runner = self._make_runner()
        runner._oakd_synced_queue = None

        with pytest.raises(RuntimeError, match='not initialized'):
            runner._capture_once()

    def test_timestamps_use_pipeline_reference_time(self, monkeypatch):
        rgb_q: queue.Queue = queue.Queue(maxsize=10)
        depth_q: queue.Queue = queue.Queue(maxsize=10)
        runner = self._make_runner(
            rgb_queue=rgb_q, depth_queue=depth_q)

        monkeypatch.setattr(oakd_capture_device.cv2, 'flip', lambda f, code: f)
        monkeypatch.setattr(oakd_capture_device.cv2, 'cvtColor', lambda f, code: f)

        device_offset = datetime.timedelta(seconds=5)
        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_frame = np.zeros((480, 640), dtype=np.uint16)

        fake_msg = _FakeMessageGroup({
            oakd_capture_device._RecordingPipelineRunner._RGB_STREAM_NAME:
                _FakeDaiCapture(rgb_frame, device_offset),
            oakd_capture_device._RecordingPipelineRunner._DEPTH_STREAM_NAME:
                _FakeDaiCapture(depth_frame, device_offset),
        })

        runner._oakd_synced_queue = _FakeSyncedQueue(fake_msg)
        runner._capture_once()

        _, rgb_ts = rgb_q.get_nowait()
        _, depth_ts = depth_q.get_nowait()

        expected_ts = runner._pipeline_start_reference_time + device_offset
        assert rgb_ts == expected_ts
        assert depth_ts == expected_ts

    def test_warmup_frames_discarded(self, monkeypatch):
        """Frames during warmup should be counted but not enqueued."""
        rgb_q: queue.Queue = queue.Queue(maxsize=10)
        depth_q: queue.Queue = queue.Queue(maxsize=10)
        runner = self._make_runner(
            rgb_queue=rgb_q, depth_queue=depth_q, skip_warmup=False)

        monkeypatch.setattr(oakd_capture_device.cv2, 'flip', lambda f, code: f)
        monkeypatch.setattr(oakd_capture_device.cv2, 'cvtColor', lambda f, code: f)

        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_frame = np.zeros((480, 640), dtype=np.uint16)

        class _ReusableSyncedQueue:
            def __init__(self, msg):
                self._msg = msg

            def tryGet(self):
                return self._msg

            def isClosed(self):
                return False

        fake_msg = _FakeMessageGroup({
            oakd_capture_device._RecordingPipelineRunner._RGB_STREAM_NAME:
                _FakeDaiCapture(rgb_frame),
            oakd_capture_device._RecordingPipelineRunner._DEPTH_STREAM_NAME:
                _FakeDaiCapture(depth_frame),
        })
        runner._oakd_synced_queue = _ReusableSyncedQueue(fake_msg)

        # Call _capture_once() for warmup frames — queues should stay empty.
        for _ in range(oakd_capture_device.PipelineConfig().warmup_frames):
            runner._capture_once()
        assert rgb_q.empty()
        assert depth_q.empty()

        # The next frame should be enqueued.
        runner._capture_once()
        assert not rgb_q.empty()
        assert not depth_q.empty()

    def test_missing_rgb_or_depth_in_message_group(self):
        """When synced message has None for rgb or depth, log and return."""
        runner = self._make_runner()
        initial_total = runner._total_frames

        fake_msg = _FakeMessageGroup({
            oakd_capture_device._RecordingPipelineRunner._RGB_STREAM_NAME: None,
            oakd_capture_device._RecordingPipelineRunner._DEPTH_STREAM_NAME: None,
        })

        runner._oakd_synced_queue = _FakeSyncedQueue(fake_msg)
        runner._capture_once()  # Should not raise.

        # No frames should have been counted beyond the initial value.
        assert runner._total_frames == initial_total


class TestRecordingPipelineRunnerIMU:
    """Test IMU data routing through _capture_once()."""

    def _make_imu_runner(
            self,
            imu_rate_hz: int = 100,
            ) -> tuple[oakd_capture_device._RecordingPipelineRunner, queue.Queue]:
        """Create a runner with IMU enabled and warmup skipped.

        Returns:
            Tuple of (runner, imu_queue).
        """
        cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        imu_q: queue.Queue = queue.Queue(maxsize=10)

        runner = oakd_capture_device._RecordingPipelineRunner(
            rgb_config=cfg,
            depth_config=cfg,
            output_rgb_queue=queue.Queue(maxsize=10),
            output_depth_queue=queue.Queue(maxsize=10),
            imu_rate_hz=imu_rate_hz,
            output_imu_queue=imu_q,
        )
        runner._pipeline_running_event.set()
        runner._pipeline_started_monotonic_s = time.monotonic()
        runner._pipeline_start_reference_time = datetime.datetime(
            2025, 7, 1, 12, 0, 0)
        runner._total_frames = oakd_capture_device.PipelineConfig().warmup_frames
        return runner, imu_q

    def test_imu_data_enqueued(self, monkeypatch):
        runner, imu_q = self._make_imu_runner()

        monkeypatch.setattr(oakd_capture_device.cv2, 'flip', lambda f, code: f)
        monkeypatch.setattr(oakd_capture_device.cv2, 'cvtColor', lambda f, code: f)

        accel = _FakeIMUReport(1.0, 2.0, 3.0, datetime.timedelta(seconds=1))
        gyro = _FakeIMUReport(0.1, 0.2, 0.3, datetime.timedelta(seconds=1))
        imu_data = _FakeIMUData([_FakeIMUPacket(accel, gyro)])

        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_frame = np.zeros((480, 640), dtype=np.uint16)

        fake_msg = _FakeMessageGroup({
            oakd_capture_device._RecordingPipelineRunner._RGB_STREAM_NAME:
                _FakeDaiCapture(rgb_frame),
            oakd_capture_device._RecordingPipelineRunner._DEPTH_STREAM_NAME:
                _FakeDaiCapture(depth_frame),
            oakd_capture_device._RecordingPipelineRunner._IMU_STREAM_NAME:
                imu_data,
        })

        runner._oakd_synced_queue = _FakeSyncedQueue(fake_msg)
        runner._capture_once()

        assert not imu_q.empty()
        payload_bytes, _ = imu_q.get_nowait()
        readings = json.loads(payload_bytes)
        assert len(readings) == 1
        # X and Y should be negated for 180-degree rotation.
        assert readings[0]['a'] == [-1.0, -2.0, 3.0]
        assert readings[0]['g'] == [-0.1, -0.2, 0.3]

    def test_imu_empty_packets_not_enqueued(self, monkeypatch):
        runner, imu_q = self._make_imu_runner()

        monkeypatch.setattr(oakd_capture_device.cv2, 'flip', lambda f, code: f)
        monkeypatch.setattr(oakd_capture_device.cv2, 'cvtColor', lambda f, code: f)

        imu_data = _FakeIMUData([])  # No packets.

        rgb_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_frame = np.zeros((480, 640), dtype=np.uint16)

        fake_msg = _FakeMessageGroup({
            oakd_capture_device._RecordingPipelineRunner._RGB_STREAM_NAME:
                _FakeDaiCapture(rgb_frame),
            oakd_capture_device._RecordingPipelineRunner._DEPTH_STREAM_NAME:
                _FakeDaiCapture(depth_frame),
            oakd_capture_device._RecordingPipelineRunner._IMU_STREAM_NAME:
                imu_data,
        })

        runner._oakd_synced_queue = _FakeSyncedQueue(fake_msg)
        runner._capture_once()

        assert imu_q.empty()


class TestRecordingPipelineRunnerValidation:

    def test_mismatched_fps_raises_value_error(self):
        rgb_cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        depth_cfg = cameras.CameraConfig(fps=15, width=640, height=480)

        with pytest.raises(ValueError, match='framerates must match'):
            oakd_capture_device._RecordingPipelineRunner(
                rgb_config=rgb_cfg,
                depth_config=depth_cfg,
                output_rgb_queue=queue.Queue(),
                output_depth_queue=queue.Queue(),
            )

    def test_imu_without_queue_raises_value_error(self):
        cfg = cameras.CameraConfig(fps=30, width=640, height=480)

        with pytest.raises(ValueError, match='output_imu_queue is required'):
            oakd_capture_device._RecordingPipelineRunner(
                rgb_config=cfg,
                depth_config=cfg,
                output_rgb_queue=queue.Queue(),
                output_depth_queue=queue.Queue(),
                imu_rate_hz=100,
                output_imu_queue=None,
            )


class TestPipelineRunnerReadiness:
    """Test readiness tracking in _PipelineRunner via _got_frame()."""

    def _make_base_runner(self, fps: int = 30) -> oakd_capture_device._RecordingPipelineRunner:
        """Create a runner and fake-start it for readiness tests."""
        cfg = cameras.CameraConfig(fps=fps, width=640, height=480)
        runner = oakd_capture_device._RecordingPipelineRunner(
            rgb_config=cfg,
            depth_config=cfg,
            output_rgb_queue=queue.Queue(),
            output_depth_queue=queue.Queue(),
        )
        # Simulate pipeline running.
        runner._pipeline_running_event.set()
        runner._pipeline_started_monotonic_s = time.monotonic()
        return runner

    def test_not_ready_initially(self):
        runner = self._make_base_runner()
        assert not runner.is_ready

    def test_not_ready_during_warmup(self):
        runner = self._make_base_runner()
        for _ in range(oakd_capture_device.PipelineConfig().warmup_frames - 1):
            runner._got_frame()
        assert not runner.is_ready

    def test_ready_after_warmup(self):
        runner = self._make_base_runner()
        for _ in range(oakd_capture_device.PipelineConfig().warmup_frames):
            runner._got_frame()
        assert runner.is_ready

    def test_total_frames_incremented(self):
        runner = self._make_base_runner()
        runner._got_frame()
        runner._got_frame()
        assert runner._total_frames == 2

    def test_not_ready_when_pipeline_not_running(self):
        runner = self._make_base_runner()
        runner._pipeline_running_event.clear()  # "stop" the pipeline.
        runner._got_frame()
        assert not runner.is_ready

    def test_wait_until_ready_returns_true_when_ready(self):
        runner = self._make_base_runner()
        for _ in range(oakd_capture_device.PipelineConfig().warmup_frames):
            runner._got_frame()
        assert runner.wait_until_ready(timeout=0.1) is True

    def test_wait_until_ready_returns_false_on_timeout(self):
        runner = self._make_base_runner()
        assert runner.wait_until_ready(timeout=0.05) is False


class TestPipelineRunnerStartStop:

    def test_start_raises_if_already_running(self):
        cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        runner = oakd_capture_device._RecordingPipelineRunner(
            rgb_config=cfg,
            depth_config=cfg,
            output_rgb_queue=queue.Queue(),
            output_depth_queue=queue.Queue(),
        )
        # Fake that a thread is already assigned (simulating prior start).
        runner._capture_thread = threading.Thread(target=lambda: None)

        with pytest.raises(RuntimeError, match='already running'):
            runner.start()


class TestOakDCaptureDeviceAccessors:
    """Test OakDCaptureDevice properties and camera/source accessors.

    These tests avoid starting the pipeline — they only verify wiring.
    """

    def _make_device(
            self,
            imu_rate_hz: int = 0,
            ) -> oakd_capture_device.OakDCaptureDevice:
        """Create a device without starting anything."""
        rgb_cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        depth_cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        return oakd_capture_device.OakDCaptureDevice(
            rgb_config=rgb_cfg,
            depth_config=depth_cfg,
            imu_rate_hz=imu_rate_hz,
        )

    def test_rgb_queue_is_queue(self):
        dev = self._make_device()
        assert isinstance(dev.rgb_queue, queue.Queue)

    def test_depth_queue_is_queue(self):
        dev = self._make_device()
        assert isinstance(dev.depth_queue, queue.Queue)

    def test_get_rgb_camera_returns_oakd_rgb(self):
        dev = self._make_device()
        cam = dev.get_rgb_camera()
        assert isinstance(cam, oakd_camera.OakDRGBCamera)

    def test_get_depth_camera_returns_oakd_depth(self):
        dev = self._make_device()
        cam = dev.get_depth_camera()
        assert isinstance(cam, oakd_camera.OakDDepthCamera)

    def test_device_type_none_before_start(self):
        dev = self._make_device()
        assert dev.device_type is None

    def test_device_type_set_by_callback(self):
        dev = self._make_device()
        dev._set_device_type(oakd_device_type.OakDDeviceType.OAKD_PRO_WIDE)
        assert dev.device_type == 'oakd_pro_wide'

    def test_imu_disabled_by_default(self):
        dev = self._make_device(imu_rate_hz=0)
        assert not dev.imu_enabled
        assert dev.imu_queue is None

    def test_imu_enabled_when_rate_set(self):
        dev = self._make_device(imu_rate_hz=100)
        assert dev.imu_enabled
        assert dev.imu_queue is not None

    def test_get_imu_source_raises_when_disabled(self):
        dev = self._make_device(imu_rate_hz=0)
        with pytest.raises(RuntimeError, match='not enabled'):
            dev.get_imu_source()

    def test_get_imu_source_returns_sensor(self):
        dev = self._make_device(imu_rate_hz=100)
        src = dev.get_imu_source()
        assert isinstance(src, queue_reader_sensor.QueueReaderSensor)

    def test_ready_is_false_before_start(self):
        dev = self._make_device()
        assert not dev.ready


class TestOakDCaptureDevicePreview:
    """Test preview-related accessors and guards."""

    def _make_device_with_preview(self) -> oakd_capture_device.OakDCaptureDevice:
        rgb_cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        depth_cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        preview_cfg = cameras.CameraConfig(fps=15, width=320, height=240)
        return oakd_capture_device.OakDCaptureDevice(
            rgb_config=rgb_cfg,
            depth_config=depth_cfg,
            preview_config=preview_cfg,
        )

    def _make_device_without_preview(self) -> oakd_capture_device.OakDCaptureDevice:
        rgb_cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        depth_cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        return oakd_capture_device.OakDCaptureDevice(
            rgb_config=rgb_cfg,
            depth_config=depth_cfg,
        )

    def test_get_preview_camera_raises_when_not_configured(self):
        dev = self._make_device_without_preview()
        with pytest.raises(RuntimeError, match='not enabled'):
            dev.get_preview_camera()

    def test_get_preview_camera_returns_camera(self):
        dev = self._make_device_with_preview()
        cam = dev.get_preview_camera()
        assert isinstance(cam, oakd_camera.OakDRGBCamera)

    def test_preview_raises_when_not_configured(self):
        dev = self._make_device_without_preview()
        with (
            pytest.raises(capture_device.PreviewUnavailableError, match='not enabled'),
            dev.preview(),
        ):
            pass


class TestParseOakDDeviceType:
    """Supplementary tests beyond what test_oakd_device_type.py already covers."""

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match='Unknown'):
            oakd_device_type.parse_device_type('')

    def test_none_raises(self):
        with pytest.raises(ValueError, match='Unknown'):
            oakd_device_type.parse_device_type(None)

    @pytest.mark.parametrize(
        ('product_name', 'expected'),
        [
            ('OAK-D PRO W', oakd_device_type.OakDDeviceType.OAKD_PRO_WIDE),
            ('OAK-D_PRO_W-97', oakd_device_type.OakDDeviceType.OAKD_PRO_WIDE),
            ('some-LITE-device', oakd_device_type.OakDDeviceType.OAKD_LITE),
            ('OAK-D-W-456', oakd_device_type.OakDDeviceType.OAKD_WIDE),
            ('WIDE-ANGLE-OAK', oakd_device_type.OakDDeviceType.OAKD_WIDE),
        ],
    )
    def test_additional_product_name_variants(self, product_name, expected):
        assert oakd_device_type.parse_device_type(product_name) == expected

    def test_pro_wide_takes_priority_over_wide(self):
        """PRO-W should match PRO_WIDE, not WIDE."""
        result = oakd_device_type.parse_device_type('OAK-D-PRO-W-FOO-WIDE')
        assert result == oakd_device_type.OakDDeviceType.OAKD_PRO_WIDE
