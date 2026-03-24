"""Tests for the screen manager preview lifecycle.

These tests focus on the preview start/stop contract and the
interaction between ScreenManager, CaptureDevice, and Camera.
Hardware (OLED, I2C) is replaced with a fake ScreenWriter.
"""
from typing import Optional

import contextlib
import datetime
import queue

import numpy as np
import PIL.Image
import pytest

from omgrab.cameras import cameras
from omgrab.display import screen_manager


class _FakeScreenWriter:
    """In-memory replacement for ScreenWriter (no I2C hardware)."""

    def __init__(self, available: bool = True, width: int = 128, height: int = 64):
        self._available = available
        self._width = width
        self._height = height
        self.displayed: list[PIL.Image.Image] = []

    @property
    def available(self) -> bool:
        return self._available

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def display(self, image: PIL.Image.Image):
        self.displayed.append(image)

    def clear(self):
        pass

    def cleanup(self):
        pass


class _FakeCamera(cameras.Camera[cameras.RGBFrame]):
    """Fake camera backed by a queue for test control."""

    def __init__(self, frame_queue: queue.Queue[tuple[cameras.RGBFrame, datetime.datetime]]):
        super().__init__(
            cameras.CameraConfig(fps=10, width=640, height=400),
            enforce_frame_timing=False)
        self._frame_queue = frame_queue

    def setup(self):
        pass

    def close(self):
        pass

    def get_next_frame(
            self, timeout_s: Optional[float] = None
            ) -> tuple[cameras.RGBFrame, datetime.datetime]:
        try:
            return self._frame_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise cameras.FrameUnavailableError('No frame') from exc

    def flush_queue(self) -> int:
        flushed = 0
        while True:
            try:
                self._frame_queue.get_nowait()
                flushed += 1
            except queue.Empty:
                break
        return flushed


class _FakeCaptureDevice:
    """Minimal capture device whose preview() tracks enter/exit."""

    def __init__(self):
        self.preview_entered = False
        self.preview_exited = False

    @property
    def label(self) -> str:
        return 'fake'

    @property
    def connected(self) -> bool:
        return True

    @property
    def ready(self) -> bool:
        return True

    @property
    def device_type(self) -> Optional[str]:
        return None

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        return True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    @contextlib.contextmanager
    def preview(self):
        self.preview_entered = True
        try:
            yield self
        finally:
            self.preview_exited = True


def _make_frame(width: int = 640, height: int = 400) -> cameras.RGBFrame:
    """Create a dummy RGB frame."""
    return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)


class TestStartPreview:
    """start_preview() enters the device preview context."""

    def test_enters_device_preview_context(self):
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)
        device = _FakeCaptureDevice()
        frame_queue: queue.Queue = queue.Queue()
        cam = _FakeCamera(frame_queue)
        mgr.set_preview_source(device, cam)

        mgr.start_preview()

        assert device.preview_entered
        assert mgr._preview_active

    def test_raises_when_source_not_set(self):
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)

        with pytest.raises(RuntimeError, match='Preview source not set'):
            mgr.start_preview()

    def test_raises_when_display_unavailable(self):
        writer = _FakeScreenWriter(available=False)
        mgr = screen_manager.ScreenManager(writer=writer)
        device = _FakeCaptureDevice()
        frame_queue: queue.Queue = queue.Queue()
        cam = _FakeCamera(frame_queue)
        mgr.set_preview_source(device, cam)

        with pytest.raises(RuntimeError, match='Display hardware not available'):
            mgr.start_preview()


class TestStopPreview:
    """stop_preview() exits the device preview context."""

    def test_exits_device_preview_context(self):
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)
        device = _FakeCaptureDevice()
        frame_queue: queue.Queue = queue.Queue()
        cam = _FakeCamera(frame_queue)
        mgr.set_preview_source(device, cam)

        mgr.start_preview()
        mgr.stop_preview()

        assert device.preview_exited
        assert not mgr._preview_active

    def test_stop_is_idempotent(self):
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)
        device = _FakeCaptureDevice()
        frame_queue: queue.Queue = queue.Queue()
        cam = _FakeCamera(frame_queue)
        mgr.set_preview_source(device, cam)

        mgr.start_preview()
        mgr.stop_preview()
        mgr.stop_preview()

        assert not mgr._preview_active


class TestRenderPreviewFrame:
    """_render_preview_frame() reads frames and converts to 1-bit images."""

    def test_renders_frame_from_camera(self):
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)
        frame_queue: queue.Queue = queue.Queue()
        cam = _FakeCamera(frame_queue)
        mgr._preview_camera = cam

        frame_queue.put((_make_frame(), datetime.datetime.now()))

        image = mgr._render_preview_frame()
        assert image.mode == '1'
        assert image.size == (128, 64)

    def test_drains_to_freshest_frame(self):
        """When multiple frames are queued, the rendered image uses the last."""
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)
        frame_queue: queue.Queue = queue.Queue()
        cam = _FakeCamera(frame_queue)
        mgr._preview_camera = cam

        # Put a black frame then a white frame.
        black = np.zeros((400, 640, 3), dtype=np.uint8)
        white = np.full((400, 640, 3), 255, dtype=np.uint8)
        now = datetime.datetime.now()
        frame_queue.put((black, now))
        frame_queue.put((white, now))

        image = mgr._render_preview_frame()

        # The white frame should produce a mostly-white 1-bit image.
        pixels = np.array(image)
        white_ratio = pixels.sum() / pixels.size
        assert white_ratio > 0.5

    def test_returns_blank_when_no_frame(self):
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)
        frame_queue: queue.Queue = queue.Queue()
        cam = _FakeCamera(frame_queue)
        mgr._preview_camera = cam

        image = mgr._render_preview_frame()

        assert image.mode == '1'
        assert np.array(image).sum() == 0

    def test_returns_blank_when_no_camera(self):
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)

        image = mgr._render_preview_frame()

        assert image.mode == '1'
        assert np.array(image).sum() == 0


class TestPreviewDoesNotFlushCamera:
    """start_preview() no longer calls flush_queue on the camera.

    The device's preview() context manager is now responsible for
    flushing stale frames before yielding.
    """

    def test_camera_flush_not_called(self):
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)
        device = _FakeCaptureDevice()
        frame_queue: queue.Queue = queue.Queue()
        cam = _FakeCamera(frame_queue)

        # Put a stale frame before starting preview.
        frame_queue.put((_make_frame(), datetime.datetime.now()))

        mgr.set_preview_source(device, cam)
        mgr.start_preview()

        # The stale frame should still be in the queue (not flushed by
        # screen manager — the device is responsible for flushing).
        assert not frame_queue.empty()

        mgr.stop_preview()


class TestShutdownStopsPreview:
    """shutdown() stops an active preview before stopping the worker thread."""

    def test_shutdown_stops_active_preview(self):
        writer = _FakeScreenWriter(available=True)
        mgr = screen_manager.ScreenManager(writer=writer)
        device = _FakeCaptureDevice()
        frame_queue: queue.Queue = queue.Queue()
        cam = _FakeCamera(frame_queue)
        mgr.set_preview_source(device, cam)

        mgr.start_preview()
        assert mgr._preview_active

        mgr.shutdown()

        assert not mgr._preview_active
        assert device.preview_exited
