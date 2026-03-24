"""Tests for QueueReaderCamera: frame reading, resize, flush, and error mapping."""
import datetime
import queue

import numpy as np
import pytest

from omgrab.cameras import cameras
from omgrab.cameras import queue_reader_camera


@pytest.fixture(autouse=True)
def stub_cv2_resize(monkeypatch):
    """Replace cv2.resize with a numpy-based resize so tests don't need OpenCV.

    Returns a list that captures (target_width, target_height) for each call.
    """
    resize_calls: list[tuple[int, int]] = []

    def fake_resize(frame, size):
        w, h = size
        resize_calls.append((w, h))
        # Return a frame of the target shape (h, w, ...) or (h, w).
        if frame.ndim == 3:
            return np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)
        return np.zeros((h, w), dtype=frame.dtype)

    monkeypatch.setattr(queue_reader_camera.cv2, 'resize', fake_resize)
    return resize_calls


def _make_camera(
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        ) -> tuple[
            queue_reader_camera.QueueReaderCamera,
            queue.Queue]:
    """Create a QueueReaderCamera with a fresh queue."""
    q: queue.Queue = queue.Queue()
    cfg = cameras.CameraConfig(fps=fps, width=width, height=height)
    cam: queue_reader_camera.QueueReaderCamera = queue_reader_camera.QueueReaderCamera(cfg, q)
    return cam, q


class TestGetNextFrame:

    def test_returns_frame_and_timestamp(self):
        cam, q = _make_camera(width=4, height=3)
        ts = datetime.datetime(2025, 6, 15, 12, 0, 0)
        frame_in = np.zeros((6, 8, 3), dtype=np.uint8)
        q.put((frame_in, ts))

        frame_out, ts_out = cam.get_next_frame(timeout_s=1.0)

        assert ts_out == ts
        assert frame_out.shape == (3, 4, 3)

    def test_resizes_to_config_dimensions(self, stub_cv2_resize):
        cam, q = _make_camera(width=320, height=240)
        q.put((np.zeros((480, 640, 3), dtype=np.uint8), datetime.datetime.now()))

        cam.get_next_frame(timeout_s=1.0)

        assert stub_cv2_resize == [(320, 240)]

    def test_skips_resize_when_frame_already_matches(self, stub_cv2_resize):
        """No cv2.resize call when input frame already matches config dims."""
        cam, q = _make_camera(width=320, height=240)
        frame_in = np.full((240, 320, 3), 42, dtype=np.uint8)
        q.put((frame_in, datetime.datetime.now()))

        frame_out, _ = cam.get_next_frame(timeout_s=1.0)

        assert stub_cv2_resize == []
        assert np.array_equal(frame_out, frame_in)

    def test_raises_frame_not_available_on_empty_queue(self):
        cam, _q = _make_camera()

        with pytest.raises(cameras.FrameUnavailableError, match='Queue empty'):
            cam.get_next_frame(timeout_s=0.01)

    def test_exception_chains_original_queue_empty(self):
        """FrameUnavailableError should chain the underlying queue.Empty via __cause__."""
        cam, _q = _make_camera()

        with pytest.raises(cameras.FrameUnavailableError) as exc_info:
            cam.get_next_frame(timeout_s=0.01)

        assert exc_info.value.__cause__ is not None
        assert isinstance(exc_info.value.__cause__, queue.Empty)

    def test_raises_frame_not_available_with_zero_timeout(self):
        cam, _q = _make_camera()

        with pytest.raises(cameras.FrameUnavailableError):
            cam.get_next_frame(timeout_s=0)

    def test_multiple_frames_returned_in_order(self):
        cam, q = _make_camera(width=2, height=2)
        ts1 = datetime.datetime(2025, 1, 1, 0, 0, 0)
        ts2 = datetime.datetime(2025, 1, 1, 0, 0, 1)
        q.put((np.full((2, 2, 3), 10, dtype=np.uint8), ts1))
        q.put((np.full((2, 2, 3), 20, dtype=np.uint8), ts2))

        _, t1 = cam.get_next_frame(timeout_s=1.0)
        _, t2 = cam.get_next_frame(timeout_s=1.0)

        assert t1 == ts1
        assert t2 == ts2


class TestFrameTimingDisabled:
    """QueueReaderCamera sets enforce_frame_timing=False in __init__."""

    def test_does_not_sleep(self, monkeypatch):
        sleep_calls: list[float] = []

        def fake_sleep(duration: float):
            sleep_calls.append(duration)

        monkeypatch.setattr(cameras.time, 'sleep', fake_sleep)

        cam, q = _make_camera(fps=10, width=2, height=2)
        q.put((np.zeros((2, 2, 3), dtype=np.uint8), datetime.datetime.now()))

        cam.get_next_frame(timeout_s=1.0)
        assert sleep_calls == []


class TestFlushQueue:

    def test_flushes_all_pending_frames(self):
        cam, q = _make_camera()
        ts = datetime.datetime.now()
        q.put((np.zeros((2, 2, 3), dtype=np.uint8), ts))
        q.put((np.zeros((2, 2, 3), dtype=np.uint8), ts))

        flushed = cam.flush_queue()

        assert flushed == 2
        assert q.empty()

    def test_returns_zero_when_empty(self):
        cam, _q = _make_camera()
        assert cam.flush_queue() == 0

    def test_flush_then_get_raises(self):
        cam, q = _make_camera()
        ts = datetime.datetime.now()
        q.put((np.zeros((2, 2, 3), dtype=np.uint8), ts))
        cam.flush_queue()

        with pytest.raises(cameras.FrameUnavailableError):
            cam.get_next_frame(timeout_s=0.01)


class TestSetupAndClose:
    """Setup and close are no-ops but should not raise."""

    def test_setup_is_noop(self):
        cam, _q = _make_camera()
        cam.setup()  # Should not raise.

    def test_close_is_noop(self):
        cam, _q = _make_camera()
        cam.close()  # Should not raise.

    def test_context_manager_works(self):
        cam, q = _make_camera(width=2, height=2)
        q.put((np.zeros((2, 2, 3), dtype=np.uint8), datetime.datetime.now()))

        with cam:
            frame, _ = cam.get_next_frame(timeout_s=1.0)
            assert frame.shape == (2, 2, 3)
