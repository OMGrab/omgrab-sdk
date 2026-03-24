"""Tests for OAK-D camera classes (OakDRGBCamera, OakDDepthCamera)."""
import datetime
import queue

import numpy as np
import pytest

from omgrab.cameras import cameras
from omgrab.cameras import oakd_camera


@pytest.fixture(autouse=True)
def stub_cv2_resize(monkeypatch):
    """Replace cv2.resize with numpy-based resize."""
    import cv2
    resize_calls: list[tuple[int, int]] = []

    def fake_resize(frame, size):
        w, h = size
        resize_calls.append((w, h))
        if frame.ndim == 3:
            return np.zeros((h, w, frame.shape[2]), dtype=frame.dtype)
        return np.zeros((h, w), dtype=frame.dtype)

    monkeypatch.setattr(cv2, 'resize', fake_resize)
    return resize_calls


class TestOakDRGBCamera:

    def test_reads_frame_from_queue(self):
        q: queue.Queue[oakd_camera.TimestampedRGBFrame] = queue.Queue()
        cfg = cameras.CameraConfig(fps=30, width=4, height=3)
        cam = oakd_camera.OakDRGBCamera(cfg, q)

        ts = datetime.datetime(2025, 7, 1, 12, 0, 0)
        q.put((np.zeros((6, 8, 3), dtype=np.uint8), ts))

        frame, t = cam.get_next_frame(timeout_s=1.0)
        assert t == ts
        assert frame.shape == (3, 4, 3)

    def test_raises_frame_not_available_when_empty(self):
        q: queue.Queue[oakd_camera.TimestampedRGBFrame] = queue.Queue()
        cfg = cameras.CameraConfig(fps=30, width=4, height=3)
        cam = oakd_camera.OakDRGBCamera(cfg, q)

        with pytest.raises(cameras.FrameUnavailableError):
            cam.get_next_frame(timeout_s=0.01)


class TestOakDDepthCamera:

    def test_reads_depth_frame_from_queue(self):
        q: queue.Queue[oakd_camera.TimestampedDepthFrame] = queue.Queue()
        cfg = cameras.CameraConfig(fps=30, width=4, height=3)
        cam = oakd_camera.OakDDepthCamera(cfg, q)

        ts = datetime.datetime(2025, 7, 1, 12, 0, 0)
        q.put((np.zeros((6, 8), dtype=np.uint16), ts))

        frame, t = cam.get_next_frame(timeout_s=1.0)
        assert t == ts
        assert frame.shape == (3, 4)
