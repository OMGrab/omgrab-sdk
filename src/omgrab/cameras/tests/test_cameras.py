"""Tests for the Camera base class: frame pacing, config, and context manager."""
from typing import Optional

import datetime

import numpy as np
import pytest

from omgrab.cameras import cameras


class _StubCamera(cameras.Camera[cameras.Frame]):
    """Minimal concrete Camera for testing the base class."""

    def __init__(
            self,
            config: cameras.CameraConfig,
            enforce_frame_timing: bool = True):
        super().__init__(config, enforce_frame_timing=enforce_frame_timing)
        self.setup_called = False
        self.close_called = False
        self._frame_value: int = 0

    def get_next_frame(
            self, timeout_s: Optional[float] = None
            ) -> tuple[cameras.Frame, datetime.datetime]:
        """Return a tiny frame with incrementing pixel values."""
        self._maybe_wait_remainder_of_frame()
        frame = np.full(
            (self._config.height, self._config.width, 3),
            self._frame_value, dtype=np.uint8)
        self._frame_value += 1
        return frame, datetime.datetime.now()

    def setup(self):
        self.setup_called = True

    def close(self):
        self.close_called = True


@pytest.fixture
def camera_sleep(monkeypatch):
    """Stub out frame pacing sleeps and return the list of durations."""
    sleep_calls: list[float] = []

    def fake_sleep(duration: float):
        sleep_calls.append(duration)

    monkeypatch.setattr(cameras.time, 'sleep', fake_sleep)
    return sleep_calls


class TestCameraConfig:

    def test_config_stores_values(self):
        cfg = cameras.CameraConfig(fps=30, width=640, height=480)
        assert cfg.fps == 30
        assert cfg.width == 640
        assert cfg.height == 480

    def test_config_exposes_property(self):
        cfg = cameras.CameraConfig(fps=24, width=1920, height=1080)
        cam = _StubCamera(cfg)
        assert cam.config is cfg


class TestFrameUnavailableError:

    def test_is_runtime_error(self):
        assert issubclass(cameras.FrameUnavailableError, RuntimeError)

    def test_carries_message(self):
        exc = cameras.FrameUnavailableError('no frame')
        assert str(exc) == 'no frame'


class TestFramePacing:

    def test_sleeps_when_enforce_frame_timing_enabled(
            self, monkeypatch, camera_sleep):
        """With enforce_frame_timing=True the camera should sleep."""
        timeline = {'now': 100.0}
        monkeypatch.setattr(cameras.time, 'monotonic', lambda: timeline['now'])

        cfg = cameras.CameraConfig(fps=10, width=2, height=2)
        cam = _StubCamera(cfg, enforce_frame_timing=True)
        # Force _last_frame_arrival_time_s to the mocked clock.
        cam._last_frame_arrival_time_s = 100.0

        cam.get_next_frame()
        assert len(camera_sleep) == 1
        assert camera_sleep[0] == pytest.approx(1.0 / 10)

    def test_no_sleep_when_enforce_frame_timing_disabled(
            self, monkeypatch, camera_sleep):
        """With enforce_frame_timing=False the camera should not sleep."""
        monkeypatch.setattr(cameras.time, 'monotonic', lambda: 0.0)

        cfg = cameras.CameraConfig(fps=30, width=2, height=2)
        cam = _StubCamera(cfg, enforce_frame_timing=False)
        cam.get_next_frame()
        assert camera_sleep == []

    def test_no_sleep_when_frame_period_elapsed(
            self, monkeypatch, camera_sleep):
        """No sleep when enough time has passed since the last frame."""
        timeline = {'now': 100.0}
        monkeypatch.setattr(cameras.time, 'monotonic', lambda: timeline['now'])

        cfg = cameras.CameraConfig(fps=10, width=2, height=2)
        cam = _StubCamera(cfg, enforce_frame_timing=True)
        # Simulate: last frame was 0.2s ago, frame period is 0.1s → no sleep.
        cam._last_frame_arrival_time_s = 99.8

        cam.get_next_frame()
        assert camera_sleep == []

    @pytest.mark.parametrize('fps', [5, 24, 60, 120])
    def test_sleep_duration_matches_fps(
            self, monkeypatch, camera_sleep, fps):
        """Sleep duration should equal 1/fps when called immediately."""
        timeline = {'now': 0.0}
        monkeypatch.setattr(cameras.time, 'monotonic', lambda: timeline['now'])

        cfg = cameras.CameraConfig(fps=fps, width=2, height=2)
        cam = _StubCamera(cfg, enforce_frame_timing=True)
        cam._last_frame_arrival_time_s = 0.0

        cam.get_next_frame()
        assert camera_sleep[0] == pytest.approx(1.0 / fps)

    def test_uses_monotonic_clock_not_wall_clock(
            self, monkeypatch, camera_sleep):
        """Frame pacing should use time.monotonic, not time.time.

        Ensures that a wall-clock jump (e.g. NTP sync) does not cause a
        spurious long sleep or skip the sleep entirely.
        """
        monotonic_time = {'now': 1000.0}
        wall_time = {'now': 1000.0}

        monkeypatch.setattr(
            cameras.time, 'monotonic', lambda: monotonic_time['now'])
        # Wall clock jumps forward by 10 seconds (NTP sync).
        wall_time['now'] += 10.0
        monkeypatch.setattr(cameras.time, 'time', lambda: wall_time['now'])

        cfg = cameras.CameraConfig(fps=10, width=2, height=2)
        cam = _StubCamera(cfg, enforce_frame_timing=True)
        cam._last_frame_arrival_time_s = 1000.0  # Monotonic reference.

        cam.get_next_frame()

        # Sleep should be based on monotonic (0.1s), not wall clock.
        assert len(camera_sleep) == 1
        assert camera_sleep[0] == pytest.approx(1.0 / 10)


class TestContextManager:

    def test_enter_calls_setup(self):
        cfg = cameras.CameraConfig(fps=30, width=2, height=2)
        cam = _StubCamera(cfg, enforce_frame_timing=False)

        with cam:
            assert cam.setup_called

    def test_exit_calls_close(self):
        cfg = cameras.CameraConfig(fps=30, width=2, height=2)
        cam = _StubCamera(cfg, enforce_frame_timing=False)

        with cam:
            pass
        assert cam.close_called

    def test_exit_does_not_suppress_exceptions(self):
        cfg = cameras.CameraConfig(fps=30, width=2, height=2)
        cam = _StubCamera(cfg, enforce_frame_timing=False)

        with pytest.raises(ValueError, match='boom'), cam:
            raise ValueError('boom')
        assert cam.close_called
