"""Tests for USB capture device preview support."""
from typing import Optional

import datetime
import unittest.mock

import numpy as np
import pytest

from omgrab.cameras import cameras
from omgrab.cameras import queue_reader_camera
from omgrab.devices import capture_device
from omgrab.devices import usb_capture_device


def _make_frame(width: int = 640, height: int = 400) -> cameras.RGBFrame:
    """Create a dummy RGB frame."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class _FakeUSBCamera:
    """Fake USBCamera that yields a fixed number of frames then stops."""

    def __init__(self, config: cameras.CameraConfig, num_frames: int = 10):
        self._config = config
        self._num_frames = num_frames
        self._frame_count = 0
        self._setup_called = False
        self._closed = False

    def setup(self):
        self._setup_called = True

    def close(self):
        self._closed = True

    def get_next_frame(
            self, timeout_s: Optional[float] = None
            ) -> tuple[cameras.RGBFrame, datetime.datetime]:
        if self._frame_count >= self._num_frames:
            raise cameras.FrameUnavailableError('No more frames')
        self._frame_count += 1
        frame = _make_frame(self._config.width, self._config.height)
        return frame, datetime.datetime.now()


class TestUSBCaptureDevicePreviewUnavailable:
    """Preview raises PreviewUnavailableError when not configured."""

    def test_preview_raises_when_no_config(self):
        dev = usb_capture_device.USBCaptureDevice(
            cameras.CameraConfig(fps=15, width=1280, height=720),
            usb_port_path='3-2',
            label='left_wrist',
        )
        with pytest.raises(capture_device.PreviewUnavailableError), dev.preview():
            pass

    def test_get_preview_camera_raises_when_no_config(self):
        dev = usb_capture_device.USBCaptureDevice(
            cameras.CameraConfig(fps=15, width=1280, height=720),
            usb_port_path='3-2',
            label='left_wrist',
        )
        with pytest.raises(capture_device.PreviewUnavailableError):
            dev.get_preview_camera()


class TestUSBCaptureDevicePreview:
    """Preview context manager lifecycle and frame production."""

    def _make_device(self) -> usb_capture_device.USBCaptureDevice:
        return usb_capture_device.USBCaptureDevice(
            cameras.CameraConfig(fps=15, width=1280, height=720),
            usb_port_path='3-2',
            label='left_wrist',
            preview_config=cameras.CameraConfig(fps=10, width=640, height=400),
        )

    def test_get_preview_camera_returns_queue_reader(self):
        dev = self._make_device()
        cam = dev.get_preview_camera()
        assert isinstance(cam, queue_reader_camera.QueueReaderCamera)

    @unittest.mock.patch(
        'omgrab.devices.usb_capture_device.usb_camera.USBCamera',
        autospec=True)
    def test_preview_starts_and_stops(self, mock_usb_camera_cls):
        fake_cam = _FakeUSBCamera(
            cameras.CameraConfig(fps=10, width=640, height=400))
        mock_usb_camera_cls.return_value = fake_cam

        dev = self._make_device()
        assert not dev._preview_active

        with dev.preview() as d:
            assert d is dev
            assert dev._preview_active

        assert not dev._preview_active
        assert fake_cam._closed

    @unittest.mock.patch(
        'omgrab.devices.usb_capture_device.usb_camera.USBCamera',
        autospec=True)
    def test_preview_produces_readable_frames(self, mock_usb_camera_cls):
        fake_cam = _FakeUSBCamera(
            cameras.CameraConfig(fps=10, width=640, height=400),
            num_frames=100)
        mock_usb_camera_cls.return_value = fake_cam

        dev = self._make_device()
        preview_cam = dev.get_preview_camera()

        with dev.preview():
            frame, ts = preview_cam.get_next_frame(timeout_s=2.0)
            assert frame.shape == (400, 640, 3)
            assert isinstance(ts, datetime.datetime)

    @unittest.mock.patch(
        'omgrab.devices.usb_capture_device.usb_camera.USBCamera',
        autospec=True)
    def test_preview_cleans_up_on_exception(self, mock_usb_camera_cls):
        fake_cam = _FakeUSBCamera(
            cameras.CameraConfig(fps=10, width=640, height=400),
            num_frames=100)
        mock_usb_camera_cls.return_value = fake_cam

        dev = self._make_device()

        with pytest.raises(ValueError), dev.preview():
            raise ValueError('boom')

        assert not dev._preview_active
        assert fake_cam._closed

    @unittest.mock.patch(
        'omgrab.devices.usb_capture_device.usb_camera.USBCamera',
        autospec=True)
    def test_double_preview_raises(self, mock_usb_camera_cls):
        fake_cam = _FakeUSBCamera(
            cameras.CameraConfig(fps=10, width=640, height=400),
            num_frames=100)
        mock_usb_camera_cls.return_value = fake_cam

        dev = self._make_device()

        with dev.preview(), pytest.raises(RuntimeError, match='already active'), dev.preview():
            pass


class TestUSBCaptureDeviceMutualExclusion:
    """Preview and recording are mutually exclusive."""

    @unittest.mock.patch(
        'omgrab.devices.usb_capture_device.usb_camera.USBCamera',
        autospec=True)
    def test_enter_raises_while_preview_active(self, mock_usb_camera_cls):
        fake_cam = _FakeUSBCamera(
            cameras.CameraConfig(fps=10, width=640, height=400),
            num_frames=100)
        mock_usb_camera_cls.return_value = fake_cam

        dev = usb_capture_device.USBCaptureDevice(
            cameras.CameraConfig(fps=15, width=1280, height=720),
            usb_port_path='3-2',
            label='left_wrist',
            preview_config=cameras.CameraConfig(fps=10, width=640, height=400),
        )

        with dev.preview(), pytest.raises(RuntimeError, match='preview is active'):
            dev.__enter__()
