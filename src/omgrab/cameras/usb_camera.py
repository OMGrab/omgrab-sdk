"""USB camera implementation using OpenCV VideoCapture."""
from typing import Optional
from typing import cast

import datetime
import logging

import cv2

from omgrab.cameras import cameras
from omgrab.cameras import usb_port

logger = logging.getLogger(__name__)


class USBCamera(cameras.Camera[cameras.RGBFrame]):
    """Camera that reads frames from a USB camera via OpenCV."""

    def __init__(
            self,
            config: cameras.CameraConfig,
            device_path: str = '',
            usb_port_path: str = ''):
        """Initialize the USB camera.

        Exactly one of device_path or usb_port_path must be provided.

        Args:
            config: Camera configuration (fps, width, height).
            device_path: Path to the video device (e.g. '/dev/video2').
            usb_port_path: USB port path (e.g. '3-2'). The device path
                is resolved from sysfs each time setup() is called.
        """
        if not device_path and not usb_port_path:
            raise ValueError('Either device_path or usb_port_path is required')
        super().__init__(config, enforce_frame_timing=False)
        self._device_path = device_path
        self._usb_port_path = usb_port_path
        self._cap: Optional[cv2.VideoCapture] = None

    def setup(self):
        """Open the USB camera and configure resolution/fps."""
        device_path = self._device_path
        if self._usb_port_path:
            resolved = usb_port.find_video_device_by_usb_port(
                self._usb_port_path)
            if resolved is None:
                raise RuntimeError(
                    f'No video device found for USB port {self._usb_port_path}')
            device_path = resolved

        logger.info('Opening USB camera at %s', device_path)
        self._cap = cv2.VideoCapture(device_path, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f'Failed to open USB camera at {device_path}')
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)
        self._cap.set(cv2.CAP_PROP_FPS, self._config.fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            'USB camera opened: requested %dx%d@%dfps, got %dx%d@%.1ffps',
            self._config.width, self._config.height, self._config.fps,
            actual_w, actual_h, actual_fps)
        # Store the resolved path for logging in get_next_frame/close.
        self._device_path = device_path

    def close(self):
        """Release the USB camera."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info('USB camera at %s closed', self._device_path)

    def get_next_frame(
            self, timeout_s: Optional[float] = None
            ) -> tuple[cameras.RGBFrame, datetime.datetime]:
        """Read the next frame from the USB camera.

        Args:
            timeout_s: Not used. On V4L2 with BUFFERSIZE=1, grab() blocks for
                at most one frame period (~33ms at 30fps). If the device is
                disconnected, grab() returns False immediately.

        Returns:
            Tuple of (frame, timestamp).

        Raises:
            FrameUnavailableError: If the frame could not be read.
        """
        if self._cap is None:
            raise cameras.FrameUnavailableError('USB camera not opened')
        self._maybe_wait_remainder_of_frame()
        if not self._cap.grab():
            raise cameras.FrameUnavailableError(
                f'Failed to grab frame from {self._device_path}')
        timestamp = datetime.datetime.now()
        ret, frame = self._cap.retrieve()
        if not ret:
            raise cameras.FrameUnavailableError(
                f'Failed to retrieve frame from {self._device_path}')
        # OpenCV returns BGR; convert to RGB for consistency with OAK-D.
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize if the camera didn't honour the requested resolution.
        if frame.shape[1] != self._config.width or frame.shape[0] != self._config.height:
            frame = cv2.resize(frame, (self._config.width, self._config.height))
        return cast(cameras.RGBFrame, frame), timestamp
