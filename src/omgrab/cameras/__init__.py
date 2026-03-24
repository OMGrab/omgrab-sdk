"""Camera abstractions for frame capture."""

from omgrab.cameras.cameras import Camera
from omgrab.cameras.cameras import CameraConfig
from omgrab.cameras.cameras import DepthFrame
from omgrab.cameras.cameras import Frame
from omgrab.cameras.cameras import FrameUnavailableError
from omgrab.cameras.cameras import RGBFrame
from omgrab.cameras.oakd_camera import OakDDepthCamera
from omgrab.cameras.oakd_camera import OakDRGBCamera
from omgrab.cameras.usb_camera import USBCamera

__all__ = [
    'Camera',
    'CameraConfig',
    'DepthFrame',
    'Frame',
    'FrameUnavailableError',
    'OakDDepthCamera',
    'OakDRGBCamera',
    'RGBFrame',
    'USBCamera',
]
