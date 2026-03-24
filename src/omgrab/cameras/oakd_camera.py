"""OAK-D camera implementations.

Camera classes that read frames from queues populated by the OAK-D capture
device (see omgrab.devices.oakd_capture_device). Each camera instance
represents a single stream (RGB or depth) for the duration of a recording
session.
"""
import datetime
import queue

from omgrab.cameras import cameras
from omgrab.cameras import queue_reader_camera

TimestampedRGBFrame = tuple[cameras.RGBFrame, datetime.datetime]
TimestampedDepthFrame = tuple[cameras.DepthFrame, datetime.datetime]
TimestampedIMUData = tuple[bytes, datetime.datetime]


class OakDRGBCamera(queue_reader_camera.QueueReaderCamera[cameras.RGBFrame]):
    """OakD RGB camera class."""
    def __init__(
            self, config: cameras.CameraConfig,
            frame_queue: queue.Queue[TimestampedRGBFrame]):
        """Initialize the OakD RGB camera."""
        super().__init__(config, frame_queue)


class OakDDepthCamera(queue_reader_camera.QueueReaderCamera[cameras.DepthFrame]):
    """OakD Depth camera class."""
    def __init__(
            self, config: cameras.CameraConfig,
            frame_queue: queue.Queue[TimestampedDepthFrame]):
        """Initialize the OakD Depth camera."""
        super().__init__(config, frame_queue)
