"""USB capture device."""
from typing import Optional

import contextlib
import logging
import os
import queue
import threading
import types
from collections.abc import Iterator

from omgrab.cameras import cameras
from omgrab.cameras import queue_reader_camera
from omgrab.cameras import usb_camera
from omgrab.cameras import usb_port
from omgrab.devices import capture_device

logger = logging.getLogger(__name__)

_PREVIEW_QUEUE_SIZE = 4


class USBCaptureDevice(capture_device.CaptureDevice):
    """Capture device wrapping a single USB camera identified by USB port path.

    Implements the CaptureDevice protocol. The camera is opened/closed
    per-recording by the capture thread (via the Camera context manager),
    not by this device's __enter__/__exit__.
    """

    def __init__(
            self,
            config: cameras.CameraConfig,
            usb_port_path: str,
            label: str,
            preview_config: Optional[cameras.CameraConfig] = None):
        """Initialize the USB capture device.

        Args:
            config: Camera configuration (fps, width, height).
            usb_port_path: USB port path (e.g. '3-2') used to discover
                the /dev/videoN device via sysfs.
            label: Human-readable label for logging (e.g. 'left_wrist').
            preview_config: Optional camera configuration for preview mode.
                If not provided, preview() will raise PreviewUnavailableError.
        """
        self._config = config
        self._usb_port_path = usb_port_path
        self._label = label
        self._camera = usb_camera.USBCamera(
            config, usb_port_path=usb_port_path)
        self._preview_config = preview_config
        self._preview_active = False
        if preview_config is not None:
            self._output_preview_queue: Optional[queue.Queue] = queue.Queue(
                maxsize=_PREVIEW_QUEUE_SIZE)
        else:
            self._output_preview_queue = None

    @property
    def label(self) -> str:
        """Human-readable label for this device."""
        return self._label

    @property
    def connected(self) -> bool:
        """Whether the USB camera is present and openable.

        Resolves the USB port path to a /dev/videoN device via sysfs, then
        verifies the device node can actually be opened. A device may appear
        in sysfs even with a loose connection; the open check catches that.
        """
        dev_path = usb_port.find_video_device_by_usb_port(self._usb_port_path)
        if dev_path is None:
            logger.debug(
                'USB camera %s not found at port %s',
                self._label, self._usb_port_path)
            return False
        try:
            fd = os.open(dev_path, os.O_RDWR | os.O_NONBLOCK)
            try:
                os.close(fd)
            except OSError:
                pass
            return True
        except OSError as e:
            logger.debug(
                'USB camera %s at %s found in sysfs but not openable: %s',
                self._label, dev_path, e)
            return False

    @property
    def ready(self) -> bool:
        """USB cameras have no warmup; ready if connected."""
        return True

    @property
    def device_type(self) -> Optional[str]:
        """USB cameras do not have a device type."""
        return None

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """USB cameras are immediately ready."""
        return True

    def get_camera(self) -> usb_camera.USBCamera:
        """Get the USB camera instance."""
        return self._camera

    def get_preview_camera(
            self) -> queue_reader_camera.QueueReaderCamera[cameras.RGBFrame]:
        """Get the preview camera.

        The returned camera reads from the preview queue and only produces
        frames while the ``preview()`` context manager is active.

        Returns:
            A QueueReaderCamera reading from the preview queue.

        Raises:
            PreviewUnavailableError: If preview_config was not provided.
        """
        if self._preview_config is None or self._output_preview_queue is None:
            raise capture_device.PreviewUnavailableError(
                'Preview is not enabled (preview_config was not provided)')
        return queue_reader_camera.QueueReaderCamera(
            self._preview_config, self._output_preview_queue)

    def __enter__(self) -> 'USBCaptureDevice':
        """No-op; the camera is opened by the capture thread."""
        if self._preview_active:
            raise RuntimeError(
                'Cannot start recording while preview is active.')
        return self

    def __exit__(
            self,
            exc_type: Optional[type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[types.TracebackType]):
        """No-op; the camera is closed by the capture thread."""
        pass

    @contextlib.contextmanager
    def preview(self) -> Iterator['USBCaptureDevice']:
        """Context manager that runs a preview capture loop.

        Opens a separate USBCamera with the preview configuration and
        reads frames in a background thread, pushing them into a queue.
        The preview is mutually exclusive with recording.

        Yields:
            USBCaptureDevice: This device instance.

        Raises:
            PreviewUnavailableError: If preview_config was not provided.
            RuntimeError: If preview is already active.
        """
        if self._preview_config is None or self._output_preview_queue is None:
            raise capture_device.PreviewUnavailableError(
                f'Preview not enabled for USB camera {self._label}')
        if self._preview_active:
            raise RuntimeError(
                f'Preview already active for USB camera {self._label}')

        preview_cam = usb_camera.USBCamera(
            self._preview_config, usb_port_path=self._usb_port_path)
        preview_cam.setup()

        stop_event = threading.Event()
        thread = threading.Thread(
            target=self._preview_capture_loop,
            args=(preview_cam, stop_event),
            daemon=True,
            name=f'usb-preview-{self._label}',
        )

        self._preview_active = True
        thread.start()
        try:
            self._flush_preview_queue()
            yield self
        finally:
            stop_event.set()
            thread.join(timeout=5.0)
            preview_cam.close()
            self._flush_preview_queue()
            self._preview_active = False

    def _preview_capture_loop(
            self,
            preview_cam: usb_camera.USBCamera,
            stop_event: threading.Event):
        """Background thread that reads frames and pushes to the preview queue."""
        assert self._output_preview_queue is not None
        while not stop_event.is_set():
            try:
                frame, timestamp = preview_cam.get_next_frame()
            except cameras.FrameUnavailableError:
                logger.debug('Preview frame not available for %s', self._label)
                break
            try:
                self._output_preview_queue.put_nowait((frame, timestamp))
            except queue.Full:
                try:
                    self._output_preview_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._output_preview_queue.put_nowait((frame, timestamp))
                except queue.Full:
                    pass

    def _flush_preview_queue(self):
        """Drain all pending frames from the preview queue."""
        if self._output_preview_queue is None:
            return
        while True:
            try:
                self._output_preview_queue.get_nowait()
            except queue.Empty:
                break
