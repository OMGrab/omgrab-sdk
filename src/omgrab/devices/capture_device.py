"""Capture device abstraction.

A capture device represents the physical hardware (or driver/pipeline) that
owns the underlying resources required to produce one or more camera streams.
Devices can be powered on or off independently of any recording session.
Cameras, by contrast, are lightweight per-recording wrappers typically
created from a capture device, and are started/stopped each time the user
begins and ends a recording.
"""

from typing import Optional
from typing import Protocol
from typing import runtime_checkable

import contextlib
import types


class PreviewUnavailableError(RuntimeError):
    """Raised when preview is not available on a capture device.

    This occurs when a device was not configured with a preview config,
    or does not support preview at all.
    """


@runtime_checkable
class CaptureDevice(Protocol):
    """A physical capture device that can be opened/closed and queried for health."""

    @property
    def label(self) -> str:
        """Human-readable label for this device (e.g. 'oakd', 'left_wrist')."""

    @property
    def connected(self) -> bool:
        """Whether the underlying device appears connected/healthy."""

    @property
    def ready(self) -> bool:
        """Whether the underlying device is ready to capture frames."""

    @property
    def device_type(self) -> Optional[str]:
        """Return a device type identifier, if available.

        This is an opaque string whose meaning is defined by the concrete
        device implementation. For example, OAK-D devices report variants
        like ``'oakd_pro_wide'``. Returns None if the device has not been
        started, does not support type detection, or detection failed.
        """

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until the underlying device is ready to capture frames."""

    def __enter__(self) -> 'CaptureDevice':
        """Acquire underlying resources needed for capture."""

    def __exit__(
            self,
            exc_type: Optional[type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[types.TracebackType]):
        """Release underlying resources needed for capture."""

    def preview(self) -> contextlib.AbstractContextManager['CaptureDevice']:
        """Start a lightweight preview pipeline.

        Returns a context manager that runs a preview-only pipeline while
        active. The device must not be in recording mode (i.e. ``__enter__``
        must not be active). Implementations that do not support preview
        should raise ``RuntimeError``.
        """
