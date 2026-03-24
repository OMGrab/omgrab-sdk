"""Camera module for omgrab."""
from typing import Generic
from typing import Optional
from typing import TypeVar

import abc
import dataclasses
import datetime
import time

import numpy as np


@dataclasses.dataclass
class CameraConfig:
    """Base camera configuration.

    Attributes:
        fps: Target capture frame rate.
        width: Frame width in pixels.
        height: Frame height in pixels.
    """

    fps: int
    width: int
    height: int


Frame = np.ndarray
FrameT = TypeVar('FrameT', bound=Frame)  # Generic type for the frame type.
RGBFrame = np.ndarray[tuple[int, int, int], np.dtype[np.uint8]]
DepthFrame = np.ndarray[tuple[int, int], np.dtype[np.uint16]]


class FrameUnavailableError(RuntimeError):
    """Exception raised when a frame is not available."""


class Camera(Generic[FrameT], abc.ABC):
    """Camera class for omgrab."""

    def __init__(self, config: CameraConfig, enforce_frame_timing: bool = True):
        """Initialize the camera.

        Args:
            config: Camera configuration.
            enforce_frame_timing: If True, get_next_frame will sleep to enforce
                the configured FPS. Set to False for queue-based cameras where
                the hardware already controls frame timing.
        """
        self._config = config
        self._enforce_frame_timing = enforce_frame_timing
        self._last_frame_arrival_time_s = time.monotonic()
        self._frametime_s = 1.0 / self._config.fps

    def _maybe_wait_remainder_of_frame(self):
        """Wait for the next frame time to be reached.

        Only sleeps if enforce_frame_timing is True.
        """
        if not self._enforce_frame_timing:
            return
        time_to_wait_s = self._frametime_s - (time.monotonic() - self._last_frame_arrival_time_s)
        if time_to_wait_s > 0:
            time.sleep(time_to_wait_s)
        self._last_frame_arrival_time_s = time.monotonic()

    @property
    def config(self) -> CameraConfig:
        """Get the camera configuration."""
        return self._config

    @abc.abstractmethod
    def get_next_frame(
            self, timeout_s: Optional[float] = None
            ) -> tuple[FrameT, datetime.datetime]:
        """Get the next frame from the camera."""
        raise NotImplementedError

    @abc.abstractmethod
    def setup(self):
        """Setup the camera."""
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        """Close the camera."""
        raise NotImplementedError

    def __enter__(self):
        """Enter the context manager."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self.close()
