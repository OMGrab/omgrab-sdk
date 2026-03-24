"""Camera implementation that reads frames from a queue."""
from typing import Generic
from typing import Optional

import datetime
import queue

import cv2

from omgrab.cameras import cameras


class QueueReaderCamera(cameras.Camera[cameras.FrameT],
                        Generic[cameras.FrameT]):
    """Camera that reads frames from a queue."""
    def __init__(self, config: cameras.CameraConfig, frame_queue: queue.Queue):
        """Initialize the queue reader camera."""
        super().__init__(config, enforce_frame_timing=False)
        self._frame_queue = frame_queue

    def setup(self):
        """Setup the queue reader camera."""
        pass

    def close(self):
        """Close the queue reader camera."""
        pass

    def get_next_frame(
            self, timeout_s: Optional[float] = None
            ) -> tuple[cameras.FrameT, datetime.datetime]:
        """Get the next frame from the queue."""
        try:
            frame, timestamp = self._frame_queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise cameras.FrameUnavailableError(
                f'Queue empty after {timeout_s}s timeout.') from exc
        target_w, target_h = self._config.width, self._config.height
        if frame.shape[0] != target_h or frame.shape[1] != target_w:
            frame = cv2.resize(frame, (target_w, target_h))
        self._maybe_wait_remainder_of_frame()
        return frame, timestamp

    def flush_queue(self) -> int:
        """Flush all pending frames from the queue.

        Returns:
            Number of frames flushed.
        """
        flushed = 0
        while True:
            try:
                self._frame_queue.get_nowait()
                flushed += 1
            except queue.Empty:
                break
        return flushed
