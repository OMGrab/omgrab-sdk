"""Queue-based sensor that reads timestamped bytes payloads from a queue."""
from typing import Optional

import datetime
import queue

from omgrab.sensors import sensor


class QueueReaderSensor(sensor.Sensor):
    """Sensor that reads timestamped bytes payloads from a queue."""

    def __init__(self, data_queue: queue.Queue[tuple[bytes, datetime.datetime]]):
        """Initialize the queue reader sensor.

        Args:
            data_queue: Queue providing (data_bytes, timestamp) tuples.
        """
        self._queue = data_queue

    def setup(self):
        """No-op for queue-based sensor."""
        pass

    def close(self):
        """No-op for queue-based sensor."""
        pass

    def get_next_item(
            self, timeout_s: Optional[float] = None
            ) -> tuple[bytes, datetime.datetime]:
        """Get the next data payload from the queue."""
        try:
            return self._queue.get(timeout=timeout_s)
        except queue.Empty as exc:
            raise sensor.SensorDataUnavailableError(
                f'Queue empty after {timeout_s}s timeout.') from exc

    def flush_queue(self) -> int:
        """Flush all pending items from the queue.

        Returns:
            Number of items flushed.
        """
        flushed = 0
        while True:
            try:
                self._queue.get_nowait()
                flushed += 1
            except queue.Empty:
                break
        return flushed
