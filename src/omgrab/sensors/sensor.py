"""Sensor abstraction for non-video timestamped data (e.g., IMU readings)."""
from typing import Optional

import abc
import datetime


class SensorDataUnavailableError(RuntimeError):
    """Exception raised when sensor data is not available within the timeout."""


class Sensor(abc.ABC):
    """Source of timestamped data payloads (e.g., serialized sensor readings)."""

    @abc.abstractmethod
    def get_next_item(
            self, timeout_s: Optional[float] = None
            ) -> tuple[bytes, datetime.datetime]:
        """Get the next data payload.

        Args:
            timeout_s: Maximum time to wait in seconds, or None to block.

        Returns:
            Tuple of (data_bytes, timestamp).

        Raises:
            SensorDataUnavailableError: If no data is available within the timeout.
        """

    @abc.abstractmethod
    def setup(self):
        """Setup the sensor."""

    @abc.abstractmethod
    def close(self):
        """Close the sensor."""

    def __enter__(self):
        """Enter the context manager."""
        self.setup()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Exit the context manager."""
        self.close()
