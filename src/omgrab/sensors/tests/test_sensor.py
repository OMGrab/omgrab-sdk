"""Tests for Sensor base class."""
import queue

import pytest

from omgrab.sensors import queue_reader_sensor
from omgrab.sensors import sensor


class TestSensorDataUnavailableError:

    def test_is_runtime_error(self):
        assert issubclass(sensor.SensorDataUnavailableError, RuntimeError)

    def test_carries_message(self):
        exc = sensor.SensorDataUnavailableError('queue empty')
        assert str(exc) == 'queue empty'


class TestSensorContextManager:
    """Verify the base Sensor context manager protocol."""

    def test_enter_returns_self(self):
        src = queue_reader_sensor.QueueReaderSensor(queue.Queue())
        with src as s:
            assert s is src

    def test_exit_does_not_suppress_exceptions(self):
        src = queue_reader_sensor.QueueReaderSensor(queue.Queue())
        with pytest.raises(ValueError, match='oops'), src:
            raise ValueError('oops')
