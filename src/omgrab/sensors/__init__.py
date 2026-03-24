"""Sensor abstractions for non-video timestamped data."""

from omgrab.sensors.queue_reader_sensor import QueueReaderSensor
from omgrab.sensors.sensor import Sensor
from omgrab.sensors.sensor import SensorDataUnavailableError

__all__ = [
    'QueueReaderSensor',
    'Sensor',
    'SensorDataUnavailableError',
]
