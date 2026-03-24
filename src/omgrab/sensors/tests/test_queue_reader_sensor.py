"""Tests for QueueReaderSensor."""
import datetime
import queue

import pytest

from omgrab.sensors import queue_reader_sensor
from omgrab.sensors import sensor


class TestQueueReaderSensor:

    def test_returns_data_from_queue(self):
        q: queue.Queue[tuple[bytes, datetime.datetime]] = queue.Queue()
        ts = datetime.datetime(2025, 6, 15, 12, 0, 0)
        q.put((b'payload', ts))

        src = queue_reader_sensor.QueueReaderSensor(q)
        data, timestamp = src.get_next_item(timeout_s=1.0)

        assert data == b'payload'
        assert timestamp == ts

    def test_returns_multiple_items_in_order(self):
        q: queue.Queue[tuple[bytes, datetime.datetime]] = queue.Queue()
        ts1 = datetime.datetime(2025, 6, 15, 12, 0, 0)
        ts2 = datetime.datetime(2025, 6, 15, 12, 0, 1)
        q.put((b'first', ts1))
        q.put((b'second', ts2))

        src = queue_reader_sensor.QueueReaderSensor(q)
        data1, t1 = src.get_next_item(timeout_s=1.0)
        data2, t2 = src.get_next_item(timeout_s=1.0)

        assert data1 == b'first'
        assert t1 == ts1
        assert data2 == b'second'
        assert t2 == ts2

    def test_raises_sensor_data_not_available_on_empty_queue(self):
        q: queue.Queue[tuple[bytes, datetime.datetime]] = queue.Queue()
        src = queue_reader_sensor.QueueReaderSensor(q)

        with pytest.raises(sensor.SensorDataUnavailableError, match='Queue empty'):
            src.get_next_item(timeout_s=0.01)

    def test_raises_sensor_data_not_available_with_zero_timeout(self):
        q: queue.Queue[tuple[bytes, datetime.datetime]] = queue.Queue()
        src = queue_reader_sensor.QueueReaderSensor(q)

        with pytest.raises(sensor.SensorDataUnavailableError):
            src.get_next_item(timeout_s=0)

    def test_flush_queue_empties_all_items(self):
        q: queue.Queue[tuple[bytes, datetime.datetime]] = queue.Queue()
        ts = datetime.datetime.now()
        q.put((b'a', ts))
        q.put((b'b', ts))
        q.put((b'c', ts))

        src = queue_reader_sensor.QueueReaderSensor(q)
        flushed = src.flush_queue()

        assert flushed == 3
        assert q.empty()

    def test_flush_queue_returns_zero_when_empty(self):
        q: queue.Queue[tuple[bytes, datetime.datetime]] = queue.Queue()
        src = queue_reader_sensor.QueueReaderSensor(q)

        assert src.flush_queue() == 0

    def test_flush_then_get_raises(self):
        q: queue.Queue[tuple[bytes, datetime.datetime]] = queue.Queue()
        ts = datetime.datetime.now()
        q.put((b'data', ts))

        src = queue_reader_sensor.QueueReaderSensor(q)
        src.flush_queue()

        with pytest.raises(sensor.SensorDataUnavailableError):
            src.get_next_item(timeout_s=0.01)
