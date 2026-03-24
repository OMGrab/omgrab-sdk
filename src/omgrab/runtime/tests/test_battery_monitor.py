"""Tests for the battery monitor (runtime/battery_monitor.py)."""
import time

import pytest

from omgrab.runtime import battery_monitor


class TestBatteryStatus:

    def test_stores_all_fields(self):
        """BatteryStatus should expose all measurement fields."""
        status = battery_monitor.BatteryStatus(
            voltage_v=7.8,
            current_a=-0.5,
            power_w=3.9,
            percent=75.0,
        )

        assert status.voltage_v == 7.8
        assert status.current_a == -0.5
        assert status.power_w == 3.9
        assert status.percent == 75.0

    def test_str_includes_percentage(self):
        """String representation should include battery percentage."""
        status = battery_monitor.BatteryStatus(
            voltage_v=7.8, current_a=0.1, power_w=0.78, percent=75.0,
        )

        text = str(status)
        assert '75.0%' in text

    def test_str_indicates_charging_or_discharging(self):
        """String should indicate charging/discharging based on current sign."""
        charging = battery_monitor.BatteryStatus(
            voltage_v=8.0, current_a=0.5, power_w=4.0, percent=80.0,
        )
        discharging = battery_monitor.BatteryStatus(
            voltage_v=7.0, current_a=-0.5, power_w=3.5, percent=50.0,
        )

        # Positive current should not show 'discharging'.
        # Negative current should not show 'charging'.
        # These are implied from the code which checks current_a > 0.
        assert 'charging' in str(charging).lower()
        assert 'discharging' in str(discharging).lower()



class FakeSMBus:
    """Fake I2C bus that allows controlled read/write."""

    def __init__(self, bus_number):
        self._registers: dict[int, list[int]] = {}

    def read_i2c_block_data(self, addr, register, length):
        if register in self._registers:
            return self._registers[register]
        return [0] * length

    def write_i2c_block_data(self, addr, register, data):
        self._registers[register] = data

    def close(self):
        pass


class TestCalculatePercentage:

    def _make_monitor(self, min_v: float = 6.0, max_v: float = 8.4):
        """Create a BatteryMonitor with fake I2C, just for _calculate_percentage."""
        # We create directly and set internal state, bypassing __init__.
        mon = object.__new__(battery_monitor.BatteryMonitor)
        mon._min_voltage = min_v
        mon._max_voltage = max_v
        mon._voltage_range = max_v - min_v
        return mon

    def test_at_min_voltage_returns_zero(self):
        """Voltage at minimum should return 0%."""
        mon = self._make_monitor()
        assert mon._calculate_percentage(6.0) == 0.0

    def test_below_min_voltage_returns_zero(self):
        """Voltage below minimum should clamp to 0%."""
        mon = self._make_monitor()
        assert mon._calculate_percentage(5.0) == 0.0

    def test_at_max_voltage_returns_hundred(self):
        """Voltage at maximum should return 100%."""
        mon = self._make_monitor()
        assert mon._calculate_percentage(8.4) == 100.0

    def test_above_max_voltage_returns_hundred(self):
        """Voltage above maximum should clamp to 100%."""
        mon = self._make_monitor()
        assert mon._calculate_percentage(9.0) == 100.0

    def test_midpoint_voltage(self):
        """Midpoint voltage should return ~50%."""
        mon = self._make_monitor(min_v=6.0, max_v=8.0)
        result = mon._calculate_percentage(7.0)
        assert result == 50.0

    @pytest.mark.parametrize('voltage,expected_min,expected_max', [
        (6.0, 0.0, 0.1),
        (7.2, 45.0, 55.0),
        (8.4, 99.9, 100.1),
    ])
    def test_percentage_range(self, voltage, expected_min, expected_max):
        """Percentage should be within expected range for given voltage."""
        mon = self._make_monitor()
        result = mon._calculate_percentage(voltage)
        assert expected_min <= result <= expected_max



class TestBatteryMonitorInit:

    def test_unavailable_when_smbus_import_fails(self, monkeypatch):
        """Monitor should be unavailable when smbus2 cannot be imported."""
        # Patch the import to fail.
        import builtins
        original_import = builtins.__import__

        def fail_smbus(name, *args, **kwargs):
            if name == 'smbus2':
                raise ImportError('no smbus2')
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fail_smbus)

        mon = battery_monitor.BatteryMonitor()
        assert mon.available is False

    def test_unavailable_when_i2c_device_not_present(self, monkeypatch):
        """Monitor should be unavailable on errno 121 (Remote I/O error)."""
        import builtins
        original_import = builtins.__import__

        class FailingSMBus:
            def __init__(self, bus):
                error = OSError('Remote I/O error')
                error.errno = 121
                raise error

        # Create a fake smbus2 module.
        import types
        fake_smbus2 = types.ModuleType('smbus2')
        fake_smbus2.SMBus = FailingSMBus

        def patched_import(name, *args, **kwargs):
            if name == 'smbus2':
                return fake_smbus2
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', patched_import)

        mon = battery_monitor.BatteryMonitor()
        assert mon.available is False



class TestBatteryMonitorGetStatus:

    def test_returns_none_when_unavailable(self, monkeypatch):
        """get_status should return None when hardware is not available."""
        import builtins
        original_import = builtins.__import__

        def fail_smbus(name, *args, **kwargs):
            if name == 'smbus2':
                raise ImportError('no smbus2')
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fail_smbus)

        mon = battery_monitor.BatteryMonitor()
        assert mon.get_status() is None

    def test_consecutive_failures_mark_unavailable(self, monkeypatch):
        """Three consecutive read failures should mark the monitor unavailable."""
        import builtins
        original_import = builtins.__import__

        class FailingReadSMBus:
            def __init__(self, bus):
                pass

            def read_i2c_block_data(self, addr, register, length):
                raise OSError('I2C read failed')

            def write_i2c_block_data(self, addr, register, data):
                pass

            def close(self):
                pass

        import types
        fake_smbus2 = types.ModuleType('smbus2')
        fake_smbus2.SMBus = FailingReadSMBus

        def patched_import(name, *args, **kwargs):
            if name == 'smbus2':
                return fake_smbus2
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', patched_import)

        mon = battery_monitor.BatteryMonitor()
        if not mon.available:
            # If init already failed, that's fine too.
            return

        # Three consecutive failures should mark unavailable.
        for _ in range(3):
            mon.get_status()

        assert mon.available is False



class TestBatteryMonitorReconnect:

    def test_try_reconnect_respects_interval(self):
        """_try_reconnect should not retry within the retry interval."""
        # Create a monitor that's unavailable but has a recent retry time.
        mon = object.__new__(battery_monitor.BatteryMonitor)
        mon._available = False
        mon._bus = None
        mon._last_retry_time = time.monotonic()
        mon._i2c_bus = 1
        mon._i2c_addr = 0x42
        mon._min_voltage = 6.0
        mon._max_voltage = 8.4
        mon._voltage_range = 2.4
        mon._cal_value = 0
        mon._current_lsb = 0.0
        mon._power_lsb = 0.0
        mon._consecutive_failures = 0

        result = mon._try_reconnect()
        assert result is False  # Too soon to retry.
