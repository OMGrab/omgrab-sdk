"""Battery monitoring module for Waveshare UPS HAT B (INA219).

This module provides battery status information including voltage, current,
power consumption, and battery percentage for devices with the Waveshare UPS HAT B.

Based on Waveshare's INA219 example code, adapted for this project's style.
"""
from typing import TYPE_CHECKING
from typing import Optional

import dataclasses
import logging
import time

if TYPE_CHECKING:
    import smbus2

logger = logging.getLogger(__name__)

# How often to retry connecting to the INA219 if it's not available.
_RETRY_INTERVAL_SECONDS = 30.0


# I2C Register addresses for INA219.
_REG_CONFIG = 0x00
_REG_SHUNTVOLTAGE = 0x01
_REG_BUSVOLTAGE = 0x02
_REG_POWER = 0x03
_REG_CURRENT = 0x04
_REG_CALIBRATION = 0x05


# Bus voltage range constants.
_BUS_VOLTAGE_RANGE_16V = 0x00
_BUS_VOLTAGE_RANGE_32V = 0x01

# Gain constants.
_GAIN_DIV_1_40MV = 0x00
_GAIN_DIV_2_80MV = 0x01
_GAIN_DIV_4_160MV = 0x02
_GAIN_DIV_8_320MV = 0x03

# ADC resolution constants.
_ADC_RES_9BIT_1S = 0x00
_ADC_RES_10BIT_1S = 0x01
_ADC_RES_11BIT_1S = 0x02
_ADC_RES_12BIT_1S = 0x03
_ADC_RES_12BIT_32S = 0x0D

# Operating mode constants.
_MODE_POWERDOWN = 0x00
_MODE_SVOLT_TRIGGERED = 0x01
_MODE_BVOLT_TRIGGERED = 0x02
_MODE_SANDBVOLT_TRIGGERED = 0x03
_MODE_ADC_OFF = 0x04
_MODE_SVOLT_CONTINUOUS = 0x05
_MODE_BVOLT_CONTINUOUS = 0x06
_MODE_SANDBVOLT_CONTINUOUS = 0x07


@dataclasses.dataclass
class BatteryStatus:
    """Battery status information from the INA219 sensor.

    Attributes:
        voltage_v: Load voltage in volts.
        current_a: Current draw in amps (positive when charging).
        power_w: Power consumption in watts.
        percent: Estimated battery percentage (0-100).
    """

    voltage_v: float
    current_a: float
    power_w: float
    percent: float

    def __str__(self) -> str:
        """Return human-readable battery status."""
        state = 'charging' if self.current_a > 0 else 'discharging'
        return (f'Battery: {self.percent:.1f}% {state} | '
                f'{self.voltage_v:.2f}V | '
                f'{self.current_a:.3f}A | '
                f'{self.power_w:.2f}W')


class BatteryMonitor:
    """Monitor battery status using INA219 chip on Waveshare UPS HAT B.

    The INA219 measures voltage, current, and power. Battery percentage is
    calculated based on voltage (6V = 0%, 8.4V = 100% for typical 2S Li-ion).
    """

    def __init__(self, i2c_bus: int = 1, i2c_addr: int = 0x42,
                 min_voltage: float = 6.0, max_voltage: float = 8.4):
        """Initialize the battery monitor.

        Args:
            i2c_bus: I2C bus number (default: 1).
            i2c_addr: I2C address of INA219 chip (default: 0x42 for UPS HAT B).
            min_voltage: Minimum battery voltage for 0% (default: 6.0V).
            max_voltage: Maximum battery voltage for 100% (default: 8.4V).
        """
        self._i2c_bus = i2c_bus
        self._i2c_addr = i2c_addr
        self._min_voltage = min_voltage
        self._max_voltage = max_voltage
        self._voltage_range = max_voltage - min_voltage

        self._bus: Optional[smbus2.SMBus] = None
        self._available = False
        self._cal_value = 0
        self._current_lsb = 0.0
        self._power_lsb = 0.0
        self._last_retry_time: float = 0.0
        self._consecutive_failures = 0

        self._initialize()

    def _initialize(self, is_retry: bool = False):
        """Initialize the INA219 chip.

        Args:
            is_retry: True if this is a retry attempt (changes log level).
        """
        self._last_retry_time = time.monotonic()

        try:
            import smbus2

            # Close existing bus if any.
            if self._bus is not None:
                try:
                    self._bus.close()
                except Exception:
                    pass
                self._bus = None

            self._bus = smbus2.SMBus(self._i2c_bus)

            # Configure the chip for 32V, 2A measurement range.
            self._set_calibration_32v_2a()
            self._available = True
            self._consecutive_failures = 0

            if is_retry:
                logger.info('Battery monitor reconnected on I2C bus %d, addr 0x%02x',
                            self._i2c_bus, self._i2c_addr)
            else:
                logger.info('Battery monitor initialized on I2C bus %d, addr 0x%02x',
                            self._i2c_bus, self._i2c_addr)
        except OSError as e:
            if e.errno == 121:  # Remote I/O error - device not present.
                if not is_retry:
                    logger.info('Battery hardware not detected (I2C address 0x%02x not responding)',
                                self._i2c_addr)
                # Silently fail on retries - will try again later.
            else:
                logger.warning('Failed to initialize battery monitor: %s', e)
            self._available = False
        except Exception as e:
            logger.warning('Failed to initialize battery monitor: %s', e)
            self._available = False

    def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the INA219 if enough time has passed.

        Returns:
            True if reconnection was attempted (regardless of success).
        """
        now = time.monotonic()
        if now - self._last_retry_time < _RETRY_INTERVAL_SECONDS:
            return False

        logger.debug('Attempting to reconnect to battery monitor...')
        self._initialize(is_retry=True)
        return True

    def _read_register(self, address: int) -> int:
        """Read 16-bit value from register.

        Args:
            address: Register address.

        Returns:
            16-bit register value.
        """
        assert self._bus is not None
        data: list[int] = self._bus.read_i2c_block_data(self._i2c_addr, address, 2)
        return (data[0] << 8) | data[1]

    def _write_register(self, address: int, value: int):
        """Write 16-bit value to register.

        Args:
            address: Register address.
            value: 16-bit value to write.
        """
        assert self._bus is not None
        high_byte = (value >> 8) & 0xFF
        low_byte = value & 0xFF
        self._bus.write_i2c_block_data(self._i2c_addr, address, [high_byte, low_byte])

    def _set_calibration_32v_2a(self):
        """Configure INA219 to measure up to 32V and 2A.

        Assumes 0.1 ohm shunt resistor (standard for Waveshare UPS HAT B).
        Calibration calculations from Waveshare example code.
        """
        # Current LSB = 100uA per bit (0.1 mA).
        self._current_lsb = 0.1

        # Calibration value: Cal = trunc(0.04096 / (Current_LSB * R_SHUNT))
        # With Current_LSB = 0.0001A and R_SHUNT = 0.1 ohm: Cal = 4096
        self._cal_value = 4096

        # Power LSB = 20 * Current_LSB = 2mW per bit.
        self._power_lsb = 0.002

        # Set calibration register.
        self._write_register(_REG_CALIBRATION, self._cal_value)

        # Set config register for continuous shunt and bus voltage measurement.
        config = (
            (_BUS_VOLTAGE_RANGE_32V << 13) |
            (_GAIN_DIV_8_320MV << 11) |
            (_ADC_RES_12BIT_32S << 7) |
            (_ADC_RES_12BIT_32S << 3) |
            _MODE_SANDBVOLT_CONTINUOUS
        )
        self._write_register(_REG_CONFIG, config)

    def _get_bus_voltage_v(self) -> float:
        """Get bus voltage (load side) in volts.

        Returns:
            Voltage in volts.
        """
        # Refresh calibration before reading.
        self._write_register(_REG_CALIBRATION, self._cal_value)
        value = self._read_register(_REG_BUSVOLTAGE)
        # Bits 3-15 contain voltage, LSB = 4mV.
        return (value >> 3) * 0.004

    def _get_current_a(self) -> float:
        """Get current in amps.

        Returns:
            Current in amps (positive = charging, negative = discharging).
        """
        value = self._read_register(_REG_CURRENT)
        # Handle signed 16-bit value.
        if value > 32767:
            value -= 65536
        # _current_lsb is in mA, convert to amps.
        return (value * self._current_lsb) / 1000.0

    def _get_power_w(self) -> float:
        """Get power consumption in watts.

        Returns:
            Power in watts.
        """
        # Refresh calibration before reading.
        self._write_register(_REG_CALIBRATION, self._cal_value)
        value = self._read_register(_REG_POWER)
        # Handle signed 16-bit value.
        if value > 32767:
            value -= 65536
        return value * self._power_lsb

    def _calculate_percentage(self, voltage_v: float) -> float:
        """Calculate battery percentage from voltage.

        Uses linear interpolation between min and max voltage thresholds.

        Args:
            voltage_v: Battery voltage in volts.

        Returns:
            Battery percentage (0-100).
        """
        if voltage_v <= self._min_voltage:
            return 0.0
        if voltage_v >= self._max_voltage:
            return 100.0

        percent = ((voltage_v - self._min_voltage) / self._voltage_range) * 100.0
        return round(percent, 1)

    def get_status(self) -> Optional[BatteryStatus]:
        """Get current battery status.

        If the battery hardware is not available, this method will periodically
        attempt to reconnect (every 30 seconds by default).

        Returns:
            BatteryStatus with current measurements, or None if battery monitoring
            is not available or read fails.
        """
        # If not available, try to reconnect periodically.
        if not self._available or self._bus is None:
            self._try_reconnect()
            if not self._available:
                return None

        try:
            voltage_v = self._get_bus_voltage_v()
            current_a = self._get_current_a()
            power_w = self._get_power_w()
            percent = self._calculate_percentage(voltage_v)

            # Successful read - reset failure counter.
            self._consecutive_failures = 0

            return BatteryStatus(
                voltage_v=round(voltage_v, 3),
                current_a=round(current_a, 4),
                power_w=round(power_w, 3),
                percent=percent,
            )
        except OSError as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3:
                # Multiple consecutive failures - device likely disconnected.
                logger.warning('Battery monitor lost connection after %d failures: %s',
                               self._consecutive_failures, e)
                self._available = False
                # Reset retry timer so we try again soon.
                self._last_retry_time = time.monotonic() - _RETRY_INTERVAL_SECONDS + 5.0
            return None
        except Exception as e:
            logger.warning('Failed to read battery status: %s', e)
            return None

    @property
    def available(self) -> bool:
        """True if battery monitoring hardware is available."""
        return self._available
