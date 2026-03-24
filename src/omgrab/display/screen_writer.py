"""Low-level OLED screen writer for SSD1306 displays over I2C.

This module provides a hardware abstraction for writing PIL images to an
SSD1306 OLED display. It handles initialization, reconnection on failure,
and graceful degradation when the hardware is absent.

Modeled after the battery_monitor module's retry/reconnect pattern.
"""
import logging
import time

import PIL.Image

logger = logging.getLogger(__name__)

# How often to retry connecting to the display if it's not available.
_RETRY_INTERVAL_SECONDS = 30.0

# Number of consecutive write failures before marking the display as unavailable.
_MAX_CONSECUTIVE_FAILURES = 3


class ScreenWriter:
    """Hardware abstraction for SSD1306 OLED display over I2C.

    Handles initialization, display writes, and automatic reconnection.
    This class is not thread-safe on its own; callers should ensure that
    display() and clear() are called from a single thread.
    """

    def __init__(self, i2c_bus: int = 1, i2c_address: int = 0x3C,
                 width: int = 128, height: int = 64):
        """Initialize the screen writer.

        Args:
            i2c_bus: I2C bus number (default: 1).
            i2c_address: I2C address of the SSD1306 (default: 0x3C).
            width: Display width in pixels (default: 128).
            height: Display height in pixels (default: 64).
        """
        self._i2c_bus = i2c_bus
        self._i2c_address = i2c_address
        self._width = width
        self._height = height

        self._device = None  # Optional[luma_device.ssd1306]
        self._available = False
        self._last_retry_time: float = 0.0
        self._consecutive_failures = 0

        self._initialize()

    def _initialize(self, is_retry: bool = False):
        """Initialize the OLED device over I2C.

        Args:
            is_retry: True if this is a retry attempt (changes log level).
        """
        self._last_retry_time = time.monotonic()

        try:
            from luma.core.interface import serial as luma_serial
            from luma.oled import device as luma_device

            # Clean up existing device if any.
            if self._device is not None:
                try:
                    self._device.cleanup()
                except Exception:
                    pass
                self._device = None

            serial = luma_serial.i2c(port=self._i2c_bus, address=self._i2c_address)
            self._device = luma_device.ssd1306(
                serial, width=self._width, height=self._height)
            self._available = True
            self._consecutive_failures = 0

            if is_retry:
                logger.info('Screen reconnected on I2C bus %d, addr 0x%02x',
                            self._i2c_bus, self._i2c_address)
            else:
                logger.info('Screen initialized on I2C bus %d, addr 0x%02x (%dx%d)',
                            self._i2c_bus, self._i2c_address,
                            self._width, self._height)
        except OSError as e:
            if e.errno == 121:  # Remote I/O error - device not present.
                if not is_retry:
                    logger.info('Screen hardware not detected '
                                '(I2C address 0x%02x not responding)',
                                self._i2c_address)
            else:
                logger.warning('Failed to initialize screen: %s', e)
            self._available = False
        except Exception as e:
            logger.warning('Failed to initialize screen: %s', e)
            self._available = False

    def _try_reconnect(self) -> bool:
        """Attempt to reconnect to the display if enough time has passed.

        Returns:
            True if reconnection was attempted (regardless of success).
        """
        now = time.monotonic()
        if now - self._last_retry_time < _RETRY_INTERVAL_SECONDS:
            return False

        logger.debug('Attempting to reconnect to screen...')
        self._initialize(is_retry=True)
        return True

    def display(self, image: PIL.Image.Image):
        """Write an image to the OLED display.

        The image should be mode '1' (1-bit) with dimensions matching the
        display (default 128x64). If the hardware is unavailable, this method
        attempts periodic reconnection.

        Args:
            image: PIL Image to display.
        """
        if not self._available or self._device is None:
            self._try_reconnect()
            if not self._available:
                return

        assert self._device is not None
        try:
            self._device.display(image)
            self._consecutive_failures = 0
        except OSError as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= _MAX_CONSECUTIVE_FAILURES:
                logger.warning(
                    'Screen lost connection after %d failures: %s',
                    self._consecutive_failures, e)
                self._available = False
                # Reset retry timer so we try again soon.
                self._last_retry_time = (
                    time.monotonic() - _RETRY_INTERVAL_SECONDS + 5.0)
        except Exception as e:
            logger.warning('Failed to write to screen: %s', e)

    def clear(self):
        """Clear the display (set all pixels to black).

        Safe to call even if the hardware is unavailable.
        """
        if not self._available or self._device is None:
            return

        try:
            blank = PIL.Image.new('1', (self._width, self._height), 0)
            self._device.display(blank)
        except Exception as e:
            logger.debug('Failed to clear screen: %s', e)

    def cleanup(self):
        """Clean up hardware resources.

        Should be called during shutdown.
        """
        if self._device is not None:
            try:
                self._device.cleanup()
            except Exception as e:
                logger.debug('Failed to cleanup screen device: %s', e)
            self._device = None
        self._available = False

    @property
    def available(self) -> bool:
        """True if the screen hardware is available."""
        return self._available

    @property
    def width(self) -> int:
        """Display width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Display height in pixels."""
        return self._height
