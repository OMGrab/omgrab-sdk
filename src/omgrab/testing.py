"""Shared test doubles and helpers for omgrab unit tests.

This module provides reusable fake implementations of hardware-dependent
components. Import fakes directly in test files, or use the convenience
fixtures defined in conftest.py.
"""
from typing import Literal
from typing import Optional

import contextlib
import time
import types

from omgrab.devices import capture_device
from omgrab.gpio import gpio
from omgrab.runtime import network_monitor


class FakeGPIOController(gpio.GPIOController):
    """Test double for GPIOController that records all calls.

    Subclasses GPIOController but overrides __init__ to avoid spawning
    a subprocess or importing hardware-specific modules.
    """

    def __init__(self):
        # Skip super().__init__() to avoid starting the GPIO subprocess.
        self.led_calls: list[tuple[str, str]] = []  # (red_value, green_value)
        self.buzzer_states: list[bool] = []
        self.buzzer_beeps: list[tuple[float, int]] = []  # (duration, count)
        self.buzzer_volumes: list[float] = []
        self.buzzer_tones: list[float] = []
        self.button_pressed: bool = False
        self.cleaned_up: bool = False

    def set_led_states(self, red: gpio.LEDState, green: gpio.LEDState):
        """Record LED state change."""
        self.led_calls.append((red.value, green.value))

    def set_buzzer(self, state: bool):
        """Record buzzer on/off."""
        self.buzzer_states.append(state)

    def set_buzzer_volume(self, volume: float):
        """Record buzzer volume change."""
        self.buzzer_volumes.append(volume)

    def set_buzzer_tone(self, freq: float):
        """Record buzzer tone change."""
        self.buzzer_tones.append(freq)

    def read_button(self) -> bool:
        """Return the preconfigured button state."""
        return self.button_pressed

    def buzzer_beep(self, duration: float = 0.1, count: int = 1):
        """Record a beep request."""
        self.buzzer_beeps.append((duration, count))

    def cleanup(self):
        """Record cleanup."""
        self.cleaned_up = True


class FakeCaptureDevice:
    """Test double for CaptureDevice that simulates device lifecycle."""

    def __init__(
            self,
            *,
            connected: bool = True,
            ready: bool = True,
            device_type: Optional[str] = None):
        self._connected = connected
        self._ready = ready
        self._device_type = device_type
        self.opened = False

    @property
    def label(self) -> str:
        """Return the device label."""
        return 'fake'

    @property
    def connected(self) -> bool:
        """Return whether the device is connected."""
        return self._connected

    @property
    def ready(self) -> bool:
        """Return whether the device is ready."""
        return self._ready

    @property
    def device_type(self) -> Optional[str]:
        """Return the device type identifier."""
        return self._device_type

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        """Wait until the device is ready."""
        return self._ready

    def __enter__(self) -> 'FakeCaptureDevice':
        if self.opened:
            raise RuntimeError('FakeCaptureDevice already open')
        self.opened = True
        return self

    def __exit__(
            self,
            exc_type: Optional[type[BaseException]],
            exc_value: Optional[BaseException],
            traceback: Optional[types.TracebackType]) -> Literal[False]:
        self.opened = False
        return False

    def preview(self) -> contextlib.AbstractContextManager['FakeCaptureDevice']:
        """Return a preview context manager."""
        raise capture_device.PreviewUnavailableError(
            'Preview not supported in FakeCaptureDevice')


def make_snapshot(
        status: network_monitor.Status = network_monitor.Status.ONLINE,
        detail: str = 'test') -> network_monitor.Snapshot:
    """Create a network Snapshot for testing."""
    return network_monitor.Snapshot(
        status=status,
        detail=detail,
        changed_at=time.monotonic(),
    )
