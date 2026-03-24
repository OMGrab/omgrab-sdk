"""Hardware tests for GPIO — only run with --hardware on a real Pi.

These tests exercise the real GPIOController and GPIOControllerProcess,
requiring gpiod access to /dev/gpiochip4 and physical GPIO wiring.

Run with:
    pytest --hardware -k test_gpio_hardware
"""
import time

import pytest

from omgrab.gpio import gpio

pytestmark = pytest.mark.hardware


class TestGPIOControllerHardware:
    """Smoke tests for GPIOController on real hardware."""

    @pytest.fixture(autouse=True)
    def _controller(self):
        """Create and clean up a real GPIOController."""
        self.ctrl = gpio.GPIOController()
        yield
        self.ctrl.set_led_states(gpio.LEDState.OFF, gpio.LEDState.OFF)
        self.ctrl.set_buzzer(False)
        self.ctrl.cleanup()

    def test_set_led_states_does_not_raise(self):
        """Setting LED states should not raise on real hardware."""
        self.ctrl.set_led_states(gpio.LEDState.ON, gpio.LEDState.OFF)

    def test_all_led_patterns(self):
        """All LED patterns should be accepted without error."""
        for state in gpio.LEDState:
            self.ctrl.set_led_states(state, state)
            time.sleep(0.05)

    def test_read_button_returns_bool(self):
        """read_button should return a boolean on real hardware."""
        result = self.ctrl.read_button()
        assert isinstance(result, bool)

    def test_set_buzzer_does_not_raise(self):
        """Toggling the buzzer should not raise."""
        self.ctrl.set_buzzer(True)
        self.ctrl.set_buzzer(False)

    def test_buzzer_beep_does_not_raise(self):
        """A short beep should complete without error."""
        self.ctrl.buzzer_beep(duration=0.05, count=1)

    def test_set_buzzer_volume_does_not_raise(self):
        """Setting buzzer volume should not raise."""
        self.ctrl.set_buzzer_volume(0.1)

    def test_set_buzzer_tone_does_not_raise(self):
        """Setting buzzer tone should not raise."""
        self.ctrl.set_buzzer_tone(440.0)

    def test_cleanup_does_not_raise(self):
        """Cleanup should complete without error."""
        self.ctrl.cleanup()
