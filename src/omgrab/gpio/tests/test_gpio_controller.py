"""Tests for the GPIO controller (GPIO/gpio.py).

Tests focus on:
- LEDState enum values
- GPIOController delegation to GPIOControllerProcess
- Correct extraction of LEDState.value when delegating
"""
import pytest

from omgrab.gpio import gpio
from omgrab.gpio import gpio_process


class TestLEDState:
    """Verify LEDState enum members and values."""

    @pytest.mark.parametrize('member,value', [
        ('OFF', 'off'),
        ('ON', 'on'),
        ('SLOW_BLINK', 'slow_blink'),
        ('FAST_BLINK', 'fast_blink'),
    ])
    def test_member_values(self, member: str, value: str):
        """Each LED state should have the expected string value."""
        assert gpio.LEDState[member].value == value

    def test_has_exactly_four_members(self):
        """Enum should contain exactly four states."""
        assert len(gpio.LEDState) == 4



class _SpyProcessController:
    """Records all calls made to the process controller."""

    def __init__(self):
        self.calls: list[tuple[str, tuple]] = []

    def set_led_states(self, red: str, green: str):
        self.calls.append(('set_led_states', (red, green)))

    def set_buzzer(self, state: bool):
        self.calls.append(('set_buzzer', (state,)))

    def set_buzzer_volume(self, volume: float):
        self.calls.append(('set_buzzer_volume', (volume,)))

    def set_buzzer_tone(self, freq: float):
        self.calls.append(('set_buzzer_tone', (freq,)))

    def buzzer_beep(self, duration: float, count: int):
        self.calls.append(('buzzer_beep', (duration, count)))

    def read_button(self) -> bool:
        self.calls.append(('read_button', ()))
        return True

    def cleanup(self):
        self.calls.append(('cleanup', ()))


def _make_controller(monkeypatch) -> tuple[gpio.GPIOController, _SpyProcessController]:
    """Create a GPIOController with a spy process controller.

    Returns:
        Tuple of (controller, spy).
    """
    spy = _SpyProcessController()
    # Prevent GPIOControllerProcess from spawning a real process.
    monkeypatch.setattr(
        gpio_process, 'GPIOControllerProcess', lambda: spy)
    ctrl = gpio.GPIOController()
    return ctrl, spy


class TestGPIOControllerSetLedStates:

    def test_delegates_with_value_strings(self, monkeypatch):
        """set_led_states should pass LEDState.value strings to the process."""
        ctrl, spy = _make_controller(monkeypatch)

        ctrl.set_led_states(gpio.LEDState.ON, gpio.LEDState.SLOW_BLINK)

        assert spy.calls == [('set_led_states', ('on', 'slow_blink'))]

    @pytest.mark.parametrize('red,green', [
        (gpio.LEDState.OFF, gpio.LEDState.OFF),
        (gpio.LEDState.ON, gpio.LEDState.ON),
        (gpio.LEDState.FAST_BLINK, gpio.LEDState.SLOW_BLINK),
    ])
    def test_all_combinations_delegate_correctly(
            self, monkeypatch, red: gpio.LEDState, green: gpio.LEDState):
        """Various LED state combinations should delegate with correct values."""
        ctrl, spy = _make_controller(monkeypatch)

        ctrl.set_led_states(red, green)

        assert spy.calls == [('set_led_states', (red.value, green.value))]


class TestGPIOControllerSetBuzzer:

    @pytest.mark.parametrize('state', [True, False])
    def test_delegates_buzzer_state(self, monkeypatch, state: bool):
        """set_buzzer should delegate the boolean state."""
        ctrl, spy = _make_controller(monkeypatch)

        ctrl.set_buzzer(state)

        assert spy.calls == [('set_buzzer', (state,))]


class TestGPIOControllerSetBuzzerVolume:

    def test_delegates_volume(self, monkeypatch):
        """set_buzzer_volume should delegate the volume float."""
        ctrl, spy = _make_controller(monkeypatch)

        ctrl.set_buzzer_volume(0.5)

        assert spy.calls == [('set_buzzer_volume', (0.5,))]


class TestGPIOControllerSetBuzzerTone:

    def test_delegates_frequency(self, monkeypatch):
        """set_buzzer_tone should delegate the frequency float."""
        ctrl, spy = _make_controller(monkeypatch)

        ctrl.set_buzzer_tone(1000.0)

        assert spy.calls == [('set_buzzer_tone', (1000.0,))]


class TestGPIOControllerReadButton:

    def test_delegates_and_returns_result(self, monkeypatch):
        """read_button should delegate and return the process response."""
        ctrl, spy = _make_controller(monkeypatch)

        result = ctrl.read_button()

        assert result is True
        assert spy.calls == [('read_button', ())]


class TestGPIOControllerBuzzerBeep:

    def test_delegates_duration_and_count(self, monkeypatch):
        """buzzer_beep should delegate with duration and count."""
        ctrl, spy = _make_controller(monkeypatch)

        ctrl.buzzer_beep(0.2, 3)

        assert spy.calls == [('buzzer_beep', (0.2, 3))]

    def test_default_args(self, monkeypatch):
        """buzzer_beep() with defaults should delegate (0.1, 1)."""
        ctrl, spy = _make_controller(monkeypatch)

        ctrl.buzzer_beep()

        assert spy.calls == [('buzzer_beep', (0.1, 1))]


class TestGPIOControllerCleanup:

    def test_delegates_cleanup(self, monkeypatch):
        """Cleanup should delegate to the process controller."""
        ctrl, spy = _make_controller(monkeypatch)

        ctrl.cleanup()

        assert spy.calls == [('cleanup', ())]
