"""Tests for the GPIO process module (GPIO/gpio_process.py).

Tests focus on:
- Pure functions (_calculate_led_state)
- Enum definitions and cross-enum alignment
- GPIOControllerProcess command queueing (mocked multiprocessing)
- Pin constants and timing constants
"""
import multiprocessing

import pytest

from omgrab.gpio import gpio
from omgrab.gpio import gpio_process


class TestGPIOCommand:
    """Verify GPIOCommand enum members."""

    @pytest.mark.parametrize('member,value', [
        ('SET_LED_STATES', 'set_led_states'),
        ('SET_BUZZER', 'set_buzzer'),
        ('SET_BUZZER_VOLUME', 'set_buzzer_volume'),
        ('SET_BUZZER_TONE', 'set_buzzer_tone'),
        ('BUZZER_BEEP', 'buzzer_beep'),
        ('READ_BUTTON', 'read_button'),
        ('SHUTDOWN', 'shutdown'),
    ])
    def test_member_values(self, member: str, value: str):
        """Each command should have the expected string value."""
        assert gpio_process.GPIOCommand[member].value == value

    def test_has_exactly_seven_members(self):
        """Enum should contain exactly seven commands."""
        assert len(gpio_process.GPIOCommand) == 7



class TestLEDPattern:
    """Verify LEDPattern enum members."""

    @pytest.mark.parametrize('member,value', [
        ('OFF', 'off'),
        ('ON', 'on'),
        ('SLOW_BLINK', 'slow_blink'),
        ('FAST_BLINK', 'fast_blink'),
    ])
    def test_member_values(self, member: str, value: str):
        """Each pattern should have the expected string value."""
        assert gpio_process.LEDPattern[member].value == value

    def test_has_exactly_four_members(self):
        """Enum should contain exactly four patterns."""
        assert len(gpio_process.LEDPattern) == 4



class TestLEDStatePatternAlignment:
    """Ensure LEDState (gpio.py) and LEDPattern (gpio_process.py) stay in sync.

    GPIOController.set_led_states() converts LEDState.value strings to
    the process, where they are reconstituted as LEDPattern. If the enums
    diverge, LED commands will silently fail.
    """

    def test_every_led_state_has_matching_pattern(self):
        """Every LEDState value should be constructible as a LEDPattern."""
        for state in gpio.LEDState:
            pattern = gpio_process.LEDPattern(state.value)
            assert pattern.value == state.value

    def test_every_led_pattern_has_matching_state(self):
        """Every LEDPattern value should be constructible as a LEDState."""
        for pattern in gpio_process.LEDPattern:
            state = gpio.LEDState(pattern.value)
            assert state.value == pattern.value

    def test_same_member_count(self):
        """Both enums should have the same number of members."""
        assert len(gpio.LEDState) == len(gpio_process.LEDPattern)



class TestPinConstants:
    """Verify pin assignments are consistent between gpio.py and gpio_process.py."""

    def test_green_led_pin_matches(self):
        """Green LED pin should match between modules."""
        assert gpio.GPIOController.PIN_GREEN_LED == gpio_process.PIN_GREEN_LED

    def test_red_led_pin_matches(self):
        """Red LED pin should match between modules."""
        assert gpio.GPIOController.PIN_RED_LED == gpio_process.PIN_RED_LED

    def test_buzzer_pin_matches(self):
        """Buzzer pin should match between modules."""
        assert gpio.GPIOController.PIN_BUZZER == gpio_process.PIN_BUZZER

    def test_button_pin_matches(self):
        """Button pin should match between modules."""
        assert gpio.GPIOController.PIN_BUTTON == gpio_process.PIN_BUTTON



class TestTimingConstants:
    """Verify timing constants are sensible."""

    def test_slow_blink_interval_positive(self):
        """Slow blink interval must be positive."""
        assert gpio_process.SLOW_BLINK_INTERVAL > 0

    def test_fast_blink_interval_positive(self):
        """Fast blink interval must be positive."""
        assert gpio_process.FAST_BLINK_INTERVAL > 0

    def test_fast_blink_faster_than_slow(self):
        """Fast blink interval should be shorter than slow blink."""
        assert gpio_process.FAST_BLINK_INTERVAL < gpio_process.SLOW_BLINK_INTERVAL

    def test_loop_interval_positive(self):
        """Loop interval must be positive."""
        assert gpio_process.LOOP_INTERVAL > 0

    def test_loop_interval_shorter_than_fast_blink(self):
        """Loop interval must be much shorter than fast blink for smooth animation."""
        assert gpio_process.LOOP_INTERVAL < gpio_process.FAST_BLINK_INTERVAL / 2



class TestCalculateLedStateOff:
    """OFF pattern should always return False."""

    @pytest.mark.parametrize('t', [0.0, 0.5, 1.0, 100.0, 999999.0])
    def test_off_always_false(self, t: float):
        """OFF pattern returns False at any time."""
        assert gpio_process._calculate_led_state(
            gpio_process.LEDPattern.OFF, t) is False


class TestCalculateLedStateOn:
    """ON pattern should always return True."""

    @pytest.mark.parametrize('t', [0.0, 0.5, 1.0, 100.0, 999999.0])
    def test_on_always_true(self, t: float):
        """ON pattern returns True at any time."""
        assert gpio_process._calculate_led_state(
            gpio_process.LEDPattern.ON, t) is True


class TestCalculateLedStateSlowBlink:
    """SLOW_BLINK at ~1 Hz (1.0s interval, 2.0s full cycle)."""

    def test_on_at_start_of_cycle(self):
        """LED should be ON at the beginning of a cycle."""
        assert gpio_process._calculate_led_state(
            gpio_process.LEDPattern.SLOW_BLINK, 0.0) is True

    def test_on_in_first_half(self):
        """LED should be ON during the first half of the cycle."""
        assert gpio_process._calculate_led_state(
            gpio_process.LEDPattern.SLOW_BLINK, 0.5) is True

    def test_off_in_second_half(self):
        """LED should be OFF during the second half of the cycle."""
        assert gpio_process._calculate_led_state(
            gpio_process.LEDPattern.SLOW_BLINK, 1.5) is False

    def test_on_at_next_cycle(self):
        """LED should be ON at the start of the next cycle."""
        assert gpio_process._calculate_led_state(
            gpio_process.LEDPattern.SLOW_BLINK, 2.0) is True

    def test_blink_is_periodic(self):
        """State at time t should equal state at t + full_cycle."""
        full_cycle = gpio_process.SLOW_BLINK_INTERVAL * 2
        for t in [0.0, 0.3, 0.7, 1.2, 1.8]:
            a = gpio_process._calculate_led_state(
                gpio_process.LEDPattern.SLOW_BLINK, t)
            b = gpio_process._calculate_led_state(
                gpio_process.LEDPattern.SLOW_BLINK, t + full_cycle)
            assert a == b, f'Not periodic at t={t}'

    def test_50_percent_duty_cycle(self):
        """LED should be ON for roughly half the cycle and OFF for the other half."""
        full_cycle = gpio_process.SLOW_BLINK_INTERVAL * 2
        # Sample at 1ms resolution over one full cycle.
        samples = 2000
        on_count = sum(
            gpio_process._calculate_led_state(
                gpio_process.LEDPattern.SLOW_BLINK,
                i * full_cycle / samples)
            for i in range(samples)
        )
        ratio = on_count / samples
        assert 0.49 <= ratio <= 0.51, f'Duty cycle {ratio:.2%} is not ~50%'


class TestCalculateLedStateFastBlink:
    """FAST_BLINK at ~5 Hz (0.2s interval, 0.4s full cycle)."""

    def test_on_at_start_of_cycle(self):
        """LED should be ON at the beginning of a cycle."""
        assert gpio_process._calculate_led_state(
            gpio_process.LEDPattern.FAST_BLINK, 0.0) is True

    def test_off_in_second_half(self):
        """LED should be OFF during the second half of the cycle."""
        assert gpio_process._calculate_led_state(
            gpio_process.LEDPattern.FAST_BLINK, 0.3) is False

    def test_blink_is_periodic(self):
        """State at time t should equal state at t + full_cycle."""
        full_cycle = gpio_process.FAST_BLINK_INTERVAL * 2
        for t in [0.0, 0.05, 0.15, 0.25, 0.35]:
            a = gpio_process._calculate_led_state(
                gpio_process.LEDPattern.FAST_BLINK, t)
            b = gpio_process._calculate_led_state(
                gpio_process.LEDPattern.FAST_BLINK, t + full_cycle)
            assert a == b, f'Not periodic at t={t}'

    def test_faster_transition_than_slow_blink(self):
        """FAST_BLINK should transition from ON to OFF sooner than SLOW_BLINK.

        Sample at fine resolution and find the first OFF sample for each.
        """
        first_off_fast = None
        first_off_slow = None
        for ms in range(2001):
            t = ms / 1000.0
            if first_off_fast is None and not gpio_process._calculate_led_state(
                    gpio_process.LEDPattern.FAST_BLINK, t):
                first_off_fast = t
            if first_off_slow is None and not gpio_process._calculate_led_state(
                    gpio_process.LEDPattern.SLOW_BLINK, t):
                first_off_slow = t
            if first_off_fast is not None and first_off_slow is not None:
                break

        assert first_off_fast is not None
        assert first_off_slow is not None
        assert first_off_fast < first_off_slow


class TestCalculateLedStateEdgeCases:
    """Edge cases and defensive behaviour."""

    def test_negative_time_does_not_crash(self):
        """Negative times should not raise (Python modulo handles them)."""
        # Should not raise.
        result = gpio_process._calculate_led_state(
            gpio_process.LEDPattern.SLOW_BLINK, -1.5)
        assert isinstance(result, bool)

    def test_very_large_time(self):
        """Very large time values should still produce a valid boolean."""
        result = gpio_process._calculate_led_state(
            gpio_process.LEDPattern.FAST_BLINK, 1e9)
        assert isinstance(result, bool)

    def test_boundary_at_exact_transition(self):
        """At exactly the half-cycle point, LED should transition to OFF.

        For SLOW_BLINK: interval=1.0, cycle=2.0s, half=1.0s.
        At t=1.0 exactly: time_ms=1000, cycle_ms=2000, position=1000,
        1000 < 1000 is False → OFF.
        """
        assert gpio_process._calculate_led_state(
            gpio_process.LEDPattern.SLOW_BLINK, 1.0) is False



class _FakeProcess:
    """Fake multiprocessing.Process that doesn't actually start."""

    def __init__(self, **kwargs):
        self.started = False
        self._alive = False

    def start(self):
        self.started = True
        self._alive = True

    def join(self, timeout=None):
        self._alive = False

    def is_alive(self):
        return self._alive

    def terminate(self):
        self._alive = False


def _make_controller(monkeypatch) -> tuple[
        gpio_process.GPIOControllerProcess,
        multiprocessing.Queue,
        multiprocessing.Queue]:
    """Create a GPIOControllerProcess with a fake process.

    Returns:
        Tuple of (controller, command_queue, response_queue).
    """
    cmd_q: multiprocessing.Queue = multiprocessing.Queue()
    resp_q: multiprocessing.Queue = multiprocessing.Queue()

    # Prevent real Process creation.
    monkeypatch.setattr(
        multiprocessing, 'Process', lambda **kw: _FakeProcess(**kw))
    # Capture the queues that the constructor creates.
    queues_created: list[multiprocessing.Queue] = []
    original_queue = multiprocessing.Queue

    def fake_queue():
        q = queues_created[len(queues_created)] if False else original_queue()
        queues_created.append(q)
        return q

    # We need to intercept the queues. Simpler: just build the object
    # and then swap in our known queues.
    ctrl = gpio_process.GPIOControllerProcess()
    ctrl._command_queue = cmd_q
    ctrl._response_queue = resp_q
    return ctrl, cmd_q, resp_q


class TestGPIOControllerProcessSetLedStates:
    """Verify set_led_states puts the right command on the queue."""

    def test_sends_set_led_states_command(self, monkeypatch):
        """set_led_states should enqueue SET_LED_STATES with pattern strings."""
        ctrl, cmd_q, _ = _make_controller(monkeypatch)

        ctrl.set_led_states('on', 'slow_blink')

        cmd, args = cmd_q.get(timeout=1.0)
        assert cmd == gpio_process.GPIOCommand.SET_LED_STATES
        assert args == ('on', 'slow_blink')


class TestGPIOControllerProcessSetBuzzer:
    """Verify set_buzzer puts the right command on the queue."""

    @pytest.mark.parametrize('state', [True, False])
    def test_sends_set_buzzer_command(self, monkeypatch, state: bool):
        """set_buzzer should enqueue SET_BUZZER with the boolean state."""
        ctrl, cmd_q, _ = _make_controller(monkeypatch)

        ctrl.set_buzzer(state)

        cmd, args = cmd_q.get(timeout=1.0)
        assert cmd == gpio_process.GPIOCommand.SET_BUZZER
        assert args is state


class TestGPIOControllerProcessSetBuzzerVolume:
    """Verify set_buzzer_volume puts the right command on the queue."""

    def test_sends_set_buzzer_volume_command(self, monkeypatch):
        """set_buzzer_volume should enqueue SET_BUZZER_VOLUME with the volume."""
        ctrl, cmd_q, _ = _make_controller(monkeypatch)

        ctrl.set_buzzer_volume(0.75)

        cmd, args = cmd_q.get(timeout=1.0)
        assert cmd == gpio_process.GPIOCommand.SET_BUZZER_VOLUME
        assert args == 0.75


class TestGPIOControllerProcessSetBuzzerTone:
    """Verify set_buzzer_tone puts the right command on the queue."""

    def test_sends_set_buzzer_tone_command(self, monkeypatch):
        """set_buzzer_tone should enqueue SET_BUZZER_TONE with the frequency."""
        ctrl, cmd_q, _ = _make_controller(monkeypatch)

        ctrl.set_buzzer_tone(440.0)

        cmd, args = cmd_q.get(timeout=1.0)
        assert cmd == gpio_process.GPIOCommand.SET_BUZZER_TONE
        assert args == 440.0


class TestGPIOControllerProcessBuzzerBeep:
    """Verify buzzer_beep puts the right command on the queue."""

    def test_sends_buzzer_beep_command(self, monkeypatch):
        """buzzer_beep should enqueue BUZZER_BEEP with (duration, count)."""
        ctrl, cmd_q, _ = _make_controller(monkeypatch)

        ctrl.buzzer_beep(0.2, 3)

        cmd, args = cmd_q.get(timeout=1.0)
        assert cmd == gpio_process.GPIOCommand.BUZZER_BEEP
        assert args == (0.2, 3)

    def test_default_duration_and_count(self, monkeypatch):
        """buzzer_beep() with no args should use defaults (0.1, 1)."""
        ctrl, cmd_q, _ = _make_controller(monkeypatch)

        ctrl.buzzer_beep()

        _, args = cmd_q.get(timeout=1.0)
        assert args == (0.1, 1)


class TestGPIOControllerProcessReadButton:
    """Verify read_button sends a command and reads the response."""

    def test_returns_response_from_queue(self, monkeypatch):
        """read_button should return the value placed on the response queue."""
        ctrl, cmd_q, resp_q = _make_controller(monkeypatch)

        resp_q.put(True)
        result = ctrl.read_button()

        assert result is True

        cmd, args = cmd_q.get(timeout=1.0)
        assert cmd == gpio_process.GPIOCommand.READ_BUTTON
        assert args is None

    def test_returns_false_on_timeout(self, monkeypatch):
        """read_button should return False if the response queue times out."""
        ctrl, _, resp_q = _make_controller(monkeypatch)

        # Don't put anything on the response queue — will timeout.
        # Override the timeout to be very short.
        original_get = resp_q.get

        def fast_timeout_get(timeout=None):
            return original_get(timeout=0.01)

        ctrl._response_queue.get = fast_timeout_get

        result = ctrl.read_button()

        assert result is False


class TestGPIOControllerProcessCleanup:
    """Verify cleanup sends shutdown and terminates the process."""

    def test_sends_shutdown_command(self, monkeypatch):
        """Cleanup should enqueue a SHUTDOWN command."""
        ctrl, cmd_q, _ = _make_controller(monkeypatch)

        ctrl.cleanup()

        cmd, args = cmd_q.get(timeout=1.0)
        assert cmd == gpio_process.GPIOCommand.SHUTDOWN
        assert args is None
