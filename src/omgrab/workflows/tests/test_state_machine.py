"""Tests for the StateMachine workflow coordinator."""

from omgrab import testing
from omgrab.gpio import gpio
from omgrab.runtime import network_monitor
from omgrab.workflows import state_machine


class FakeRecordingManager:
    """Test double for RecordingManager that records calls."""

    def __init__(self, *, start_succeeds: bool = True):
        self._start_succeeds = start_succeeds
        self.start_calls: int = 0
        self.stop_calls: int = 0

    def start_recording(self) -> bool:
        """Record the call and return preconfigured result."""
        self.start_calls += 1
        return self._start_succeeds

    def stop_recording(self):
        """Record the call."""
        self.stop_calls += 1


def _make_sm(
    fake_gpio: testing.FakeGPIOController | None = None,
    *,
    start_succeeds: bool = True,
) -> tuple[state_machine.StateMachine, testing.FakeGPIOController, FakeRecordingManager]:
    """Create a StateMachine with fakes.

    Returns:
        Tuple of (state_machine, fake_gpio, fake_recording_manager).
    """
    if fake_gpio is None:
        fake_gpio = testing.FakeGPIOController()
    rec_mgr = FakeRecordingManager(start_succeeds=start_succeeds)
    sm = state_machine.StateMachine(
        gpio_controller=fake_gpio,
        recording_manager=rec_mgr,
    )
    return sm, fake_gpio, rec_mgr


class TestWorkflowState:
    def test_all_expected_states_exist(self):
        """All expected workflow states should be defined."""
        expected = {'boot', 'idle', 'recording', 'camera_preview', 'wifi_setup', 'shutdown'}
        actual = {s.value for s in state_machine.WorkflowState}
        assert actual == expected


class TestInitialState:
    def test_auto_transitions_to_idle(self):
        """StateMachine should auto-transition from BOOT to IDLE on init."""
        sm, _, _ = _make_sm()

        assert sm.get_current_state() == 'idle'

    def test_not_recording_initially(self):
        """Should not be recording after init."""
        sm, _, _ = _make_sm()

        assert not sm.is_recording()


class TestButtonPress:
    def test_idle_press_starts_recording(self):
        """Button press in IDLE should start recording."""
        sm, _, rec_mgr = _make_sm()

        sm.handle_button_press()

        assert rec_mgr.start_calls == 1
        assert sm.get_current_state() == 'recording'
        assert sm.is_recording()

    def test_recording_press_stops_recording(self):
        """Button press in RECORDING should stop recording."""
        sm, _, rec_mgr = _make_sm()
        sm.handle_button_press()  # Start.

        sm.handle_button_press()  # Stop.

        assert rec_mgr.stop_calls == 1
        assert sm.get_current_state() == 'idle'
        assert not sm.is_recording()

    def test_start_recording_failure_stays_idle(self):
        """If recording manager fails to start, state should remain IDLE."""
        sm, _, _ = _make_sm(start_succeeds=False)

        sm.handle_button_press()

        assert sm.get_current_state() == 'idle'
        assert not sm.is_recording()

    def test_start_recording_failure_beeps_error(self):
        """Failed recording start should trigger error buzzer pattern."""
        sm, fake_gpio, _ = _make_sm(start_succeeds=False)

        sm.handle_button_press()

        # Should have a 5-beep error pattern.
        assert any(count == 5 for _, count in fake_gpio.buzzer_beeps)

    def test_start_recording_success_beeps_once(self):
        """Successful recording start should trigger single beep."""
        sm, fake_gpio, _ = _make_sm()
        # Clear beeps from init.
        fake_gpio.buzzer_beeps.clear()

        sm.handle_button_press()

        assert any(count == 1 for _, count in fake_gpio.buzzer_beeps)

    def test_stop_recording_beeps_twice(self):
        """Stopping recording should trigger double beep."""
        sm, fake_gpio, _ = _make_sm()
        sm.handle_button_press()  # Start.
        fake_gpio.buzzer_beeps.clear()

        sm.handle_button_press()  # Stop.

        assert any(count == 2 for _, count in fake_gpio.buzzer_beeps)

    def test_button_ignored_in_wifi_setup(self):
        """Button press during WiFi setup should be ignored."""
        sm, _, rec_mgr = _make_sm()
        sm.enter_wifi_setup()

        sm.handle_button_press()

        assert rec_mgr.start_calls == 0

    def test_button_ignored_in_shutdown(self):
        """Button press during shutdown should be ignored."""
        sm, _, rec_mgr = _make_sm()
        sm.shutdown()

        sm.handle_button_press()

        assert rec_mgr.start_calls == 0


class TestLEDStates:
    def _last_led(self, fake_gpio: testing.FakeGPIOController) -> tuple[str, str]:
        """Return the most recent (red, green) LED call."""
        assert fake_gpio.led_calls, 'No LED calls recorded'
        return fake_gpio.led_calls[-1]

    def test_idle_led_green_on(self):
        """IDLE state should have green ON, red OFF."""
        _, fake_gpio, _ = _make_sm()

        red, green = self._last_led(fake_gpio)
        assert red == gpio.LEDState.OFF.value
        assert green == gpio.LEDState.ON.value

    def test_recording_led_red_blink(self):
        """RECORDING state should have red SLOW_BLINK, green OFF."""
        sm, fake_gpio, _ = _make_sm()
        sm.handle_button_press()

        red, green = self._last_led(fake_gpio)
        assert red == gpio.LEDState.SLOW_BLINK.value
        assert green == gpio.LEDState.OFF.value

    def test_wifi_setup_led_both_fast_blink(self):
        """WIFI_SETUP should have both LEDs FAST_BLINK."""
        sm, fake_gpio, _ = _make_sm()
        sm.enter_wifi_setup()

        red, green = self._last_led(fake_gpio)
        assert red == gpio.LEDState.FAST_BLINK.value
        assert green == gpio.LEDState.FAST_BLINK.value

    def test_shutdown_led_both_off(self):
        """SHUTDOWN should have both LEDs OFF."""
        sm, fake_gpio, _ = _make_sm()
        sm.shutdown()

        red, green = self._last_led(fake_gpio)
        assert red == gpio.LEDState.OFF.value
        assert green == gpio.LEDState.OFF.value

    def test_shutdown_turns_off_buzzer(self):
        """SHUTDOWN should explicitly turn off the buzzer."""
        sm, fake_gpio, _ = _make_sm()
        sm.shutdown()

        assert False in fake_gpio.buzzer_states


class TestWifiSetup:
    def test_enter_from_idle(self):
        """Should be able to enter WiFi setup from IDLE."""
        sm, _, _ = _make_sm()

        result = sm.enter_wifi_setup()

        assert result is True
        assert sm.get_current_state() == 'wifi_setup'

    def test_exit_returns_to_idle(self):
        """Exiting WiFi setup should return to IDLE."""
        sm, _, _ = _make_sm()
        sm.enter_wifi_setup()

        result = sm.exit_wifi_setup()

        assert result is True
        assert sm.get_current_state() == 'idle'

    def test_cannot_enter_from_recording(self):
        """WiFi setup should be blocked during recording."""
        sm, _, _ = _make_sm()
        sm.handle_button_press()  # Start recording.

        result = sm.enter_wifi_setup()

        assert result is False
        assert sm.get_current_state() == 'recording'

    def test_cannot_enter_from_shutdown(self):
        """WiFi setup should be blocked after shutdown."""
        sm, _, _ = _make_sm()
        sm.shutdown()

        result = sm.enter_wifi_setup()

        assert result is False

    def test_exit_while_not_in_wifi_setup(self):
        """exit_wifi_setup() should return False if not in WIFI_SETUP."""
        sm, _, _ = _make_sm()

        result = sm.exit_wifi_setup()

        assert result is False


class TestShutdown:
    def test_from_idle(self):
        """Shutdown from IDLE should reach SHUTDOWN state."""
        sm, _, _ = _make_sm()

        sm.shutdown()

        assert sm.get_current_state() == 'shutdown'

    def test_from_recording_stops_recording(self):
        """Shutdown while recording should stop the recording first."""
        sm, _, rec_mgr = _make_sm()
        sm.handle_button_press()  # Start recording.

        sm.shutdown()

        assert rec_mgr.stop_calls == 1
        assert sm.get_current_state() == 'shutdown'

    def test_from_idle_no_stop_recording(self):
        """Shutdown from IDLE should not call stop_recording."""
        sm, _, rec_mgr = _make_sm()

        sm.shutdown()

        assert rec_mgr.stop_calls == 0

    def test_cleanup_calls_gpio_cleanup(self):
        """cleanup() should forward to GPIOController.cleanup()."""
        sm, fake_gpio, _ = _make_sm()

        sm.cleanup()

        assert fake_gpio.cleaned_up


class TestDeviceUnhealthy:
    def test_stops_recording_and_transitions_to_idle(self):
        """Device unhealthy during recording should stop and go to IDLE."""
        sm, _, rec_mgr = _make_sm()
        sm.handle_button_press()  # Start recording.

        sm.on_device_unhealthy()

        assert rec_mgr.stop_calls == 1
        assert sm.get_current_state() == 'idle'

    def test_ignored_when_not_recording(self):
        """Device unhealthy when not recording should be a no-op."""
        sm, _, rec_mgr = _make_sm()

        sm.on_device_unhealthy()

        assert rec_mgr.stop_calls == 0
        assert sm.get_current_state() == 'idle'

    def test_beeps_error_pattern(self):
        """Device unhealthy should trigger 5-beep error pattern."""
        sm, fake_gpio, _ = _make_sm()
        sm.handle_button_press()
        fake_gpio.buzzer_beeps.clear()

        sm.on_device_unhealthy()

        assert any(count == 5 for _, count in fake_gpio.buzzer_beeps)

    def test_show_alert_callback_invoked(self):
        """Device unhealthy should invoke the alert callback if set."""
        sm, _, _ = _make_sm()
        alerts: list[str] = []
        sm.set_show_alert_callback(lambda msg: alerts.append(msg))
        sm.handle_button_press()

        sm.on_device_unhealthy()

        assert len(alerts) == 1
        assert 'Camera' in alerts[0]


class TestRecordingError:
    def test_stops_recording_and_transitions_to_idle(self):
        """Recording error during recording should stop and go to IDLE."""
        sm, _, rec_mgr = _make_sm()
        sm.handle_button_press()

        sm.on_recording_error()

        assert rec_mgr.stop_calls == 1
        assert sm.get_current_state() == 'idle'

    def test_ignored_when_not_recording(self):
        """Recording error when not recording should be a no-op."""
        sm, _, rec_mgr = _make_sm()

        sm.on_recording_error()

        assert rec_mgr.stop_calls == 0
        assert sm.get_current_state() == 'idle'

    def test_beeps_error_pattern(self):
        """Recording error should trigger 5-beep error pattern."""
        sm, fake_gpio, _ = _make_sm()
        sm.handle_button_press()
        fake_gpio.buzzer_beeps.clear()

        sm.on_recording_error()

        assert any(count == 5 for _, count in fake_gpio.buzzer_beeps)

    def test_show_alert_callback_invoked(self):
        """Recording error should invoke the alert callback with error message."""
        sm, _, _ = _make_sm()
        alerts: list[str] = []
        sm.set_show_alert_callback(lambda msg: alerts.append(msg))
        sm.handle_button_press()

        sm.on_recording_error()

        assert len(alerts) == 1
        assert 'Error' in alerts[0]


class TestNetworkChange:
    def test_stores_network_status(self):
        """on_network_change should update the internal network status."""
        sm, _, _ = _make_sm()

        sm.on_network_change(
            testing.make_snapshot(
                status=network_monitor.Status.ONLINE,
            )
        )

        assert sm._network_status == network_monitor.Status.ONLINE

    def test_offline_to_online_updates_status(self):
        """Transition from offline to online should be stored."""
        sm, _, _ = _make_sm()
        sm.on_network_change(
            testing.make_snapshot(
                status=network_monitor.Status.OFFLINE,
            )
        )

        sm.on_network_change(
            testing.make_snapshot(
                status=network_monitor.Status.ONLINE,
            )
        )

        assert sm._network_status == network_monitor.Status.ONLINE


class TestShowAlertCallback:
    def test_start_recording_failure_invokes_alert(self):
        """Failed start_recording should invoke the alert callback."""
        sm, _, _ = _make_sm(start_succeeds=False)
        alerts: list[str] = []
        sm.set_show_alert_callback(lambda msg: alerts.append(msg))

        sm.handle_button_press()

        assert len(alerts) == 1
        assert 'Camera' in alerts[0]

    def test_no_alert_on_success(self):
        """Successful start_recording should not invoke the alert callback."""
        sm, _, _ = _make_sm()
        alerts: list[str] = []
        sm.set_show_alert_callback(lambda msg: alerts.append(msg))

        sm.handle_button_press()

        assert len(alerts) == 0


class TestPreview:
    def test_set_preview_callbacks_stores_all(self):
        """set_preview_callbacks should store all three callbacks."""
        sm, _, _ = _make_sm()
        sm.set_preview_callbacks(
            is_available=lambda: True,
            start=lambda: None,
            stop=lambda: None,
        )
        assert sm._is_preview_available is not None
        assert sm._start_preview_callback is not None
        assert sm._stop_preview_callback is not None

    def test_double_press_from_idle_enters_preview(self):
        """Double press in IDLE with preview wired should enter CAMERA_PREVIEW."""
        sm, _, _ = _make_sm()
        sm.set_preview_callbacks(
            is_available=lambda: True,
            start=lambda: None,
            stop=lambda: None,
        )

        sm.handle_double_press()

        assert sm.get_current_state() == 'camera_preview'

    def test_double_press_from_recording_ignored(self):
        """Double press in RECORDING should be ignored."""
        sm, _, _ = _make_sm()
        sm.handle_button_press()  # Start recording.

        sm.handle_double_press()

        assert sm.get_current_state() == 'recording'

    def test_button_press_in_preview_exits_to_idle(self):
        """Button press in CAMERA_PREVIEW should exit back to IDLE."""
        sm, _, _ = _make_sm()
        sm.set_preview_callbacks(
            is_available=lambda: True,
            start=lambda: None,
            stop=lambda: None,
        )
        sm.handle_double_press()
        assert sm.get_current_state() == 'camera_preview'

        sm.handle_button_press()

        assert sm.get_current_state() == 'idle'

    def test_preview_led_green_slow_blink(self):
        """CAMERA_PREVIEW should set green LED to SLOW_BLINK."""
        sm, fake_gpio, _ = _make_sm()
        sm.set_preview_callbacks(
            is_available=lambda: True,
            start=lambda: None,
            stop=lambda: None,
        )

        sm.handle_double_press()

        red, green = fake_gpio.led_calls[-1]
        assert red == gpio.LEDState.OFF.value
        assert green == gpio.LEDState.SLOW_BLINK.value

    def test_enter_preview_without_callbacks(self):
        """Double press without preview callbacks should stay in IDLE."""
        sm, _, _ = _make_sm()

        sm.handle_double_press()

        assert sm.get_current_state() == 'idle'

    def test_enter_preview_when_not_available(self):
        """Double press when is_available returns False should stay in IDLE."""
        sm, _, _ = _make_sm()
        sm.set_preview_callbacks(
            is_available=lambda: False,
            start=lambda: None,
            stop=lambda: None,
        )

        sm.handle_double_press()

        assert sm.get_current_state() == 'idle'

    def test_enter_preview_start_callback_exception(self):
        """Exception in start preview callback should stay in IDLE with error beep."""
        sm, fake_gpio, _ = _make_sm()

        def _raise():
            raise RuntimeError('preview crash')

        sm.set_preview_callbacks(
            is_available=lambda: True,
            start=_raise,
            stop=lambda: None,
        )

        sm.handle_double_press()

        assert sm.get_current_state() == 'idle'
        assert any(count == 5 for _, count in fake_gpio.buzzer_beeps)

    def test_exit_preview_stop_callback_exception(self):
        """Exception in stop callback should still transition to IDLE."""
        sm, _, _ = _make_sm()

        def _raise():
            raise RuntimeError('stop crash')

        sm.set_preview_callbacks(
            is_available=lambda: True,
            start=lambda: None,
            stop=_raise,
        )
        sm.handle_double_press()
        assert sm.get_current_state() == 'camera_preview'

        sm.handle_button_press()

        assert sm.get_current_state() == 'idle'

    def test_shutdown_exits_preview(self):
        """shutdown() should exit preview if in CAMERA_PREVIEW."""
        sm, _, _ = _make_sm()
        stop_called = []
        sm.set_preview_callbacks(
            is_available=lambda: True,
            start=lambda: None,
            stop=lambda: stop_called.append(True),
        )
        sm.handle_double_press()

        sm.shutdown()

        assert sm.get_current_state() == 'shutdown'
        assert stop_called
