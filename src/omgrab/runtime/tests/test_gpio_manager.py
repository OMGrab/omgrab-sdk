"""Tests for the GPIO button monitor (runtime/gpio_manager.py)."""
import threading
import time

import pytest

from omgrab import testing
from omgrab.runtime import gpio_manager


class FakeStateMachine:
    """Minimal fake for StateMachine used by button_monitor_loop."""

    def __init__(self, *, initial_state: str = 'idle'):
        self.button_presses: list[float] = []
        self.double_presses: list[float] = []
        self.wifi_setup_entered: list[bool] = []
        self.wifi_setup_exited: list[bool] = []
        self._allow_wifi_setup = True
        self._state = initial_state

    def get_current_state(self) -> str:
        """Return the current workflow state."""
        return self._state

    def handle_button_press(self):
        """Record short press."""
        self.button_presses.append(time.monotonic())

    def handle_double_press(self):
        """Record double press."""
        self.double_presses.append(time.monotonic())

    def enter_wifi_setup(self) -> bool:
        """Record WiFi setup entry."""
        self.wifi_setup_entered.append(True)
        return self._allow_wifi_setup

    def exit_wifi_setup(self):
        """Record WiFi setup exit."""
        self.wifi_setup_exited.append(True)
        self._state = 'idle'


class FakeWifiManager:
    """Minimal fake for WifiManager."""

    def __init__(self, *, start_succeeds: bool = True):
        self.start_calls: list[dict] = []
        self.stop_calls: list[bool] = []
        self._start_succeeds = start_succeeds

    def start_wifi_connect(self, force_mode=True, callback=None):
        self.start_calls.append({'force_mode': force_mode, 'callback': callback})
        return self._start_succeeds

    def stop_wifi_connect(self):
        self.stop_calls.append(True)
        return True

    def is_wifi_connect_running(self):
        return False



class TestConstants:

    def test_long_press_threshold_is_five_seconds(self):
        """Long press should be 5 seconds by default."""
        assert gpio_manager.ButtonConfig().long_press_threshold_s == 5.0



class TestShortPress:

    def test_short_press_calls_handle_button_press(self):
        """A brief button press-and-release should invoke handle_button_press."""
        gpio_ctrl = testing.FakeGPIOController()
        sm = FakeStateMachine()

        def run_monitor():
            try:
                gpio_manager.button_monitor_loop(
                    gpio_controller=gpio_ctrl,
                    state_machine_obj=sm,
                    config=gpio_manager.ButtonConfig(check_interval_s=0.01),
                )
            except Exception:
                pass

        thread = threading.Thread(target=run_monitor, daemon=True)

        # Press the button briefly.
        gpio_ctrl.button_pressed = True
        thread.start()
        time.sleep(0.05)
        gpio_ctrl.button_pressed = False

        # In IDLE, single presses wait for the double-press window (0.5s)
        # before firing handle_button_press.
        time.sleep(gpio_manager.ButtonConfig().double_press_window_s + 0.1)

        assert len(sm.button_presses) == 1

    def test_on_button_press_callback_invoked(self):
        """The on_button_press callback should fire on rising edge."""
        gpio_ctrl = testing.FakeGPIOController()
        sm = FakeStateMachine()
        presses: list[bool] = []

        def run_monitor():
            try:
                gpio_manager.button_monitor_loop(
                    gpio_controller=gpio_ctrl,
                    state_machine_obj=sm,
                    config=gpio_manager.ButtonConfig(check_interval_s=0.01),
                    on_button_press=lambda: presses.append(True),
                )
            except Exception:
                pass

        thread = threading.Thread(target=run_monitor, daemon=True)
        gpio_ctrl.button_pressed = True
        thread.start()
        time.sleep(0.05)
        gpio_ctrl.button_pressed = False
        time.sleep(0.05)

        assert len(presses) == 1



class TestLongPress:

    @pytest.mark.slow
    def test_long_press_triggers_wifi_setup(self):
        """Holding the button for >= 5s should trigger WiFi setup."""
        gpio_ctrl = testing.FakeGPIOController()
        sm = FakeStateMachine()
        wifi_mgr = FakeWifiManager()

        def run_monitor():
            try:
                gpio_manager.button_monitor_loop(
                    gpio_controller=gpio_ctrl,
                    state_machine_obj=sm,
                    config=gpio_manager.ButtonConfig(check_interval_s=0.01),
                    wifi_manager=wifi_mgr,
                )
            except Exception:
                pass

        thread = threading.Thread(target=run_monitor, daemon=True)
        gpio_ctrl.button_pressed = True
        thread.start()

        # Hold for > 5 seconds.
        time.sleep(5.3)
        gpio_ctrl.button_pressed = False
        time.sleep(0.1)

        assert len(sm.wifi_setup_entered) >= 1
        assert len(wifi_mgr.start_calls) >= 1
        # Short press should NOT have been triggered for this long press.
        assert len(sm.button_presses) == 0

    @pytest.mark.slow
    def test_long_press_without_wifi_manager_is_noop(self):
        """Long press without wifi_manager should not crash or trigger setup."""
        gpio_ctrl = testing.FakeGPIOController()
        sm = FakeStateMachine()

        def run_monitor():
            try:
                gpio_manager.button_monitor_loop(
                    gpio_controller=gpio_ctrl,
                    state_machine_obj=sm,
                    config=gpio_manager.ButtonConfig(check_interval_s=0.01),
                    wifi_manager=None,
                )
            except Exception:
                pass

        thread = threading.Thread(target=run_monitor, daemon=True)
        gpio_ctrl.button_pressed = True
        thread.start()

        time.sleep(5.3)
        gpio_ctrl.button_pressed = False
        time.sleep(0.1)

        assert len(sm.wifi_setup_entered) == 0
        assert len(sm.button_presses) == 0


class TestWifiSetupCancellation:

    def test_button_press_cancels_wifi_setup(self):
        """A button press during WiFi setup should cancel it."""
        gpio_ctrl = testing.FakeGPIOController()
        sm = FakeStateMachine(initial_state='wifi_setup')
        wifi_mgr = FakeWifiManager()

        def run_monitor():
            try:
                gpio_manager.button_monitor_loop(
                    gpio_controller=gpio_ctrl,
                    state_machine_obj=sm,
                    config=gpio_manager.ButtonConfig(check_interval_s=0.01),
                    wifi_manager=wifi_mgr,
                )
            except Exception:
                pass

        thread = threading.Thread(target=run_monitor, daemon=True)
        thread.start()

        # Short press while in wifi_setup.
        gpio_ctrl.button_pressed = True
        time.sleep(0.05)
        gpio_ctrl.button_pressed = False
        time.sleep(0.1)

        assert len(sm.wifi_setup_exited) == 1
        assert sm.get_current_state() == 'idle'

    def test_button_press_during_wifi_setup_stops_service(self):
        """Button press during WiFi setup should stop the wifi service."""
        gpio_ctrl = testing.FakeGPIOController()
        sm = FakeStateMachine(initial_state='wifi_setup')
        wifi_mgr = FakeWifiManager()

        def run_monitor():
            try:
                gpio_manager.button_monitor_loop(
                    gpio_controller=gpio_ctrl,
                    state_machine_obj=sm,
                    config=gpio_manager.ButtonConfig(check_interval_s=0.01),
                    wifi_manager=wifi_mgr,
                )
            except Exception:
                pass

        thread = threading.Thread(target=run_monitor, daemon=True)
        thread.start()

        gpio_ctrl.button_pressed = True
        time.sleep(0.05)
        gpio_ctrl.button_pressed = False
        time.sleep(0.1)

        assert len(wifi_mgr.stop_calls) == 1

    def test_button_press_during_wifi_setup_beeps(self):
        """Button press during WiFi setup should provide buzzer feedback."""
        gpio_ctrl = testing.FakeGPIOController()
        sm = FakeStateMachine(initial_state='wifi_setup')
        wifi_mgr = FakeWifiManager()

        def run_monitor():
            try:
                gpio_manager.button_monitor_loop(
                    gpio_controller=gpio_ctrl,
                    state_machine_obj=sm,
                    config=gpio_manager.ButtonConfig(check_interval_s=0.01),
                    wifi_manager=wifi_mgr,
                )
            except Exception:
                pass

        thread = threading.Thread(target=run_monitor, daemon=True)
        thread.start()

        gpio_ctrl.button_pressed = True
        time.sleep(0.05)
        gpio_ctrl.button_pressed = False
        time.sleep(0.1)

        assert any(count == 2 for _, count in gpio_ctrl.buzzer_beeps)
