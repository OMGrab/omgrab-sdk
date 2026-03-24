"""GPIO management for button monitoring and connection status."""

from typing import Optional

import dataclasses
import logging
import time
from collections.abc import Callable

from omgrab.gpio import gpio
from omgrab.runtime import wifi_connect
from omgrab.workflows import state_machine

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ButtonConfig:
    """Configuration for button press detection.

    Attributes:
        long_press_threshold_s: Hold duration to trigger WiFi reconfiguration.
        double_press_window_s: Window after a first tap in which a second tap
            counts as a double-press. Only used in IDLE state.
        check_interval_s: How often to poll the button GPIO (seconds).
    """

    long_press_threshold_s: float = 5.0
    double_press_window_s: float = 0.5
    check_interval_s: float = 0.02


def button_monitor_loop(
    gpio_controller: gpio.GPIOController,
    state_machine_obj: state_machine.StateMachine,
    config: Optional[ButtonConfig] = None,
    on_wifi_setup_complete: Optional[Callable[[bool], None]] = None,
    on_button_press: Optional[Callable[[], None]] = None,
    on_wifi_hotspot_started: Optional[Callable[[], None]] = None,
    wifi_manager: Optional[wifi_connect.WifiManager] = None,
):
    """Monitor button presses and delegate to state machine.

    Short press: Start/stop recording (handled by state machine)
    Double press (IDLE only): Camera preview (handled by state machine)
    Long press (5+ seconds): Trigger WiFi reconfiguration portal

    Args:
        gpio_controller: GPIO controller instance.
        state_machine_obj: State machine for managing workflow.
        config: Button detection configuration.
        on_wifi_setup_complete: Optional callback invoked when WiFi setup completes.
        on_button_press: Optional callback invoked on any button press (rising edge).
        on_wifi_hotspot_started: Optional callback invoked when WiFi hotspot is started.
        wifi_manager: Optional WifiManager instance for WiFi setup. If None,
            long-press and WiFi cancel are no-ops.
    """
    config = config or ButtonConfig()
    last_state = False
    press_start_time: float | None = None
    long_press_triggered = False

    # Double-press detection state.
    pending_tap = False
    pending_tap_time: float = 0.0

    logger.info('Button monitor started')
    logger.info(
        'Long press threshold: %.1fs for WiFi reconfiguration', config.long_press_threshold_s
    )

    while True:
        try:
            current_state = gpio_controller.read_button()

            # Detect rising edge (button press start)
            if current_state and not last_state:
                press_start_time = time.time()
                long_press_triggered = False
                logger.debug('Button pressed - hold for 5s for WiFi setup')
                if on_button_press is not None:
                    on_button_press()

            # Button is being held - check for long press
            if current_state and press_start_time is not None:
                hold_duration = time.time() - press_start_time
                if hold_duration >= config.long_press_threshold_s and not long_press_triggered:
                    # Long press detected - trigger WiFi reconfiguration
                    long_press_triggered = True
                    logger.info(
                        'Long press detected (%.1fs) - triggering WiFi setup', hold_duration
                    )

                    if wifi_manager is not None:
                        # Enter WiFi setup state (updates LEDs to both blinking)
                        if state_machine_obj.enter_wifi_setup():
                            gpio_controller.buzzer_beep(duration=0.3, count=3)  # Audio feedback
                            if on_wifi_hotspot_started is not None:
                                on_wifi_hotspot_started()

                            # Define callback to exit WiFi setup state when complete
                            def on_wifi_complete(success: bool):
                                logger.info('WiFi setup completed (success=%s)', success)
                                state_machine_obj.exit_wifi_setup()
                                # Trigger external callback (e.g., software update)
                                if on_wifi_setup_complete is not None:
                                    on_wifi_setup_complete(success)

                            started = wifi_manager.start_wifi_connect(
                                force_mode=True, callback=on_wifi_complete
                            )
                            if not started:
                                logger.warning(
                                    'WiFi connect service failed to start, exiting WiFi setup'
                                )
                                state_machine_obj.exit_wifi_setup()
                        else:
                            logger.warning('Cannot enter WiFi setup state')
                    else:
                        logger.debug('WiFi manager not configured, ignoring long press')

            # Detect falling edge (button release)
            if not current_state and last_state:
                if press_start_time is not None:
                    hold_duration = time.time() - press_start_time
                    if not long_press_triggered:
                        # TODO(isaac): Checking get_current_state() here increases coupling between
                        # gpio_manager and the state machine (gpio_manager now knows about workflow
                        # semantics). We accept this to avoid adding 500ms latency to stop-recording
                        # and exit-preview in non-IDLE states.
                        current_workflow_state = state_machine_obj.get_current_state()
                        if current_workflow_state == 'idle':
                            now = time.time()
                            if (
                                pending_tap
                                and now - pending_tap_time < config.double_press_window_s
                            ):
                                # Second tap within window -- double-press.
                                logger.debug(
                                    'Double press detected (%.2fs + %.2fs)',
                                    pending_tap_time,
                                    hold_duration,
                                )
                                pending_tap = False
                                state_machine_obj.handle_double_press()
                            else:
                                # First tap -- start the double-press window.
                                logger.debug(
                                    'Short press (%.2fs) - waiting for double press', hold_duration
                                )
                                pending_tap = True
                                pending_tap_time = now
                        elif current_workflow_state == 'wifi_setup':
                            # Cancel WiFi setup on button press.
                            logger.info('Button press during WiFi setup - cancelling')
                            if wifi_manager is not None:
                                wifi_manager.stop_wifi_connect()
                            state_machine_obj.exit_wifi_setup()
                            gpio_controller.buzzer_beep(duration=0.1, count=2)
                        else:
                            # Not IDLE -- fire immediately, no double-press detection.
                            logger.debug(
                                'Short press (%.2fs) - handling as button press', hold_duration
                            )
                            state_machine_obj.handle_button_press()
                    else:
                        logger.debug(
                            'Long press released (%.1fs) - WiFi setup triggered', hold_duration
                        )
                press_start_time = None

            # Expire pending tap after the double-press window.
            if pending_tap and time.time() - pending_tap_time >= config.double_press_window_s:
                logger.debug('Double-press window expired - handling as single press')
                pending_tap = False
                state_machine_obj.handle_button_press()

            last_state = current_state
            time.sleep(config.check_interval_s)

        except Exception as e:
            logger.error('Error in button monitor: %s', e)
            time.sleep(1.0)
