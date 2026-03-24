"""State machine for omgrab workflow management."""

from typing import Optional

import enum
import logging
import threading
from collections.abc import Callable

from omgrab.gpio import gpio
from omgrab.runtime import network_monitor as network_monitor_module
from omgrab.runtime import recording_manager

logger = logging.getLogger(__name__)


class WorkflowState(enum.Enum):
    """High-level workflow states."""

    BOOT = 'boot'
    IDLE = 'idle'
    RECORDING = 'recording'
    CAMERA_PREVIEW = 'camera_preview'
    WIFI_SETUP = 'wifi_setup'
    SHUTDOWN = 'shutdown'


class StateMachine:
    """Manages omgrab state transitions and GPIO updates."""

    def __init__(
        self,
        gpio_controller: gpio.GPIOController,
        recording_manager: recording_manager.RecordingController,
    ):
        """Initialize state machine.

        Args:
            gpio_controller: GPIO controller instance for LED/buzzer control.
            recording_manager: Recording controller for start/stop recording.
        """
        self._lock = threading.Lock()
        self._gpio = gpio_controller
        self._recording_manager = recording_manager

        self._state = WorkflowState.BOOT
        self._show_alert_callback: Optional[Callable[[str], None]] = None

        # Camera preview callbacks (wired via set_preview_callbacks).
        self._is_preview_available: Optional[Callable[[], bool]] = None
        self._start_preview_callback: Optional[Callable[[], None]] = None
        self._stop_preview_callback: Optional[Callable[[], None]] = None

        self._network_status = network_monitor_module.Status.OFFLINE

        with self._lock:
            side_effects = self._transition_to_locked(WorkflowState.BOOT, allow_same_state=True)
        self._run_side_effects(side_effects)

    def _run_side_effects(self, side_effects: list[Callable[[], None]]):
        """Run side-effect callables outside the lock."""
        for effect in side_effects:
            effect()

    def _on_change_locked(self) -> list[Callable[[], None]]:
        """Callback for changes to the state machine.

        NOTE: This method must be called with the lock held.
        """
        # NOTE: _transition_to_locked() calls _on_change_locked(), so if this
        # method triggers a transition, GPIO updates can be collected more than
        # once. Keep in mind if we notice duplicate LED updates.
        side_effects: list[Callable[[], None]] = []
        side_effects.extend(self._check_automatic_state_changes_locked())
        side_effects.extend(self._collect_gpio_changes_locked())
        return side_effects

    def on_network_change(self, snapshot: network_monitor_module.Snapshot):
        """Callback for network state changes."""
        with self._lock:
            old_status = self._network_status
            self._network_status = snapshot.status
            side_effects = self._on_change_locked()
        logger.info(
            'Network state changed (%s -> %s) (%s)',
            old_status.name,
            snapshot.status.name,
            snapshot.detail,
        )
        self._run_side_effects(side_effects)

    def _transition_to_locked(
        self, new_state: WorkflowState, allow_same_state: bool = False
    ) -> list[Callable[[], None]]:
        """Internal method to transition to a new state with consistent logging.

        NOTE: This method must be called with the lock held.
        NOTE: This method calls _on_change_locked(), which can trigger additional
        transitions (e.g., boot -> idle). The recursion currently terminates
        because the state changes, but be careful adding new auto transitions.

        Args:
            new_state: The state to transition to.
            allow_same_state: Whether to allow transitioning to the same state.

        Returns:
            List of side effects to run after releasing the lock.
        """
        if self._state == new_state:
            logger.warning('Attempted to transition to current state: %s', new_state)
            if not allow_same_state:
                return []
            logger.info('Continuing to current state: %s (allowed explicitly by caller)', new_state)

        old_state_name = self._state.value.upper()
        new_state_name = new_state.value.upper()
        self._state = new_state
        logger.info('Transition: %s -> %s', old_state_name, new_state_name)
        return self._on_change_locked()

    def is_recording(self) -> bool:
        """Check if currently in RECORDING state.

        Returns:
            True if in RECORDING workflow state.
        """
        with self._lock:
            return self._state == WorkflowState.RECORDING

    def get_current_state(self) -> str:
        """Get the current workflow state.

        Returns:
            Current state as a string
            (boot/idle/recording/camera_preview/wifi_setup/shutdown).
        """
        with self._lock:
            return self._state.value

    def set_show_alert_callback(self, callback: Callable[[str], None]):
        """Set callback for displaying alert messages on screen.

        Args:
            callback: Called with a warning message string when the
                state machine needs to alert the user (e.g. camera error).
        """
        self._show_alert_callback = callback

    def set_preview_callbacks(
        self, is_available: Callable[[], bool], start: Callable[[], None], stop: Callable[[], None]
    ):
        """Set callbacks for camera preview lifecycle.

        Args:
            is_available: Returns True if the display hardware is connected
                and preview can be started.
            start: Called to start the preview pipeline and display.
            stop: Called to stop the preview pipeline and display.
        """
        self._is_preview_available = is_available
        self._start_preview_callback = start
        self._stop_preview_callback = stop

    def handle_button_press(self):
        """Handle single button press event."""
        with self._lock:
            state = self._state

        logger.debug('Button pressed in state: %s', state)

        if state == WorkflowState.IDLE:
            self._start_recording()
        elif state == WorkflowState.RECORDING:
            self._stop_recording()
        elif state == WorkflowState.CAMERA_PREVIEW:
            self._exit_preview()
        else:
            logger.debug('Button press ignored in state: %s', state)

    def handle_double_press(self):
        """Handle double button press event (camera preview)."""
        with self._lock:
            state = self._state

        logger.debug('Double press in state: %s', state)

        if state == WorkflowState.IDLE:
            self._enter_preview()
        else:
            logger.debug('Double press ignored in state: %s', state)

    def _start_recording(self) -> bool:
        """Start recording via the recording manager.

        Returns:
            True if recording started, False if unable to start.
        """
        with self._lock:
            if self._state != WorkflowState.IDLE:
                logger.warning('Cannot start recording from state: %s', self._state)
                return False

        # Delegate to recording manager (handles device, threads)
        success = self._recording_manager.start_recording()

        if not success:
            logger.warning('Recording manager failed to start recording')
            self._gpio.buzzer_beep(duration=0.1, count=5)  # Failure pattern
            if self._show_alert_callback is not None:
                self._show_alert_callback('Check Camera\nConnection')
            return False

        # Transition state and beep
        with self._lock:
            side_effects = self._transition_to_locked(WorkflowState.RECORDING)

        self._gpio.buzzer_beep(duration=0.1, count=1)
        self._run_side_effects(side_effects)
        return True

    def _stop_recording(self) -> bool:
        """Stop recording via the recording manager.

        Returns:
            True if recording stopped, False if not recording.
        """
        with self._lock:
            if self._state != WorkflowState.RECORDING:
                logger.warning('Cannot stop recording from state: %s', self._state)
                return False

            side_effects = self._transition_to_locked(WorkflowState.IDLE)

        # Delegate to recording manager (handles device, cleanup)
        self._recording_manager.stop_recording()

        self._gpio.buzzer_beep(duration=0.1, count=2)
        self._run_side_effects(side_effects)
        return True

    def _enter_preview(self) -> bool:
        """Enter camera preview mode.

        Returns:
            True if preview started, False if unable to start.
        """
        with self._lock:
            if self._state != WorkflowState.IDLE:
                logger.warning('Cannot enter preview from state: %s', self._state)
                return False

        if self._start_preview_callback is None:
            logger.warning('Preview not available (callbacks not wired)')
            return False

        if self._is_preview_available is not None and not self._is_preview_available():
            logger.info('Preview ignored (display not connected)')
            return False

        try:
            self._start_preview_callback()
        except Exception as e:
            logger.warning('Failed to start preview: %s', e)
            self._gpio.buzzer_beep(duration=0.1, count=5)  # Failure pattern
            if self._show_alert_callback is not None:
                self._show_alert_callback('Check Camera\nConnection')
            return False

        with self._lock:
            side_effects = self._transition_to_locked(WorkflowState.CAMERA_PREVIEW)

        self._gpio.buzzer_beep(duration=0.1, count=2)
        self._run_side_effects(side_effects)
        return True

    def _exit_preview(self) -> bool:
        """Exit camera preview mode.

        Returns:
            True if preview stopped, False if not in preview.
        """
        with self._lock:
            if self._state != WorkflowState.CAMERA_PREVIEW:
                logger.warning('Cannot exit preview from state: %s', self._state)
                return False

            side_effects = self._transition_to_locked(WorkflowState.IDLE)

        if self._stop_preview_callback is not None:
            try:
                self._stop_preview_callback()
            except Exception as e:
                logger.warning('Error stopping preview: %s', e)

        self._gpio.buzzer_beep(duration=0.1, count=1)
        self._run_side_effects(side_effects)
        return True

    def on_device_unhealthy(self):
        """Handle device becoming unhealthy during recording.

        Called by RecordingManager's health monitor. This method is responsible
        for stopping the recording (the health monitor only detects and signals).
        """
        logger.warning('Device unhealthy callback triggered')

        with self._lock:
            if self._state != WorkflowState.RECORDING:
                # Recording already stopped or state changed
                logger.debug('Device unhealthy but not in RECORDING state: %s', self._state)
                return

            side_effects = self._transition_to_locked(WorkflowState.IDLE)

        # Stop the recording (health monitor already stopped itself)
        self._recording_manager.stop_recording()

        # Beep failure pattern and show alert
        self._gpio.buzzer_beep(duration=0.1, count=5)
        if self._show_alert_callback is not None:
            self._show_alert_callback('Check Camera\nConnection')
        self._run_side_effects(side_effects)

    def enter_wifi_setup(self) -> bool:
        """Enter WiFi setup state (triggered by long press).

        Returns:
            True if transitioned to WIFI_SETUP, False if not possible.
        """
        with self._lock:
            # Only allow entering WiFi setup from IDLE or BOOT states
            if self._state not in (WorkflowState.IDLE, WorkflowState.BOOT):
                logger.warning('Cannot enter WiFi setup from state: %s', self._state)
                return False
            side_effects = self._transition_to_locked(WorkflowState.WIFI_SETUP)
        self._run_side_effects(side_effects)
        return True

    def exit_wifi_setup(self) -> bool:
        """Exit WiFi setup state (called when wifi-connect completes).

        Returns:
            True if exited WIFI_SETUP, False if not in WIFI_SETUP state.
        """
        with self._lock:
            if self._state != WorkflowState.WIFI_SETUP:
                logger.warning('Cannot exit WiFi setup from state: %s', self._state)
                return False
            side_effects = self._transition_to_locked(WorkflowState.IDLE)
        self._run_side_effects(side_effects)
        return True

    def shutdown(self):
        """Public method to shutdown the state machine."""
        self._exit_preview()
        self._stop_recording()
        with self._lock:
            side_effects = self._transition_to_locked(WorkflowState.SHUTDOWN)
        self._run_side_effects(side_effects)

    def _collect_gpio_changes_locked(self) -> list[Callable[[], None]]:
        """Collect GPIO changes based on current state.

        NOTE: This method must be called with the lock held.

        Returns:
            List of side effects to run after releasing the lock.
        """
        side_effects: list[Callable[[], None]] = []
        red = gpio.LEDState.OFF
        green = gpio.LEDState.OFF
        buzzer_off = False

        # Boot state.
        if self._state == WorkflowState.BOOT or self._state == WorkflowState.IDLE:
            red = gpio.LEDState.OFF
            green = gpio.LEDState.ON
        # Recording state.
        elif self._state == WorkflowState.RECORDING:
            red = gpio.LEDState.SLOW_BLINK
            green = gpio.LEDState.OFF
        # Camera preview state (double press triggered).
        elif self._state == WorkflowState.CAMERA_PREVIEW:
            red = gpio.LEDState.OFF
            green = gpio.LEDState.SLOW_BLINK
        # WiFi setup state (long press triggered).
        elif self._state == WorkflowState.WIFI_SETUP:
            red = gpio.LEDState.FAST_BLINK
            green = gpio.LEDState.FAST_BLINK
        # Shutdown state.
        elif self._state == WorkflowState.SHUTDOWN:
            red = gpio.LEDState.OFF
            green = gpio.LEDState.OFF
            buzzer_off = True

        def set_leds(red_state: gpio.LEDState, green_state: gpio.LEDState):
            self._gpio.set_led_states(red=red_state, green=green_state)

        side_effects.append(lambda: set_leds(red, green))
        if buzzer_off:
            side_effects.append(lambda: self._gpio.set_buzzer(False))

        return side_effects

    def cleanup(self):
        """Cleanup resources."""
        self._gpio.cleanup()

    def _check_automatic_state_changes_locked(self) -> list[Callable[[], None]]:
        """Handle automatic state changes.

        Currently handles:
        - BOOT -> IDLE: Immediately after boot.

        NOTE: This method must be called with the lock held.
        """
        side_effects: list[Callable[[], None]] = []
        if self._state == WorkflowState.BOOT:
            # For now we go to IDLE immediately on boot.
            transition_side_effects = self._transition_to_locked(WorkflowState.IDLE)
            side_effects.extend(transition_side_effects)

        return side_effects
