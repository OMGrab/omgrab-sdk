"""GPIO control module for Raspberry Pi 5 LED, buzzer, and button management.

This module uses a separate process for GPIO control to avoid GIL contention
with video encoding and other CPU-intensive operations in the main process.
"""

import enum

from omgrab.gpio import gpio_process


class LEDState(enum.Enum):
    """LED state patterns."""
    OFF = 'off'
    ON = 'on'
    SLOW_BLINK = 'slow_blink'  # ~1 Hz
    FAST_BLINK = 'fast_blink'  # ~5 Hz


class GPIOController:
    """Controller for GPIO components: LEDs, buzzer, and push button.

    This controller delegates all GPIO operations to a separate process,
    ensuring smooth LED blinking even during heavy CPU load from video
    encoding or other operations that would normally cause GIL contention.
    """

    # GPIO pin assignments (BCM numbering) - for reference/documentation
    PIN_GREEN_LED = 17  # Physical pin 11
    PIN_RED_LED = 23    # Physical pin 16
    PIN_BUZZER = 12     # Physical pin 32 (PWM capable)
    PIN_BUTTON = 24     # Physical pin 18

    def __init__(self):
        """Initialize GPIO controller with a separate process."""
        self._process_controller = gpio_process.GPIOControllerProcess()

    def cleanup(self):
        """Cleanup GPIO resources."""
        self._process_controller.cleanup()

    def set_led_states(self, red: LEDState, green: LEDState):
        """Set both LED states.

        Args:
            red: Desired state for red LED.
            green: Desired state for green LED.
        """
        self._process_controller.set_led_states(red.value, green.value)

    def set_buzzer(self, state: bool):
        """Set buzzer state.

        Args:
            state: True for ON, False for OFF.
        """
        self._process_controller.set_buzzer(state)

    def set_buzzer_volume(self, volume: float):
        """Set buzzer volume via PWM duty cycle.

        Args:
            volume: Volume level from 0.0 (silent) to 1.0 (full).
        """
        self._process_controller.set_buzzer_volume(volume)

    def set_buzzer_tone(self, freq: float):
        """Set buzzer tone frequency.

        Args:
            freq: Tone frequency in Hz (clamped to 20-20000).
        """
        self._process_controller.set_buzzer_tone(freq)

    def read_button(self) -> bool:
        """Read the state of the push button.

        Returns:
            True if button is pressed, False otherwise.
        """
        result: bool = self._process_controller.read_button()
        return result

    def buzzer_beep(self, duration: float = 0.1, count: int = 1):
        """Sound the buzzer for specified duration and count.

        Args:
            duration: Duration of each beep in seconds.
            count: Number of beeps.
        """
        self._process_controller.buzzer_beep(duration, count)
