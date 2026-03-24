"""Separate process for GPIO control - immune to GIL contention.

This module runs LED blinking logic in a dedicated process, completely
isolated from the main Python interpreter's GIL. This ensures smooth
LED timing even when the main process is doing heavy video encoding.
"""

from typing import Optional

import enum
import multiprocessing
import os
import pathlib
import time


class GPIOCommand(enum.Enum):
    """Commands that can be sent to the GPIO process."""

    SET_LED_STATES = 'set_led_states'
    SET_BUZZER = 'set_buzzer'
    SET_BUZZER_VOLUME = 'set_buzzer_volume'
    SET_BUZZER_TONE = 'set_buzzer_tone'
    BUZZER_BEEP = 'buzzer_beep'
    READ_BUTTON = 'read_button'
    SHUTDOWN = 'shutdown'


class LEDPattern(enum.Enum):
    """LED state patterns (must match LEDState in gpio.py)."""

    OFF = 'off'
    ON = 'on'
    SLOW_BLINK = 'slow_blink'
    FAST_BLINK = 'fast_blink'


# GPIO pin assignments (BCM numbering)
PIN_GREEN_LED = 17  # Physical pin 11
PIN_RED_LED = 23  # Physical pin 16
PIN_BUZZER = 12  # Physical pin 32 (PWM capable)
PIN_BUTTON = 24  # Physical pin 18

# Blink timing (seconds)
SLOW_BLINK_INTERVAL = 1.0
FAST_BLINK_INTERVAL = 0.2
LOOP_INTERVAL = 0.005  # 5ms for smooth blinking

# Buzzer PWM settings
BUZZER_PWM_FREQ = 2000  # 2kHz tone frequency
DEFAULT_BUZZER_VOLUME = 0.1  # 0.0 to 1.0 (duty cycle)

# Button trigger file path (for remote simulation)
BUTTON_TRIGGER_FILE = pathlib.Path('/button-trigger/press')


def _buzzer_pwm_beep(buzzer_line, duration: float, volume: float, freq: float):
    """Sound the buzzer using software PWM for volume and tone control.

    Uses absolute-time toggle points so that the waveform self-corrects
    after any OS scheduling jitter. If the process gets preempted
    mid-cycle, subsequent toggle times are computed from the ideal
    schedule rather than relative to when the previous toggle actually
    happened, preventing frequency drift.

    Args:
        buzzer_line: The gpiod line for the buzzer.
        duration: Duration of the beep in seconds.
        volume: Volume level from 0.0 (silent) to 1.0 (full).
            Controls PWM duty cycle: higher = louder.
        freq: Tone frequency in Hz (e.g. 440 for A4).
    """
    import gpiod.line

    if volume <= 0:
        time.sleep(duration)
        return

    volume = min(1.0, max(0.0, volume))
    period = 1.0 / freq
    on_time = period * volume
    off_time = period - on_time

    start = time.perf_counter()
    end_time = start + duration

    # Begin with buzzer on; first toggle is after on_time.
    buzzer_line.set_value(PIN_BUZZER, gpiod.line.Value.ACTIVE)
    is_on = True
    next_toggle = start + on_time

    while True:
        now = time.perf_counter()
        if now >= end_time:
            break
        if now >= next_toggle:
            if is_on:
                buzzer_line.set_value(PIN_BUZZER, gpiod.line.Value.INACTIVE)
                next_toggle += off_time
            else:
                buzzer_line.set_value(PIN_BUZZER, gpiod.line.Value.ACTIVE)
                next_toggle += on_time
            is_on = not is_on

    buzzer_line.set_value(PIN_BUZZER, gpiod.line.Value.INACTIVE)


def _calculate_led_state(pattern: LEDPattern, current_time: float) -> bool:
    """Calculate whether LED should be on based on pattern and time.

    Args:
        pattern: LED pattern to apply.
        current_time: Current monotonic time in seconds.

    Returns:
        True if LED should be ON, False if OFF.
    """
    if pattern == LEDPattern.OFF:
        return False
    elif pattern == LEDPattern.ON:
        return True
    elif pattern in (LEDPattern.SLOW_BLINK, LEDPattern.FAST_BLINK):
        interval = SLOW_BLINK_INTERVAL if pattern == LEDPattern.SLOW_BLINK else FAST_BLINK_INTERVAL
        time_ms = int(current_time * 1000)
        cycle_ms = int(interval * 1000 * 2)  # Full cycle = 2x interval
        position_in_cycle = time_ms % cycle_ms
        return position_in_cycle < (cycle_ms // 2)
    return False


def _gpio_process_main(command_queue: multiprocessing.Queue, response_queue: multiprocessing.Queue):
    """Main function for the GPIO process.

    This runs in a separate process, completely isolated from the main
    Python interpreter's GIL.

    Args:
        command_queue: Queue to receive commands from main process.
        response_queue: Queue to send responses back to main process.
    """
    import gpiod
    import gpiod.line

    # Elevate to real-time scheduling for clean PWM timing.
    # SCHED_FIFO at priority 1 (lowest RT level) prevents the kernel
    # from preempting this process during short busy-wait PWM cycles
    # while still yielding to any higher-priority RT tasks.
    # Requires root / CAP_SYS_NICE (the container runs privileged).
    if hasattr(os, 'sched_setscheduler'):
        try:
            os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(1))
        except (OSError, PermissionError):
            pass  # Fall back to normal scheduling if not permitted

    # Initialize GPIO
    chip = gpiod.Chip('/dev/gpiochip4')

    red_led_line = chip.request_lines(
        consumer='red_led',
        config={
            PIN_RED_LED: gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT, output_value=gpiod.line.Value.INACTIVE
            )
        },
    )
    green_led_line = chip.request_lines(
        consumer='green_led',
        config={
            PIN_GREEN_LED: gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT, output_value=gpiod.line.Value.INACTIVE
            )
        },
    )
    buzzer_line = chip.request_lines(
        consumer='buzzer',
        config={
            PIN_BUZZER: gpiod.LineSettings(
                direction=gpiod.line.Direction.OUTPUT, output_value=gpiod.line.Value.INACTIVE
            )
        },
    )
    button_line = chip.request_lines(
        consumer='button',
        config={
            PIN_BUTTON: gpiod.LineSettings(
                direction=gpiod.line.Direction.INPUT, bias=gpiod.line.Bias.PULL_UP
            )
        },
    )

    # LED state patterns
    red_pattern = LEDPattern.OFF
    green_pattern = LEDPattern.OFF
    red_last_physical: Optional[bool] = None
    green_last_physical: Optional[bool] = None

    # Buzzer settings
    buzzer_volume = DEFAULT_BUZZER_VOLUME
    buzzer_tone = BUZZER_PWM_FREQ

    running = True

    while running:
        # Check for commands (non-blocking)
        try:
            while not command_queue.empty():
                cmd, args = command_queue.get_nowait()

                if cmd == GPIOCommand.SET_LED_STATES:
                    red_str, green_str = args
                    red_pattern = LEDPattern(red_str)
                    green_pattern = LEDPattern(green_str)
                elif cmd == GPIOCommand.SET_BUZZER:
                    buzzer_line.set_value(
                        PIN_BUZZER, gpiod.line.Value.ACTIVE if args else gpiod.line.Value.INACTIVE
                    )
                elif cmd == GPIOCommand.SET_BUZZER_VOLUME:
                    buzzer_volume = min(1.0, max(0.0, args))
                elif cmd == GPIOCommand.SET_BUZZER_TONE:
                    buzzer_tone = max(20.0, min(20000.0, args))
                elif cmd == GPIOCommand.BUZZER_BEEP:
                    duration, count = args
                    for i in range(count):
                        _buzzer_pwm_beep(buzzer_line, duration, buzzer_volume, buzzer_tone)
                        if i < count - 1:
                            time.sleep(0.1)
                elif cmd == GPIOCommand.READ_BUTTON:
                    # Check physical button
                    val = button_line.get_value(PIN_BUTTON)
                    is_pressed = val == gpiod.line.Value.INACTIVE

                    # Also check for trigger file (for remote simulation)
                    if not is_pressed and BUTTON_TRIGGER_FILE.exists():
                        try:
                            BUTTON_TRIGGER_FILE.unlink()  # Delete trigger file
                            is_pressed = True
                        except Exception:
                            pass

                    response_queue.put(is_pressed)
                elif cmd == GPIOCommand.SHUTDOWN:
                    running = False
        except Exception:
            pass

        # Calculate LED states using monotonic time
        current_time = time.monotonic()

        red_should_be = _calculate_led_state(red_pattern, current_time)
        green_should_be = _calculate_led_state(green_pattern, current_time)

        # Only write GPIO if changed
        if red_should_be != red_last_physical:
            red_led_line.set_value(
                PIN_RED_LED, gpiod.line.Value.ACTIVE if red_should_be else gpiod.line.Value.INACTIVE
            )
            red_last_physical = red_should_be

        if green_should_be != green_last_physical:
            green_led_line.set_value(
                PIN_GREEN_LED,
                gpiod.line.Value.ACTIVE if green_should_be else gpiod.line.Value.INACTIVE,
            )
            green_last_physical = green_should_be

        time.sleep(LOOP_INTERVAL)

    # Cleanup
    red_led_line.set_value(PIN_RED_LED, gpiod.line.Value.INACTIVE)
    green_led_line.set_value(PIN_GREEN_LED, gpiod.line.Value.INACTIVE)
    buzzer_line.set_value(PIN_BUZZER, gpiod.line.Value.INACTIVE)
    red_led_line.release()
    green_led_line.release()
    buzzer_line.release()
    button_line.release()
    chip.close()


class GPIOControllerProcess:
    """GPIO controller that runs in a separate process.

    This controller is completely immune to GIL contention in the main
    process. LED blinking will be smooth even during heavy video encoding.
    """

    def __init__(self):
        """Initialize the GPIO controller process."""
        self._command_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._response_queue: multiprocessing.Queue = multiprocessing.Queue()
        self._process = multiprocessing.Process(
            target=_gpio_process_main, args=(self._command_queue, self._response_queue), daemon=True
        )
        self._process.start()

    def set_led_states(self, red: str, green: str):
        """Set LED states.

        Args:
            red: Red LED pattern ('off', 'on', 'slow_blink', or 'fast_blink').
            green: Green LED pattern ('off', 'on', 'slow_blink', or 'fast_blink').
        """
        self._command_queue.put((GPIOCommand.SET_LED_STATES, (red, green)))

    def set_buzzer(self, state: bool):
        """Set buzzer state.

        Args:
            state: True for ON, False for OFF.
        """
        self._command_queue.put((GPIOCommand.SET_BUZZER, state))

    def set_buzzer_volume(self, volume: float):
        """Set buzzer volume via PWM duty cycle.

        Args:
            volume: Volume level from 0.0 (silent) to 1.0 (full).
        """
        self._command_queue.put((GPIOCommand.SET_BUZZER_VOLUME, volume))

    def set_buzzer_tone(self, freq: float):
        """Set buzzer tone frequency.

        Args:
            freq: Tone frequency in Hz (clamped to 20-20000).
        """
        self._command_queue.put((GPIOCommand.SET_BUZZER_TONE, freq))

    def buzzer_beep(self, duration: float = 0.1, count: int = 1):
        """Sound the buzzer.

        Args:
            duration: Duration of each beep in seconds.
            count: Number of beeps.
        """
        self._command_queue.put((GPIOCommand.BUZZER_BEEP, (duration, count)))

    def read_button(self) -> bool:
        """Read the state of the push button.

        Returns:
            True if button is pressed, False otherwise.
        """
        self._command_queue.put((GPIOCommand.READ_BUTTON, None))
        try:
            result: bool = self._response_queue.get(timeout=1.0)
            return result
        except Exception:
            return False

    def cleanup(self):
        """Cleanup and stop the GPIO process."""
        try:
            self._command_queue.put((GPIOCommand.SHUTDOWN, None))
            self._process.join(timeout=2.0)
        except Exception:
            pass
        if self._process.is_alive():
            self._process.terminate()
