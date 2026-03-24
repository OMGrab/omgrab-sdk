"""Interactive test script that cycles through every display configuration.

Run directly on the Pi to visually verify each screen state on the OLED:

    python -m omgrab.display.test_display

Press Enter to advance to the next screen, or Ctrl-C to quit.
"""

from omgrab.display import screen_manager as screen_manager_module
from omgrab.display import screen_writer as screen_writer_module
from omgrab.runtime import device_status as ds

_STORAGE = ds.StorageInfo(
    total_bytes=32_000_000_000,
    used_bytes=8_000_000_000,
    available_bytes=24_000_000_000,
    used_percent=25.0,
)
_CPU = ds.CPUInfo(
    temperature_celsius=52.3, usage_percent=12.0, usage_per_core=[10.0, 14.0, 11.0, 13.0]
)
_MEMORY = ds.MemoryInfo(
    total_bytes=4_000_000_000,
    used_bytes=1_500_000_000,
    available_bytes=2_500_000_000,
    used_percent=37.5,
)


def _make_status(
    wifi_ssid=None,
    wifi_signal=None,
    battery_percent=None,
    battery_current=None,
    is_recording=False,
    duration_seconds=None,
    state='idle',
    network_status=None,
) -> ds.DeviceStatus:
    """Build a DeviceStatus with the given overrides."""
    if network_status is None:
        network_status = 'online' if wifi_ssid else 'offline'
    network = ds.NetworkInfo(
        status=network_status,
        wifi_ssid=wifi_ssid,
        wifi_signal_strength=wifi_signal,
    )
    recording = ds.RecordingInfo(
        is_recording=is_recording,
        recording_id='test-rec-id' if is_recording else None,
        duration_seconds=duration_seconds,
    )
    battery = None
    if battery_percent is not None:
        battery = ds.BatteryInfo(
            percent=battery_percent,
            voltage_v=7.4,
            current_a=battery_current if battery_current is not None else -0.5,
            power_w=3.7,
            is_charging=battery_current is not None and battery_current > 0,
        )
    return ds.DeviceStatus(
        device_id='test-device',
        software_version='1.0.0',
        uptime_seconds=3600.0,
        state_machine_state=state,
        storage=_STORAGE,
        cpu=_CPU,
        memory=_MEMORY,
        network=network,
        recording=recording,
        device_healthy=True,
        device_error=None,
        battery=battery,
    )


SCENARIOS: list[tuple[str, dict]] = [
    # Idle states
    (
        'Idle, WiFi strong, battery 80%',
        dict(
            wifi_ssid='HomeWiFi',
            wifi_signal=-45,
            battery_percent=80.0,
        ),
    ),
    (
        'Idle, WiFi medium, battery 50%',
        dict(
            wifi_ssid='HomeWiFi',
            wifi_signal=-62,
            battery_percent=50.0,
        ),
    ),
    (
        'Idle, WiFi weak, battery 20%',
        dict(
            wifi_ssid='CoffeeShop',
            wifi_signal=-75,
            battery_percent=20.0,
        ),
    ),
    (
        'Idle, WiFi very weak, battery 5%',
        dict(
            wifi_ssid='FarAway',
            wifi_signal=-85,
            battery_percent=5.0,
        ),
    ),
    (
        'Idle, no WiFi, battery 60%',
        dict(
            battery_percent=60.0,
        ),
    ),
    (
        'Idle, WiFi strong, no battery info',
        dict(
            wifi_ssid='HomeWiFi',
            wifi_signal=-50,
        ),
    ),
    (
        'Idle, WiFi strong, battery charging 30%',
        dict(
            wifi_ssid='HomeWiFi',
            wifi_signal=-50,
            battery_percent=30.0,
            battery_current=1.2,
        ),
    ),
    (
        'Idle, WiFi strong, battery charging 75%',
        dict(
            wifi_ssid='HomeWiFi',
            wifi_signal=-50,
            battery_percent=75.0,
            battery_current=0.5,
        ),
    ),
    ('Idle, no WiFi, no battery (bare minimum)', dict()),
    (
        'Idle, WiFi connected but no internet, battery 60%',
        dict(
            wifi_ssid='HomeWiFi',
            wifi_signal=-50,
            battery_percent=60.0,
            network_status='network_only',
        ),
    ),
    # Recording states
    (
        'Recording, 0m05s, WiFi strong, battery 75%',
        dict(
            wifi_ssid='HomeWiFi',
            wifi_signal=-48,
            battery_percent=75.0,
            is_recording=True,
            duration_seconds=5.0,
            state='recording',
        ),
    ),
    (
        'Recording, 2m30s, WiFi medium, battery 40%',
        dict(
            wifi_ssid='Studio',
            wifi_signal=-60,
            battery_percent=40.0,
            is_recording=True,
            duration_seconds=150.0,
            state='recording',
        ),
    ),
    (
        'Recording, 59m59s, no WiFi, battery 10%',
        dict(
            battery_percent=10.0,
            is_recording=True,
            duration_seconds=3599.0,
            state='recording',
        ),
    ),
    (
        'Recording, 1h23m45s (hours format), WiFi strong, battery 65%',
        dict(
            wifi_ssid='HomeWiFi',
            wifi_signal=-50,
            battery_percent=65.0,
            is_recording=True,
            duration_seconds=5025.0,
            state='recording',
        ),
    ),
]

# (description, message, warning)
NOTIFICATION_SCENARIOS: list[tuple[str, str, bool]] = [
    ('Info: WiFi hotspot', 'WiFi Hotspot\nStarted', False),
    ('Warning: low battery 10%', 'Low Battery\n10%', True),
    ('Warning: low battery 5%', 'Low Battery\n5%', True),
    ('Warning: storage almost full', 'Storage\nAlmost Full', True),
    ('Warning: check camera', 'Check Camera\nConnection', True),
]


def _show_on_hardware(
    writer: screen_writer_module.ScreenWriter,
    manager: screen_manager_module.ScreenManager,
    status: ds.DeviceStatus,
):
    """Render a status and push it to the hardware display."""
    manager._get_device_status = lambda: status  # type: ignore[assignment]
    image = manager._render_status_screen()
    writer.display(image)


def main():
    """Run through every display scenario on the hardware OLED."""
    print('Initializing screen writer...')
    writer = screen_writer_module.ScreenWriter()
    if not writer.available:
        print('ERROR: Screen hardware not available. Is the SSD1306 connected?')
        return

    manager = screen_manager_module.ScreenManager(writer=writer)

    print(f'\n{len(SCENARIOS)} status screens + {len(NOTIFICATION_SCENARIOS)} notifications')
    print('Press Enter to advance, Ctrl-C to quit.\n')

    try:
        for i, (description, kwargs) in enumerate(SCENARIOS, 1):
            status = _make_status(**kwargs)
            _show_on_hardware(writer, manager, status)
            input(f'  [{i}/{len(SCENARIOS)}] {description}')

        print('\n-- Notifications --\n')

        for i, (description, message, warning) in enumerate(NOTIFICATION_SCENARIOS, 1):
            notification = screen_manager_module.Notification(
                message=message, duration_s=999.0, warning=warning
            )
            image = manager._render_notification(notification)
            writer.display(image)
            input(f'  [{i}/{len(NOTIFICATION_SCENARIOS)}] {description}')

        print('\nDone! Clearing screen.')
        writer.clear()

    except KeyboardInterrupt:
        print('\nInterrupted. Clearing screen.')
        writer.clear()
    finally:
        writer.cleanup()


if __name__ == '__main__':
    main()
