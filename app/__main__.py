"""Entry point for the omgrab-runtime runtime."""

import logging
import os
import pathlib
import signal
import threading
import time

from app import docker_wifi
from app import stream_configs

from omgrab import cameras
from omgrab import devices
from omgrab import display
from omgrab import gpio
from omgrab import recording
from omgrab import runtime
from omgrab import utils
from omgrab import workflows
from omgrab.cameras import usb_port
from omgrab.runtime import gpio_manager
from omgrab.runtime import linux_system_metrics
from omgrab.runtime import wifi_connect

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


SPOOL_BASE_DIR = pathlib.Path('/data/spool')
SPOOL_DIR = SPOOL_BASE_DIR / 'capture'
OUTPUT_DIR = SPOOL_BASE_DIR / 'output'
AGENT_VERSION = os.getenv('AGENT_VERSION', 'unknown')
DEVICE_ID = os.getenv('DEVICE_ID', 'unknown')
WRIST_CAMERAS_ENABLED = os.getenv('WRIST_CAMERAS_ENABLED', '').lower() in ('true', '1', 'yes')
IMU_RATE_HZ = 100


def ensure_dirs():
    """Ensure the spool and output directories exist."""
    SPOOL_BASE_DIR.mkdir(parents=True, exist_ok=True)
    SPOOL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info('Directories created: spool=%s, output=%s', SPOOL_DIR, OUTPUT_DIR)


def main():
    """Main entry point for omgrab-runtime."""
    ensure_dirs()
    threads: list[threading.Thread] = []

    logger.info('Starting omgrab-runtime version=%s device=%s', AGENT_VERSION, DEVICE_ID)

    utils.merge_orphaned_chunks(SPOOL_DIR, OUTPUT_DIR)

    device_status_manager = runtime.DeviceStatusManager(
        device_id=DEVICE_ID,
        software_version=AGENT_VERSION,
        spool_base_dir=SPOOL_BASE_DIR,
        system_metrics=linux_system_metrics.LinuxSystemMetrics(),
    )

    rgb_camera_config = cameras.CameraConfig(fps=25, width=1280, height=800)
    depth_camera_config = cameras.CameraConfig(fps=25, width=640, height=400)
    preview_camera_config = cameras.CameraConfig(fps=10, width=640, height=400)
    device = devices.OakDCaptureDevice(
        rgb_camera_config,
        depth_camera_config,
        max_queue_size=200,
        imu_rate_hz=IMU_RATE_HZ,
        preview_config=preview_camera_config,
    )
    stream_config_map = {
        'rgb': stream_configs.rgb(rgb_camera_config, title='egocentric_rgb'),
        'depth': stream_configs.depth(depth_camera_config, title='egocentric_depth'),
    }

    sensor_stream_configs = {}
    sensors = []
    sensor_stream_names = []
    if device.imu_enabled:
        imu_data_config = recording.DataStreamConfig(
            codec='ass',
            metadata={'type': 'imu', 'format': 'json', 'version': '2', 'TITLE': 'imu'},
        )
        sensor_stream_configs['imu'] = imu_data_config
        sensors.append(device.get_imu_source())
        sensor_stream_names.append('imu')
        logger.info('IMU enabled at %dHz', IMU_RATE_HZ)

    target_cameras = [device.get_rgb_camera(), device.get_depth_camera()]
    stream_names = ['rgb', 'depth']
    wrist_camera_devices: list[devices.CaptureDevice] = []
    if WRIST_CAMERAS_ENABLED:
        wrist_camera_config = cameras.CameraConfig(
            fps=15,
            width=1280,
            height=720,
        )
        left_device = devices.USBCaptureDevice(
            wrist_camera_config, usb_port.LEFT_WRIST_USB_PORT, 'left_wrist')
        right_device = devices.USBCaptureDevice(
            wrist_camera_config, usb_port.RIGHT_WRIST_USB_PORT, 'right_wrist')
        wrist_camera_devices = [left_device, right_device]
        target_cameras.append(left_device.get_camera())
        target_cameras.append(right_device.get_camera())
        stream_names.extend(['left_wrist', 'right_wrist'])
        stream_config_map['left_wrist'] = stream_configs.rgb(
            wrist_camera_config, title='left_wrist')
        stream_config_map['right_wrist'] = stream_configs.rgb(
            wrist_camera_config, title='right_wrist')
        logger.info(
            'Wrist cameras enabled (left port=%s, right port=%s)',
            usb_port.LEFT_WRIST_USB_PORT,
            usb_port.RIGHT_WRIST_USB_PORT)

    recording_manager = runtime.RecordingManager(
        devices=[device, *wrist_camera_devices],
        target_cameras=target_cameras,
        stream_names=stream_names,
        stream_configs=stream_config_map,
        spool_dir=SPOOL_DIR,
        output_dir=OUTPUT_DIR,
        sensors=sensors,
        sensor_stream_names=sensor_stream_names,
        sensor_stream_configs=sensor_stream_configs,
    )

    gpio_controller = gpio.GPIOController()
    state_machine_obj = workflows.StateMachine(
        gpio_controller=gpio_controller,
        recording_manager=recording_manager)

    recording_manager.set_on_device_unhealthy(state_machine_obj.on_device_unhealthy)

    network_monitor = runtime.NetworkMonitor(runtime.NetworkMonitorConfig())
    network_monitor.register_on_change_callback(state_machine_obj.on_network_change)

    device_status_manager.set_state_machine(state_machine_obj)
    device_status_manager.set_recording_manager(recording_manager)
    device_status_manager.set_network_monitor(network_monitor)
    device_status_manager.start()

    screen_manager = display.ScreenManager()
    screen_manager.set_device_status_manager(device_status_manager)
    screen_manager.start()

    state_machine_obj.set_show_alert_callback(
        lambda msg: screen_manager.show_notification(
            msg, duration_s=5.0, priority=10, warning=True)
    )

    screen_manager.set_preview_source(device, device.get_preview_camera())
    state_machine_obj.set_preview_callbacks(
        is_available=lambda: screen_manager.display_available,
        start=screen_manager.start_preview,
        stop=screen_manager.stop_preview,
    )

    wifi_manager = wifi_connect.WifiManager(
        start_container=docker_wifi.start,
        stop_container=docker_wifi.stop,
        is_running=docker_wifi.is_running,
    )

    network_monitor.start()

    def signal_handler(signum, _frame):
        logger.info('Received signal %s, shutting down', signum)
        state_machine_obj.shutdown()
        time.sleep(1)
        state_machine_obj.cleanup()
        recording_manager.shutdown()
        network_monitor.shutdown()
        screen_manager.shutdown()
        device_status_manager.shutdown()
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    thread = threading.Thread(
        target=gpio_manager.button_monitor_loop,
        args=(gpio_controller, state_machine_obj),
        kwargs={
            'config': gpio_manager.ButtonConfig(),
            'on_button_press': screen_manager.wake,
            'on_wifi_hotspot_started': lambda: screen_manager.show_notification(
                'WiFi Hotspot\nStarted', duration_s=5.0),
            'wifi_manager': wifi_manager,
        },
        daemon=True)
    thread.start()
    threads.append(thread)

    network_monitor.wake()

    logger.info('omgrab-runtime started successfully')

    while True:
        time.sleep(60)


if __name__ == '__main__':
    main()
