"""USB port-to-video-device discovery via sysfs."""
from typing import Optional

import logging
import pathlib

logger = logging.getLogger(__name__)

# Physically fixed USB port paths on the Pi 5 board.
LEFT_WRIST_USB_PORT = '3-2'
RIGHT_WRIST_USB_PORT = '1-2'

_SYSFS_V4L_DIR = pathlib.Path('/sys/class/video4linux')


def find_video_device_by_usb_port(port_path: str) -> Optional[str]:
    """Find the /dev/videoN device corresponding to a USB port path.

    Scans sysfs to find a V4L2 device whose parent USB device matches the
    given port path (e.g. '3-2'). Only returns the primary capture device
    (index 0) since a single USB camera may register multiple video devices.

    Args:
        port_path: USB port path component (e.g. '3-2', '1-2').

    Returns:
        Device path (e.g. '/dev/video2') or None if not found.
    """
    if not _SYSFS_V4L_DIR.exists():
        logger.warning('sysfs V4L directory not found: %s', _SYSFS_V4L_DIR)
        return None

    needle = f'/{port_path}/'

    for entry in sorted(_SYSFS_V4L_DIR.iterdir()):
        if not entry.name.startswith('video'):
            continue

        # Filter to primary capture device (index 0).
        index_file = entry / 'index'
        try:
            index_val = index_file.read_text().strip()
            if index_val != '0':
                continue
        except OSError:
            continue

        # Resolve the 'device' symlink to get the full sysfs path of the
        # parent USB device, then check if the port path appears in it.
        device_link = entry / 'device'
        try:
            resolved = str(device_link.resolve())
        except OSError:
            continue

        if needle in resolved:
            dev_path = f'/dev/{entry.name}'
            logger.debug(
                'Matched USB port %s -> %s (sysfs: %s)',
                port_path, dev_path, resolved)
            return dev_path

    return None
