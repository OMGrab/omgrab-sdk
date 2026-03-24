"""System metrics providers for reading hardware and OS information."""
from typing import Optional

import logging
import os
import pathlib
import subprocess

from omgrab.runtime import device_status

logger = logging.getLogger(__name__)


class LinuxSystemMetrics:
    """System metrics provider that reads from Linux /proc and /sys filesystems.

    All methods gracefully degrade on non-Linux hosts, returning zero or
    None values rather than raising exceptions.
    """

    def get_cpu_info(self) -> device_status.CPUInfo:
        """Get CPU temperature and usage information."""
        temp = self._get_cpu_temperature()
        usage, per_core = self._get_cpu_usage()
        return device_status.CPUInfo(
            temperature_celsius=temp,
            usage_percent=usage,
            usage_per_core=per_core,
        )

    def get_memory_info(self) -> device_status.MemoryInfo:
        """Get memory usage information."""
        try:
            with open('/proc/meminfo') as f:
                lines = f.readlines()

            mem_info = {}
            for line in lines:
                parts = line.split(':')
                if len(parts) == 2:
                    key = parts[0].strip()
                    # Value is in kB
                    value = int(parts[1].strip().split()[0])
                    mem_info[key] = value * 1024  # Convert to bytes

            total_bytes = mem_info.get('MemTotal', 0)
            available_bytes = mem_info.get('MemAvailable', 0)
            used_bytes = total_bytes - available_bytes
            used_percent = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0.0

            return device_status.MemoryInfo(
                total_bytes=total_bytes,
                used_bytes=used_bytes,
                available_bytes=available_bytes,
                used_percent=round(used_percent, 2),
            )
        except Exception as e:
            logger.warning('Failed to get memory info: %s', e)
            return device_status.MemoryInfo(
                total_bytes=0,
                used_bytes=0,
                available_bytes=0,
                used_percent=0.0,
            )

    def get_storage_info(self, path: pathlib.Path) -> device_status.StorageInfo:
        """Get storage space information for the given path."""
        try:
            vfs = os.statvfs(path)

            total_bytes = vfs.f_blocks * vfs.f_frsize
            available_bytes = vfs.f_bavail * vfs.f_frsize
            used_bytes = total_bytes - available_bytes
            used_percent = (used_bytes / total_bytes * 100) if total_bytes > 0 else 0.0

            return device_status.StorageInfo(
                total_bytes=total_bytes,
                used_bytes=used_bytes,
                available_bytes=available_bytes,
                used_percent=round(used_percent, 2),
            )
        except Exception as e:
            logger.warning('Failed to get storage info: %s', e)
            return device_status.StorageInfo(
                total_bytes=0,
                used_bytes=0,
                available_bytes=0,
                used_percent=0.0,
            )

    def get_wifi_info(self) -> tuple[Optional[str], Optional[int]]:
        """Get WiFi SSID and signal strength.

        Returns:
            Tuple of (ssid, signal_strength_dbm) or (None, None) if unavailable.
        """
        try:
            result = subprocess.run(
                ['iwgetid', '-r'],
                capture_output=True,
                text=True,
                timeout=2
            )
            ssid = result.stdout.strip() if result.returncode == 0 else None

            signal_dbm = None
            result = subprocess.run(
                ['iwconfig'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    if 'Signal level' in line:
                        parts = line.split('Signal level=')
                        if len(parts) > 1:
                            signal_str = parts[1].split()[0]
                            signal_dbm = int(
                                ''.join(c for c in signal_str
                                        if c.isdigit() or c == '-'))

            return ssid, signal_dbm

        except Exception as e:
            logger.debug('Failed to get WiFi info: %s', e)
            return None, None

    def _get_cpu_temperature(self) -> Optional[float]:
        """Get CPU temperature in Celsius."""
        try:
            thermal_file = pathlib.Path('/sys/class/thermal/thermal_zone0/temp')
            if thermal_file.exists():
                temp_millidegrees = int(thermal_file.read_text().strip())
                return round(temp_millidegrees / 1000.0, 1)
        except Exception as e:
            logger.debug('Failed to read CPU temperature: %s', e)
        return None

    def _get_cpu_usage(self) -> tuple[float, list[float]]:
        """Get CPU usage percentages.

        Returns:
            Tuple of (overall_usage_percent, per_core_usage_list).
        """
        try:
            with open('/proc/stat') as f:
                lines = f.readlines()

            overall = 0.0
            per_core = []

            for line in lines:
                if line.startswith('cpu '):
                    fields = line.split()
                    user = int(fields[1])
                    nice = int(fields[2])
                    system = int(fields[3])
                    idle = int(fields[4])
                    iowait = int(fields[5]) if len(fields) > 5 else 0

                    total = user + nice + system + idle + iowait
                    active = user + nice + system

                    overall = (active / total * 100) if total > 0 else 0.0

                elif line.startswith('cpu'):
                    fields = line.split()
                    user = int(fields[1])
                    nice = int(fields[2])
                    system = int(fields[3])
                    idle = int(fields[4])
                    iowait = int(fields[5]) if len(fields) > 5 else 0

                    total = user + nice + system + idle + iowait
                    active = user + nice + system

                    core_usage = (active / total * 100) if total > 0 else 0.0
                    per_core.append(round(core_usage, 1))

            return round(overall, 1), per_core

        except Exception as e:
            logger.debug('Failed to read CPU usage: %s', e)
            return 0.0, []
