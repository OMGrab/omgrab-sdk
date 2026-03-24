"""WiFi Connect manager with injectable container management functions.

The WifiManager class owns the WiFi setup lifecycle (state transitions,
callbacks) but delegates container operations to injected
functions. This allows the SDK to be independent of Docker while
applications can provide Docker-specific implementations.
"""
from typing import Optional

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class WifiManager:
    """Manages WiFi setup lifecycle with injectable container functions.

    The manager orchestrates the WiFi provisioning flow:
    1. Check if the WiFi setup service is already running.
    2. Start it via the injected function.
    3. Wait for completion (via the start function's blocking behavior).
    4. Invoke a callback when complete.

    Container-specific logic (Docker API calls, process management, etc.)
    is provided by the caller via the constructor arguments.
    """

    def __init__(
            self,
            start_container: Callable[[bool, Optional[Callable[[bool], None]]], bool],
            stop_container: Callable[[], bool],
            is_running: Callable[[], bool]):
        """Initialize the WiFi manager.

        Args:
            start_container: Function to start the WiFi setup service.
                Signature: (force_mode, callback) -> bool.
                Should return True if the service was started, False otherwise.
                The callback (if provided) should be invoked with True/False
                when the service completes.
            stop_container: Function to stop the WiFi setup service.
                Should return True if stopped, False otherwise.
            is_running: Function to check if the service is currently running.
                Should return True if running, False otherwise.
        """
        self._start_container = start_container
        self._stop_container = stop_container
        self._is_running = is_running

    def start_wifi_connect(self, force_mode: bool = True,
                           callback: Optional[Callable[[bool], None]] = None) -> bool:
        """Start the WiFi setup service.

        Args:
            force_mode: If True, forces the portal to start even if WiFi
                is connected.
            callback: Optional callback invoked when WiFi setup completes.

        Returns:
            True if the service was started, False otherwise.
        """
        if self._is_running():
            logger.debug('WiFi service already running, skipping start')
            return False

        logger.info('Starting WiFi setup service')
        return self._start_container(force_mode, callback)

    def stop_wifi_connect(self) -> bool:
        """Stop the WiFi setup service if running.

        Returns:
            True if the service was stopped, False otherwise.
        """
        if not self._is_running():
            logger.debug('WiFi service not running, nothing to stop')
            return False

        logger.info('Stopping WiFi setup service')
        return self._stop_container()

    def is_wifi_connect_running(self) -> bool:
        """Check if the WiFi setup service is currently running.

        Returns:
            True if running, False otherwise.
        """
        return self._is_running()
