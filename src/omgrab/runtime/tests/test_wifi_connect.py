"""Tests for the WifiManager class (runtime/wifi_connect.py)."""

from omgrab.runtime import wifi_connect


class TestWifiManagerInit:

    def test_can_construct_with_lambdas(self):
        """WifiManager should accept callable functions."""
        mgr = wifi_connect.WifiManager(
            start_container=lambda force, cb: True,
            stop_container=lambda: True,
            is_running=lambda: False,
        )
        assert mgr is not None



class TestStartWifiConnect:

    def test_returns_false_when_already_running(self):
        """Should return False if service is already running."""
        mgr = wifi_connect.WifiManager(
            start_container=lambda force, cb: True,
            stop_container=lambda: True,
            is_running=lambda: True,
        )

        result = mgr.start_wifi_connect()

        assert result is False

    def test_returns_true_when_started(self):
        """Should return True when service starts successfully."""
        mgr = wifi_connect.WifiManager(
            start_container=lambda force, cb: True,
            stop_container=lambda: True,
            is_running=lambda: False,
        )

        result = mgr.start_wifi_connect()

        assert result is True

    def test_passes_force_mode_to_start_fn(self):
        """force_mode should be passed through to start_container_fn."""
        captured: dict = {}

        def fake_start(force, cb):
            captured['force'] = force
            captured['callback'] = cb
            return True

        mgr = wifi_connect.WifiManager(
            start_container=fake_start,
            stop_container=lambda: True,
            is_running=lambda: False,
        )

        callback = lambda success: None  # noqa: E731
        mgr.start_wifi_connect(force_mode=False, callback=callback)

        assert captured['force'] is False
        assert captured['callback'] is callback

    def test_returns_false_when_start_fn_fails(self):
        """Should return False when start function returns False."""
        mgr = wifi_connect.WifiManager(
            start_container=lambda force, cb: False,
            stop_container=lambda: True,
            is_running=lambda: False,
        )

        result = mgr.start_wifi_connect()

        assert result is False



class TestStopWifiConnect:

    def test_stops_running_service(self):
        """Should return True when service is stopped successfully."""
        mgr = wifi_connect.WifiManager(
            start_container=lambda force, cb: True,
            stop_container=lambda: True,
            is_running=lambda: True,
        )

        result = mgr.stop_wifi_connect()

        assert result is True

    def test_returns_false_when_not_running(self):
        """Should return False when service is not running."""
        mgr = wifi_connect.WifiManager(
            start_container=lambda force, cb: True,
            stop_container=lambda: True,
            is_running=lambda: False,
        )

        result = mgr.stop_wifi_connect()

        assert result is False

    def test_calls_stop_fn(self):
        """Should call stop_container_fn."""
        stop_calls: list[bool] = []

        mgr = wifi_connect.WifiManager(
            start_container=lambda force, cb: True,
            stop_container=lambda: stop_calls.append(True) or True,
            is_running=lambda: True,
        )

        mgr.stop_wifi_connect()

        assert len(stop_calls) == 1



class TestIsWifiConnectRunning:

    def test_delegates_to_is_running_fn(self):
        """Should return the result of is_running_fn."""
        mgr = wifi_connect.WifiManager(
            start_container=lambda force, cb: True,
            stop_container=lambda: True,
            is_running=lambda: True,
        )

        assert mgr.is_wifi_connect_running() is True

    def test_returns_false_when_not_running(self):
        """Should return False when is_running_fn returns False."""
        mgr = wifi_connect.WifiManager(
            start_container=lambda force, cb: True,
            stop_container=lambda: True,
            is_running=lambda: False,
        )

        assert mgr.is_wifi_connect_running() is False
