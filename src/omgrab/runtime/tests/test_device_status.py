"""Tests for the device status manager (runtime/device_status.py)."""
from typing import Optional

import json
import pathlib
import time

from omgrab.runtime import device_status


class FakeSystemMetrics:
    """Deterministic system metrics for testing."""

    def get_cpu_info(self) -> device_status.CPUInfo:
        """Return fixed CPU info."""
        return device_status.CPUInfo(
            temperature_celsius=45.0, usage_percent=10.0, usage_per_core=[10.0])

    def get_memory_info(self) -> device_status.MemoryInfo:
        """Return fixed memory info."""
        return device_status.MemoryInfo(
            total_bytes=4 * 2**30, used_bytes=2 * 2**30,
            available_bytes=2 * 2**30, used_percent=50.0)

    def get_storage_info(self, path: pathlib.Path) -> device_status.StorageInfo:
        """Return fixed storage info."""
        return device_status.StorageInfo(
            total_bytes=100 * 2**30, used_bytes=40 * 2**30,
            available_bytes=60 * 2**30, used_percent=40.0)

    def get_wifi_info(self) -> tuple[Optional[str], Optional[int]]:
        """Return no WiFi info."""
        return None, None


class TestDataclasses:

    def test_storage_info(self):
        """StorageInfo should store all fields."""
        info = device_status.StorageInfo(
            total_bytes=1000, used_bytes=400, available_bytes=600, used_percent=40.0,
        )
        assert info.used_percent == 40.0

    def test_cpu_info_str(self):
        """CPUInfo.__str__ should include temperature and per-core values."""
        info = device_status.CPUInfo(
            temperature_celsius=55.0, usage_percent=30.0, usage_per_core=[25.0, 35.0],
        )
        text = str(info)
        assert '55.0' in text

    def test_memory_info_str(self):
        """MemoryInfo.__str__ should include percentage."""
        info = device_status.MemoryInfo(
            total_bytes=4 * 2**30, used_bytes=2 * 2**30,
            available_bytes=2 * 2**30, used_percent=50.0,
        )
        text = str(info)
        assert '50.0%' in text

    def test_battery_info_charging(self):
        """BatteryInfo string should indicate charging when current > 0."""
        info = device_status.BatteryInfo(
            percent=80.0, voltage_v=8.0, current_a=0.5, power_w=4.0, is_charging=True,
        )
        assert 'charging' in str(info).lower()

    def test_battery_info_discharging(self):
        """BatteryInfo string should indicate discharging when current <= 0."""
        info = device_status.BatteryInfo(
            percent=50.0, voltage_v=7.0, current_a=-0.3, power_w=2.1, is_charging=False,
        )
        assert 'discharging' in str(info).lower()

    def test_device_status_holds_all_fields(self):
        """DeviceStatus should accept all required fields."""
        status = device_status.DeviceStatus(
            device_id='dev-1',
            software_version='1.0.0',
            uptime_seconds=123.4,
            state_machine_state='idle',
            storage=device_status.StorageInfo(0, 0, 0, 0.0),
            cpu=device_status.CPUInfo(None, 0.0, []),
            memory=device_status.MemoryInfo(0, 0, 0, 0.0),
            network=device_status.NetworkInfo('online', None, None),
            recording=device_status.RecordingInfo(False, None, None),
            device_healthy=True,
            device_error=None,
            battery=None,
        )
        assert status.device_id == 'dev-1'



class TestDeviceStatusManagerInit:

    def test_creates_without_battery(self, tmp_path: pathlib.Path, monkeypatch):
        """Manager should work with battery monitoring disabled."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'status.json',
        )

        assert mgr._battery_monitor is None

    def test_setters_store_components(self, tmp_path: pathlib.Path):
        """Component setters should store references."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'status.json',
        )

        class FakeSM:
            def get_current_state(self):
                return 'idle'

        sm = FakeSM()
        mgr.set_state_machine(sm)
        assert mgr._state_machine is sm



class TestGetStatus:

    def _make_manager(self, tmp_path: pathlib.Path) -> device_status.DeviceStatusManager:
        return device_status.DeviceStatusManager(
            device_id='dev-test',
            software_version='2.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'status.json',
        )

    def test_returns_device_status(self, tmp_path: pathlib.Path):
        """get_status should return a DeviceStatus instance."""
        mgr = self._make_manager(tmp_path)
        status = mgr.get_status()

        assert isinstance(status, device_status.DeviceStatus)

    def test_device_id_and_version(self, tmp_path: pathlib.Path):
        """Status should contain the device_id and software_version."""
        mgr = self._make_manager(tmp_path)
        status = mgr.get_status()

        assert status.device_id == 'dev-test'
        assert status.software_version == '2.0.0'

    def test_uptime_increases(self, tmp_path: pathlib.Path):
        """uptime_seconds should be positive and increase between calls."""
        mgr = self._make_manager(tmp_path)

        s1 = mgr.get_status()
        time.sleep(0.05)
        s2 = mgr.get_status()

        assert s1.uptime_seconds >= 0
        assert s2.uptime_seconds > s1.uptime_seconds

    def test_state_unknown_without_state_machine(self, tmp_path: pathlib.Path):
        """state_machine_state should be 'unknown' if no SM is set."""
        mgr = self._make_manager(tmp_path)
        status = mgr.get_status()

        assert status.state_machine_state == 'unknown'

    def test_state_from_state_machine(self, tmp_path: pathlib.Path):
        """state_machine_state should come from the SM when set."""
        mgr = self._make_manager(tmp_path)

        class FakeSM:
            def get_current_state(self):
                return 'recording'

        mgr.set_state_machine(FakeSM())
        status = mgr.get_status()

        assert status.state_machine_state == 'recording'

    def test_recording_info_without_manager(self, tmp_path: pathlib.Path):
        """Without a recording manager, is_recording should be False."""
        mgr = self._make_manager(tmp_path)
        status = mgr.get_status()

        assert status.recording.is_recording is False
        assert status.recording.recording_id is None

    def test_device_healthy_by_default(self, tmp_path: pathlib.Path):
        """Device should be reported healthy by default."""
        mgr = self._make_manager(tmp_path)
        status = mgr.get_status()

        assert status.device_healthy is True
        assert status.device_error is None

    def test_battery_none_without_monitor(self, tmp_path: pathlib.Path):
        """Battery should be None when monitor is disabled."""
        mgr = self._make_manager(tmp_path)
        status = mgr.get_status()

        assert status.battery is None

    def test_network_info_unknown_without_monitor(self, tmp_path: pathlib.Path):
        """Network status should be 'unknown' without a network monitor."""
        mgr = self._make_manager(tmp_path)
        status = mgr.get_status()

        assert status.network.status == 'unknown'



class TestStatusFile:

    def test_write_status_file_creates_json(self, tmp_path: pathlib.Path):
        """_write_status_file should create a valid JSON file."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'status.json',
        )

        mgr._write_status_file({'device_id': 'dev-1', 'uptime': 10.0})

        status_file = tmp_path / 'status.json'
        assert status_file.exists()

        data = json.loads(status_file.read_text())
        assert data['device_id'] == 'dev-1'
        assert '_updated_at' in data

    def test_get_status_dict_returns_cached_copy(self, tmp_path: pathlib.Path):
        """get_status_dict should return a copy of the cached dict."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'status.json',
        )

        # Prime the cache.
        mgr._update_status()

        d1 = mgr.get_status_dict()
        d2 = mgr.get_status_dict()

        # Should be equal but not the same object.
        assert d1 == d2
        assert d1 is not d2



class TestDeviceStatusWorker:

    def test_start_and_shutdown(self, tmp_path: pathlib.Path):
        """Worker thread should start and stop cleanly."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'status.json',
        )

        mgr.start()
        assert mgr._worker_thread is not None
        assert mgr._worker_thread.is_alive()

        mgr.shutdown()
        assert mgr._worker_thread is None

    def test_start_is_idempotent(self, tmp_path: pathlib.Path):
        """Calling start() twice should not create a second thread."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'status.json',
        )

        mgr.start()
        first_thread = mgr._worker_thread
        mgr.start()
        second_thread = mgr._worker_thread

        assert first_thread is second_thread
        mgr.shutdown()

    def test_worker_writes_status_file(self, tmp_path: pathlib.Path):
        """Worker should write the status JSON file periodically."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'status.json',
        )
        mgr._update_interval_s = 0.1

        mgr.start()
        time.sleep(0.3)
        mgr.shutdown()

        status_file = tmp_path / 'status.json'
        assert status_file.exists()

    def test_worker_exception_does_not_crash(
            self, tmp_path: pathlib.Path, monkeypatch):
        """Worker should continue running if _update_status raises."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'status.json',
        )
        mgr._update_interval_s = 0.05

        call_count = [0]
        original_update = mgr._update_status

        def _failing_update():
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError('update failed')
            original_update()

        monkeypatch.setattr(mgr, '_update_status', _failing_update)
        mgr.start()
        time.sleep(0.5)
        mgr.shutdown()

        # Should have recovered after the initial failures.
        assert call_count[0] >= 3



class TestDeviceStatusSetters:

    def test_set_recording_manager(self, tmp_path: pathlib.Path):
        """set_recording_manager should store the reference."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
        )

        class FakeRM:
            is_recording = False
            active_recording_id = None
            recording_started_at = None

        rm = FakeRM()
        mgr.set_recording_manager(rm)  # type: ignore[arg-type]
        assert mgr._recording_manager is rm

    def test_set_network_monitor(self, tmp_path: pathlib.Path):
        """set_network_monitor should store the reference."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
        )

        class FakeNM:
            @property
            def snapshot(self):
                from omgrab.runtime import network_monitor
                return network_monitor.Snapshot(
                    status=network_monitor.Status.ONLINE,
                    detail='ok',
                    changed_at=0.0,
                )

        nm = FakeNM()
        mgr.set_network_monitor(nm)  # type: ignore[arg-type]
        assert mgr._network_monitor is nm



class TestDeviceStatusRecordingInfo:

    def test_recording_info_with_active_recording(self, tmp_path: pathlib.Path):
        """Recording info should include duration when actively recording."""
        import datetime

        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
        )

        class FakeRM:
            is_recording = True
            active_recording_id = 'rec-1'
            recording_started_at = (
                datetime.datetime.now(datetime.UTC)
                - datetime.timedelta(seconds=10))

        mgr.set_recording_manager(FakeRM())  # type: ignore[arg-type]
        status = mgr.get_status()

        assert status.recording.is_recording is True
        assert status.recording.recording_id == 'rec-1'
        assert status.recording.duration_seconds is not None
        assert status.recording.duration_seconds >= 9.0



class TestDeviceStatusNetworkInfo:

    def test_network_info_with_monitor(self, tmp_path: pathlib.Path):
        """Network info should report status from the network monitor."""
        from omgrab.runtime import network_monitor

        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
        )

        class FakeNM:
            @property
            def snapshot(self):
                return network_monitor.Snapshot(
                    status=network_monitor.Status.ONLINE,
                    detail='ok',
                    changed_at=0.0,
                )

        mgr.set_network_monitor(FakeNM())  # type: ignore[arg-type]
        status = mgr.get_status()

        assert status.network.status == 'online'



class TestDeviceStatusBattery:

    def test_battery_init_exception_handled(
            self, tmp_path: pathlib.Path, monkeypatch):
        """Battery monitor init failure should be caught, not crash."""
        monkeypatch.setattr(
            device_status.battery_monitor, 'BatteryMonitor',
            lambda: (_ for _ in ()).throw(RuntimeError('i2c fail')),
        )

        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=True,
        )

        assert mgr._battery_monitor is None

    def test_battery_info_exception_returns_none(self, tmp_path: pathlib.Path):
        """Battery info should return None if get_status raises."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
        )

        class FailingBatteryMonitor:
            available = True
            def get_status(self):
                raise RuntimeError('i2c read failed')

        mgr._battery_monitor = FailingBatteryMonitor()  # type: ignore[assignment]
        status = mgr.get_status()

        assert status.battery is None

    def test_battery_info_returns_none_when_status_none(
            self, tmp_path: pathlib.Path):
        """Battery info should return None if get_status returns None."""
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
        )

        class NoBatteryMonitor:
            available = True
            def get_status(self):
                return None

        mgr._battery_monitor = NoBatteryMonitor()  # type: ignore[assignment]
        status = mgr.get_status()

        assert status.battery is None



class TestStatusFileErrors:

    def test_write_failure_handled(self, tmp_path: pathlib.Path):
        """Status file write failure should be caught, not crash."""
        # Point status_file at a non-existent directory.
        mgr = device_status.DeviceStatusManager(
            device_id='dev-1',
            software_version='1.0.0',
            spool_base_dir=tmp_path,
            system_metrics=FakeSystemMetrics(),
            enable_battery_monitor=False,
            status_file=tmp_path / 'nonexistent' / 'deep' / 'status.json',
        )

        # Should not raise.
        mgr._write_status_file({'test': True})
