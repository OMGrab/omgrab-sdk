"""Tests for the recording manager (runtime/recording_manager.py)."""
import datetime
import json
import pathlib
import threading
import time

from omgrab import testing
from omgrab.cameras import cameras
from omgrab.recording import chunked_writer
from omgrab.recording import py_av_writer
from omgrab.runtime import recording_manager

_FAST_CONFIG = recording_manager.RecordingConfig(health_check_interval_s=0.05)


class FakeCamera(cameras.Camera):
    """Minimal fake camera for testing."""

    def __init__(self):
        super().__init__(
            cameras.CameraConfig(fps=30, width=640, height=480),
            enforce_frame_timing=False,
        )
        self._queue_flushed = False

    def get_next_frame(self, timeout_s=None):
        raise cameras.FrameUnavailableError('fake')

    def setup(self):
        pass

    def close(self):
        pass

    def flush_queue(self):
        self._queue_flushed = True
        return 0


class FakeRecordingSession:
    """Minimal fake for recording_session.RecordingSession."""

    def __init__(self, recording_id: str):
        self._recording_id = recording_id
        self._started = False
        self._stopped = False
        self._alive = False

    @property
    def recording_id(self):
        return self._recording_id

    @property
    def stopped(self):
        return self._stopped

    @property
    def is_alive(self):
        return self._alive

    @property
    def recording_length_s(self):
        return None

    def start(self):
        self._started = True
        self._alive = True

    def stop(self):
        self._stopped = True
        self._alive = False

    def join(self, timeout=None):
        self._alive = False
        return True


def _make_manager(
        tmp_path: pathlib.Path,
        capture_device=None,
        extra_devices=None,
        on_device_unhealthy=None) -> recording_manager.RecordingManager:
    """Create a RecordingManager with fakes."""
    if capture_device is None:
        capture_device = testing.FakeCaptureDevice()

    devices = [capture_device] + (extra_devices or [])

    cam = FakeCamera()
    spool_dir = tmp_path / 'spool'
    spool_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)

    return recording_manager.RecordingManager(
        devices=devices,
        target_cameras=[cam],
        stream_names=['rgb'],
        stream_configs={
            'rgb': chunked_writer.VideoStreamConfig(
                width=640, height=480, fps=30,
                metadata={'type': 'rgb'},
            ),
        },
        spool_dir=spool_dir,
        output_dir=output_dir,
        config=_FAST_CONFIG,
        on_device_unhealthy=on_device_unhealthy,
    )



class TestRecordingManagerInit:

    def test_not_recording_initially(self, tmp_path: pathlib.Path):
        """Manager should not be recording after construction."""
        mgr = _make_manager(tmp_path)
        assert mgr.is_recording is False

    def test_no_active_recording_id(self, tmp_path: pathlib.Path):
        """There should be no active recording ID initially."""
        mgr = _make_manager(tmp_path)
        assert mgr.active_recording_id is None

    def test_no_recording_started_at(self, tmp_path: pathlib.Path):
        """recording_started_at should be None initially."""
        mgr = _make_manager(tmp_path)
        assert mgr.recording_started_at is None



class TestStartRecording:

    def test_returns_false_when_device_not_connected(self, tmp_path: pathlib.Path):
        """start_recording should fail if the device is not connected."""
        device = testing.FakeCaptureDevice(connected=False)
        mgr = _make_manager(tmp_path, capture_device=device)

        assert mgr.start_recording() is False

    def test_returns_false_when_device_not_ready(self, tmp_path: pathlib.Path):
        """start_recording should fail if the device doesn't become ready."""
        device = testing.FakeCaptureDevice(ready=False)
        mgr = _make_manager(tmp_path, capture_device=device)

        assert mgr.start_recording() is False

    def test_returns_false_when_extra_device_not_connected(
            self, tmp_path: pathlib.Path):
        """start_recording should fail if any device is not connected."""
        extra = testing.FakeCaptureDevice(connected=False)
        mgr = _make_manager(tmp_path, extra_devices=[extra])

        assert mgr.start_recording() is False

    def test_returns_false_when_extra_device_not_ready(
            self, tmp_path: pathlib.Path, monkeypatch):
        """start_recording should fail if any device doesn't become ready."""
        extra = testing.FakeCaptureDevice(ready=False)
        mgr = _make_manager(tmp_path, extra_devices=[extra])

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        assert mgr.start_recording() is False

    def test_returns_false_when_already_recording(
            self, tmp_path: pathlib.Path, monkeypatch):
        """start_recording should fail if a session is already active."""
        mgr = _make_manager(tmp_path)

        # Patch RecordingSession to avoid needing real video writers.
        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        assert mgr.start_recording() is True
        assert mgr.start_recording() is False

    def test_sets_active_recording_id_as_timestamp(
            self, tmp_path: pathlib.Path, monkeypatch):
        """After starting, active_recording_id should be a timestamp string."""
        mgr = _make_manager(tmp_path)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        mgr.start_recording()

        rec_id = mgr.active_recording_id
        assert rec_id is not None
        # Should be a UTC timestamp like '2026-03-17T00-47-42Z'
        assert rec_id.endswith('Z')
        assert 'T' in rec_id

    def test_sets_recording_started_at(
            self, tmp_path: pathlib.Path, monkeypatch):
        """After starting, recording_started_at should be set."""
        mgr = _make_manager(tmp_path)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        before = datetime.datetime.now(datetime.UTC)
        mgr.start_recording()
        after = datetime.datetime.now(datetime.UTC)

        assert mgr.recording_started_at is not None
        assert before <= mgr.recording_started_at <= after

    def test_opens_capture_device(
            self, tmp_path: pathlib.Path, monkeypatch):
        """start_recording should open the capture device."""
        device = testing.FakeCaptureDevice()
        mgr = _make_manager(tmp_path, capture_device=device)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        mgr.start_recording()

        assert device.opened

    def test_creates_recording_directory(
            self, tmp_path: pathlib.Path, monkeypatch):
        """start_recording should create a recording subdirectory in spool."""
        mgr = _make_manager(tmp_path)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        mgr.start_recording()

        spool_dir = tmp_path / 'spool'
        subdirs = [d for d in spool_dir.iterdir() if d.is_dir()]
        assert len(subdirs) == 1
        assert subdirs[0].name.endswith('Z')



class TestStopRecording:

    def test_stop_when_not_recording_is_noop(self, tmp_path: pathlib.Path):
        """stop_recording when no session is active should not crash."""
        mgr = _make_manager(tmp_path)
        mgr.stop_recording()  # Should not raise.

    def test_stop_clears_active_session(
            self, tmp_path: pathlib.Path, monkeypatch):
        """stop_recording should clear the active session and recording ID."""
        mgr = _make_manager(tmp_path)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        mgr.start_recording()
        mgr.stop_recording()

        assert mgr.is_recording is False
        assert mgr.active_recording_id is None
        assert mgr.recording_started_at is None

    def test_stop_closes_capture_device(
            self, tmp_path: pathlib.Path, monkeypatch):
        """stop_recording should close the capture device."""
        device = testing.FakeCaptureDevice()
        mgr = _make_manager(tmp_path, capture_device=device)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        mgr.start_recording()
        assert device.opened

        mgr.stop_recording()
        assert not device.opened

    def test_writes_manifest_on_stop(
            self, tmp_path: pathlib.Path, monkeypatch):
        """stop_recording should write a JSON manifest alongside the MKV."""
        device = testing.FakeCaptureDevice(device_type='oakd_pro_wide')
        mgr = _make_manager(tmp_path, capture_device=device)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )
        # Patch merge to create a dummy MKV instead of running ffmpeg.
        monkeypatch.setattr(
            py_av_writer, 'merge_recording_chunks',
            lambda recording_dir, output_path: output_path.write_bytes(b'mkv'),
        )

        mgr.start_recording()
        recording_id = mgr.active_recording_id
        mgr.stop_recording()

        output_dir = tmp_path / 'output'
        manifest_path = output_dir / f'{recording_id}.json'
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert data['recording_id'] == recording_id
        assert data['schema_version'] == 1
        assert data['started_at'] is not None
        assert data['stopped_at'] is not None
        assert data['duration_s'] is not None
        assert data['duration_s'] >= 0
        assert data['recovered'] is False
        assert len(data['devices']) == 1
        assert data['devices'][0]['label'] == 'fake'
        assert data['devices'][0]['device_type'] == 'oakd_pro_wide'
        assert 'rgb' in data['streams']
        assert data['output_file'] == f'{recording_id}.mkv'


class TestRecordingManagerShutdown:

    def test_shutdown_stops_active_recording(
            self, tmp_path: pathlib.Path, monkeypatch):
        """shutdown() should stop an active recording if one exists."""
        mgr = _make_manager(tmp_path)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        mgr.start_recording()
        mgr.shutdown()

        assert mgr.is_recording is False

    def test_shutdown_when_not_recording(self, tmp_path: pathlib.Path):
        """shutdown() when not recording should not raise."""
        mgr = _make_manager(tmp_path)
        mgr.shutdown()



class TestHealthMonitor:

    def test_device_unhealthy_callback_invoked(
            self, tmp_path: pathlib.Path, monkeypatch):
        """on_device_unhealthy should be called when device becomes not ready."""
        callback_called: list[bool] = []
        device = testing.FakeCaptureDevice()
        mgr = _make_manager(
            tmp_path,
            capture_device=device,
            on_device_unhealthy=lambda: callback_called.append(True),
        )

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        mgr.start_recording()
        # Simulate device becoming unhealthy.
        device._ready = False

        import time
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if callback_called:
                break
            time.sleep(0.1)

        assert callback_called
        mgr.shutdown()

    def test_extra_device_unhealthy_triggers_callback(
            self, tmp_path: pathlib.Path, monkeypatch):
        """on_device_unhealthy should fire when any device becomes not ready."""
        callback_called: list[bool] = []
        extra = testing.FakeCaptureDevice()
        mgr = _make_manager(
            tmp_path,
            extra_devices=[extra],
            on_device_unhealthy=lambda: callback_called.append(True),
        )

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        mgr.start_recording()
        extra._ready = False

        import time
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline:
            if callback_called:
                break
            time.sleep(0.1)

        assert callback_called
        mgr.shutdown()

    def test_set_on_device_unhealthy_updates_callback(self, tmp_path: pathlib.Path):
        """set_on_device_unhealthy should replace the callback."""
        mgr = _make_manager(tmp_path)
        calls: list[str] = []

        mgr.set_on_device_unhealthy(lambda: calls.append('called'))

        assert mgr._on_device_unhealthy is not None



class TestRecordingName:

    def test_make_recording_name_format(self):
        """Recording name should be a UTC timestamp ending with Z."""
        name = recording_manager._make_recording_name()
        assert name.endswith('Z')
        assert 'T' in name
        # Should be parseable as a timestamp
        datetime.datetime.strptime(name, '%Y-%m-%dT%H-%M-%SZ')



class TestErrorPaths:

    def test_merge_failure_cleans_up_state(
            self, tmp_path: pathlib.Path, monkeypatch):
        """stop_recording should clean up state even if chunk merge fails."""
        mgr = _make_manager(tmp_path)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        def _raise_merge(recording_dir, output_path):
            raise RuntimeError('merge failed')

        monkeypatch.setattr(
            py_av_writer, 'merge_recording_chunks', _raise_merge)

        mgr.start_recording()
        assert mgr.is_recording is True

        mgr.stop_recording()

        assert mgr.is_recording is False
        assert mgr.active_recording_id is None
        assert mgr.recording_started_at is None

    def test_merge_failure_skips_recording_complete_callback(
            self, tmp_path: pathlib.Path, monkeypatch):
        """on_recording_complete should not be called when merge fails."""
        callback_paths: list[pathlib.Path] = []
        mgr = _make_manager(tmp_path)
        mgr._on_recording_complete = lambda p: callback_paths.append(p)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        def _raise_merge(recording_dir, output_path):
            raise RuntimeError('merge failed')

        monkeypatch.setattr(
            py_av_writer, 'merge_recording_chunks', _raise_merge)

        mgr.start_recording()
        mgr.stop_recording()

        assert not callback_paths

    def test_merge_failure_does_not_write_manifest(
            self, tmp_path: pathlib.Path, monkeypatch):
        """No manifest should be written when merge fails."""
        mgr = _make_manager(tmp_path)

        monkeypatch.setattr(
            recording_manager.recording_session, 'RecordingSession',
            lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
        )

        def _raise_merge(recording_dir, output_path):
            raise RuntimeError('merge failed')

        monkeypatch.setattr(
            py_av_writer, 'merge_recording_chunks', _raise_merge)

        mgr.start_recording()
        recording_id = mgr.active_recording_id
        mgr.stop_recording()

        output_dir = tmp_path / 'output'
        manifest_path = output_dir / f'{recording_id}.json'
        assert not manifest_path.exists()

    def test_health_callback_exception_does_not_crash(
            self, tmp_path: pathlib.Path, monkeypatch):
        """Health monitor should catch callback exceptions gracefully."""
        thread_exceptions: list[Exception] = []
        original_excepthook = threading.excepthook

        def capture_exception(args):
            thread_exceptions.append(args.exc_value)

        threading.excepthook = capture_exception
        try:
            callback_invoked = threading.Event()
            device = testing.FakeCaptureDevice()

            def failing_callback():
                callback_invoked.set()
                raise RuntimeError('callback error')

            mgr = _make_manager(
                tmp_path,
                capture_device=device,
                on_device_unhealthy=failing_callback,
            )

            monkeypatch.setattr(
                recording_manager.recording_session, 'RecordingSession',
                lambda **kwargs: FakeRecordingSession(kwargs['recording_id']),
            )

            mgr.start_recording()
            device._ready = False

            assert callback_invoked.wait(timeout=3.0), (
                'Callback was never invoked')
            time.sleep(0.2)

            assert not thread_exceptions, (
                f'Unhandled thread exception: {thread_exceptions}')
            mgr.stop_recording()
        finally:
            threading.excepthook = original_excepthook
