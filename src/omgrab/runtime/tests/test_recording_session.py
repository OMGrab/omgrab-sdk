"""Tests for the recording session (runtime/recording_session.py)."""
import datetime
import fractions
import pathlib
import shutil
import threading
import time

import numpy as np
import pytest

from omgrab.cameras import cameras
from omgrab.recording import chunked_writer
from omgrab.runtime import recording_session
from omgrab.sensors import sensor

_has_ffmpeg = shutil.which('ffmpeg') is not None


@pytest.fixture(autouse=True)
def _fast_drain(monkeypatch):
    """Reduce writer drain quiet period so stop/join tests finish quickly."""
    monkeypatch.setattr(recording_session, 'WRITER_DRAIN_QUIET_PERIOD_S', 0.05)


class FakeCamera(cameras.Camera):
    """Camera that yields a finite set of frames then blocks."""

    def __init__(self, frames=None):
        super().__init__(
            cameras.CameraConfig(fps=30, width=4, height=4),
            enforce_frame_timing=False,
        )
        self._frames = frames or []
        self._index = 0

    def get_next_frame(self, timeout_s=None):
        if self._index < len(self._frames):
            frame, ts = self._frames[self._index]
            self._index += 1
            return frame, ts
        raise cameras.FrameUnavailableError('exhausted')

    def setup(self):
        pass

    def close(self):
        pass


class CrashingCamera(cameras.Camera):
    """Camera that raises an unexpected exception after a few frames."""

    def __init__(self, crash_after: int = 3):
        super().__init__(
            cameras.CameraConfig(fps=30, width=4, height=4),
            enforce_frame_timing=False,
        )
        self._crash_after = crash_after
        self._count = 0
        self._start = datetime.datetime.now(datetime.UTC)

    def get_next_frame(self, timeout_s=None):
        self._count += 1
        if self._count > self._crash_after:
            raise RuntimeError('simulated device crash')
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        ts = self._start + datetime.timedelta(milliseconds=self._count * 33)
        return frame, ts

    def setup(self):
        self._count = 0
        self._start = datetime.datetime.now(datetime.UTC)

    def close(self):
        pass


class SetupFailCamera(cameras.Camera):
    """Camera whose setup() raises, simulating a device init failure."""

    def __init__(self):
        super().__init__(
            cameras.CameraConfig(fps=30, width=4, height=4),
            enforce_frame_timing=False,
        )

    def get_next_frame(self, timeout_s=None):
        raise cameras.FrameUnavailableError('never reached')

    def setup(self):
        raise RuntimeError('simulated setup failure')

    def close(self):
        pass


def _make_frame(value: int = 0) -> np.ndarray:
    """Create a small test frame."""
    return np.zeros((4, 4, 3), dtype=np.uint8) + value


def _stream_configs():
    """Standard stream configs for tests using real encoding."""
    return {
        'rgb': chunked_writer.VideoStreamConfig(
            width=4, height=4, fps=fractions.Fraction(30),
            codec='libx264',
            stream_options={'preset': 'ultrafast'},
            metadata={'type': 'rgb'},
        ),
    }


class TestRecordingSessionInit:

    def test_camera_stream_count_mismatch_raises(self, tmp_path: pathlib.Path):
        """Constructor should raise if camera count != stream name count."""
        cam = FakeCamera()

        with pytest.raises(ValueError, match='Mismatch'):
            recording_session.RecordingSession(
                recording_id='rec-1',
                target_cameras=[cam],
                stream_names=['rgb', 'depth'],
                stream_configs={},
                spool_dir=tmp_path,
                start_chunk_callback=lambda *a: 'clip-1',
            )

    def test_data_source_stream_count_mismatch_raises(self, tmp_path: pathlib.Path):
        """Constructor should raise if data source count != data stream name count."""
        cam = FakeCamera()

        class FakeDataSource(sensor.Sensor):
            def get_next_item(self, timeout_s=None):
                raise sensor.SensorDataUnavailableError('fake')
            def setup(self):
                pass
            def close(self):
                pass

        with pytest.raises(ValueError, match='Mismatch'):
            recording_session.RecordingSession(
                recording_id='rec-1',
                target_cameras=[cam],
                stream_names=['rgb'],
                stream_configs={},
                spool_dir=tmp_path,
                start_chunk_callback=lambda *a: 'clip-1',
                sensors=[FakeDataSource()],
                sensor_stream_names=['imu', 'gps'],
            )

    def test_initial_state(self, tmp_path: pathlib.Path):
        """Session should start in a non-started, non-stopped state."""
        cam = FakeCamera()
        session = recording_session.RecordingSession(
            recording_id='rec-1',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs={},
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
        )

        assert session.recording_id == 'rec-1'
        assert session.stopped is False
        assert session.is_alive is False


class TestRecordingSessionStartStop:

    def test_start_raises_if_already_started(self, tmp_path: pathlib.Path):
        """Calling start() twice should raise RuntimeError."""
        cam = FakeCamera()
        session = recording_session.RecordingSession(
            recording_id='rec-1',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs=_stream_configs(),
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
        )

        session.start()
        try:
            with pytest.raises(RuntimeError, match='already started'):
                session.start()
        finally:
            session.stop()
            session.join(timeout=2.0)

    def test_stop_sets_stopped_flag(self, tmp_path: pathlib.Path):
        """Calling stop() should set the stopped property."""
        cam = FakeCamera()
        session = recording_session.RecordingSession(
            recording_id='rec-1',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs=_stream_configs(),
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
        )

        session.start()
        session.stop()

        assert session.stopped is True
        session.join(timeout=2.0)

    def test_stop_is_idempotent(self, tmp_path: pathlib.Path):
        """Calling stop() multiple times should not raise."""
        cam = FakeCamera()
        session = recording_session.RecordingSession(
            recording_id='rec-1',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs=_stream_configs(),
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
        )

        session.start()
        session.stop()
        session.stop()  # Should not raise.
        session.join(timeout=2.0)

    def test_join_returns_true_when_not_started(self, tmp_path: pathlib.Path):
        """join() on a non-started session should return True immediately."""
        cam = FakeCamera()
        session = recording_session.RecordingSession(
            recording_id='rec-1',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs={},
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
        )

        assert session.join(timeout=1.0) is True


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestRecordingSessionFrameFlow:

    def test_frames_flow_from_camera_to_file(self, tmp_path: pathlib.Path):
        """Frames produced by cameras should be encoded and written to MKV."""
        now = datetime.datetime.now()
        frames = [(_make_frame(i), now + datetime.timedelta(milliseconds=i * 33))
                  for i in range(5)]
        cam = FakeCamera(frames=frames)

        session = recording_session.RecordingSession(
            recording_id='rec-1',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs=_stream_configs(),
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
        )

        session.start()
        time.sleep(0.5)
        session.stop()
        session.join(timeout=5.0)

        mkv_files = list(tmp_path.glob('*.mkv'))
        assert len(mkv_files) >= 1

    def test_queue_full_drops_frame_without_crashing(self, tmp_path: pathlib.Path):
        """When the encoder queue is full, frames should be dropped silently."""
        now = datetime.datetime.now()
        frames = [(_make_frame(i % 256), now) for i in range(500)]
        cam = FakeCamera(frames=frames)

        session = recording_session.RecordingSession(
            recording_id='rec-1',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs=_stream_configs(),
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
            max_queue_size=5,
        )

        session.start()
        time.sleep(0.5)
        session.stop()
        session.join(timeout=5.0)

        assert session.is_alive is False


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestCaptureThreadCrash:

    def test_camera_crash_does_not_hang_stop(self, tmp_path: pathlib.Path):
        """stop() should complete even if a capture thread crashed mid-recording."""
        cam = CrashingCamera(crash_after=3)
        session = recording_session.RecordingSession(
            recording_id='rec-crash',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs=_stream_configs(),
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
        )

        session.start()
        time.sleep(0.5)
        session.stop()
        assert session.join(timeout=5.0) is True

    def test_camera_setup_failure_does_not_hang_stop(
            self, tmp_path: pathlib.Path):
        """stop() should complete even if camera setup() raises."""
        cam = SetupFailCamera()
        session = recording_session.RecordingSession(
            recording_id='rec-setup-fail',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs=_stream_configs(),
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
        )

        session.start()
        time.sleep(0.5)
        session.stop()
        assert session.join(timeout=5.0) is True

    def test_camera_crash_logs_error(
            self, tmp_path: pathlib.Path, caplog):
        """A camera crash should be logged as an error."""
        cam = CrashingCamera(crash_after=2)
        session = recording_session.RecordingSession(
            recording_id='rec-crash-log',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs=_stream_configs(),
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
        )

        session.start()
        time.sleep(0.5)
        session.stop()
        session.join(timeout=5.0)

        crash_records = [
            r for r in caplog.records
            if 'capture thread crashed' in r.message
        ]
        assert len(crash_records) == 1


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestWriterErrorCallback:

    def test_on_error_fired_when_encoder_crashes(
            self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
        """on_error should be called when the writer's encoder thread crashes."""
        error_fired = threading.Event()

        now = datetime.datetime.now()
        frames = [
            (np.zeros((4, 4, 3), dtype=np.uint8), now + datetime.timedelta(milliseconds=i * 33))
            for i in range(10)
        ]
        cam = FakeCamera(frames=frames)

        session = recording_session.RecordingSession(
            recording_id='rec-err',
            target_cameras=[cam],
            stream_names=['rgb'],
            stream_configs=_stream_configs(),
            spool_dir=tmp_path,
            start_chunk_callback=lambda *a: 'clip-1',
            on_error=lambda: error_fired.set(),
        )

        session.start()
        time.sleep(0.3)

        # Bomb the encoder to crash it
        assert session._parallel_writer is not None
        writer = session._parallel_writer
        if 'rgb' in writer._encoders:
            def bombing_encode(data, timestamp_s):
                raise RuntimeError('simulated encoder crash')

            monkeypatch.setattr(writer._encoders['rgb'], 'encode', bombing_encode)

            # Feed another frame to trigger the crash
            eq = writer.get_encoder_queue('rgb')
            ts = now + datetime.timedelta(seconds=10)
            eq.put((np.zeros((4, 4, 3), dtype=np.uint8), ts))

            assert error_fired.wait(timeout=5.0), 'on_error was never called'

        session.stop()
        session.join(timeout=5.0)
