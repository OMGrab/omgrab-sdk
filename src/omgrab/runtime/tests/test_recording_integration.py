"""Integration tests: RecordingManager → RecordingSession → ChunkedWriter → merge.

These tests exercise the full recording pipeline with real encoding
but fake cameras/devices. They require ffmpeg to be installed.
"""
from typing import Optional

import datetime
import fractions
import json
import pathlib
import shutil
import time

import av
import numpy as np
import pytest

from omgrab import testing
from omgrab.cameras import cameras
from omgrab.recording import chunked_writer
from omgrab.runtime import recording_manager
from omgrab.runtime import recording_session
from omgrab.sensors import sensor

pytestmark = pytest.mark.skipif(
    not shutil.which('ffmpeg'), reason='ffmpeg not installed')


@pytest.fixture(autouse=True)
def _fast_drain(monkeypatch):
    """Reduce writer drain quiet period so tests finish quickly."""
    monkeypatch.setattr(recording_session, 'WRITER_DRAIN_QUIET_PERIOD_S', 0.05)


class FakeCamera(cameras.Camera):
    """Camera that yields frames on demand, resetting each time it's opened."""

    def __init__(self, num_frames: int = 10):
        super().__init__(
            cameras.CameraConfig(fps=30, width=4, height=4),
            enforce_frame_timing=False,
        )
        self._num_frames = num_frames
        self._produced = 0
        self._start = datetime.datetime.now(datetime.UTC)

    def get_next_frame(self, timeout_s=None):
        if self._produced < self._num_frames:
            frame = np.zeros((4, 4, 3), dtype=np.uint8) + (self._produced % 256)
            ts = self._start + datetime.timedelta(milliseconds=self._produced * 33)
            self._produced += 1
            return frame, ts
        raise cameras.FrameUnavailableError('exhausted')

    def setup(self):
        self._produced = 0
        self._start = datetime.datetime.now(datetime.UTC)

    def close(self):
        pass


class FakeSensor(sensor.Sensor):
    """Sensor that yields data items, resetting each time it's opened."""

    def __init__(self, items: Optional[list[tuple[bytes, datetime.datetime]]] = None):
        self._original_items = items or []
        self._items = list(self._original_items)
        self._index = 0

    def get_next_item(self, timeout_s=None):
        if self._index < len(self._items):
            data, ts = self._items[self._index]
            self._index += 1
            return data, ts
        raise sensor.SensorDataUnavailableError('exhausted')

    def setup(self):
        self._index = 0

    def close(self):
        pass


def _video_stream_configs() -> dict[str, chunked_writer.VideoStreamConfig]:
    return {
        'rgb': chunked_writer.VideoStreamConfig(
            width=4, height=4, fps=fractions.Fraction(30),
            codec='libx264',
            stream_options={'preset': 'ultrafast'},
            metadata={'type': 'rgb'},
        ),
    }


def _make_manager(
        spool_dir: pathlib.Path,
        output_dir: pathlib.Path,
        cam: Optional[FakeCamera] = None,
        device: Optional[testing.FakeCaptureDevice] = None,
        on_recording_complete=None,
        sensors: Optional[list[sensor.Sensor]] = None,
        sensor_stream_names: Optional[list[str]] = None,
        sensor_stream_configs: Optional[dict[str, chunked_writer.DataStreamConfig]] = None,
) -> recording_manager.RecordingManager:
    """Create a RecordingManager wired to real sessions with fake hardware."""
    if cam is None:
        cam = FakeCamera()
    if device is None:
        device = testing.FakeCaptureDevice(device_type='oakd_pro_wide')

    return recording_manager.RecordingManager(
        devices=[device],
        target_cameras=[cam],
        stream_names=['rgb'],
        stream_configs=_video_stream_configs(),
        spool_dir=spool_dir,
        output_dir=output_dir,
        config=recording_manager.RecordingConfig(
            health_check_interval_s=0.05,
        ),
        on_recording_complete=on_recording_complete,
        sensors=sensors,
        sensor_stream_names=sensor_stream_names,
        sensor_stream_configs=sensor_stream_configs,
    )


class TestRecordingIntegration:

    def test_recording_produces_merged_mkv(self, spool_dirs):
        """Full pipeline should produce a merged MKV in output_dir."""
        mgr = _make_manager(spool_dirs['spool'], spool_dirs['output'])

        assert mgr.start_recording() is True
        time.sleep(0.5)
        mgr.stop_recording()

        mkv_files = list(spool_dirs['output'].glob('*.mkv'))
        assert len(mkv_files) == 1
        assert mkv_files[0].stat().st_size > 0

        # Spool directory should be cleaned up after merge.
        spool_subdirs = [d for d in spool_dirs['spool'].iterdir() if d.is_dir()]
        assert len(spool_subdirs) == 0

    def test_recording_produces_manifest(self, spool_dirs):
        """Full pipeline should produce a JSON manifest alongside the MKV."""
        mgr = _make_manager(spool_dirs['spool'], spool_dirs['output'])

        mgr.start_recording()
        recording_id = mgr.active_recording_id
        time.sleep(0.5)
        mgr.stop_recording()

        manifest_path = spool_dirs['output'] / f'{recording_id}.json'
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert data['schema_version'] == 1
        assert data['recording_id'] == recording_id
        assert data['started_at'] is not None
        assert data['stopped_at'] is not None
        assert data['duration_s'] >= 0
        assert data['recovered'] is False
        assert len(data['devices']) == 1
        assert data['devices'][0]['device_type'] == 'oakd_pro_wide'
        assert 'rgb' in data['streams']
        assert data['streams']['rgb']['width'] == 4
        assert data['streams']['rgb']['height'] == 4
        assert data['output_file'] == f'{recording_id}.mkv'

    def test_merged_mkv_has_expected_streams(self, spool_dirs):
        """Merged MKV should contain a video stream with correct dimensions."""
        mgr = _make_manager(spool_dirs['spool'], spool_dirs['output'])

        mgr.start_recording()
        time.sleep(0.5)
        mgr.stop_recording()

        mkv_files = list(spool_dirs['output'].glob('*.mkv'))
        assert len(mkv_files) == 1

        with av.open(str(mkv_files[0])) as container:
            assert len(container.streams.video) >= 1
            video_stream = container.streams.video[0]
            assert video_stream.width == 4
            assert video_stream.height == 4

    def test_recording_with_sensor_data(self, spool_dirs):
        """Recording with camera + sensor should produce MKV with both streams."""
        now = datetime.datetime.now(datetime.UTC)
        imu_items = [
            (
                json.dumps({'accel': [0, 0, 9.8]}).encode(),
                now + datetime.timedelta(milliseconds=i * 100),
            )
            for i in range(5)
        ]
        fake_sensor = FakeSensor(items=imu_items)

        mgr = _make_manager(
            spool_dirs['spool'],
            spool_dirs['output'],
            sensors=[fake_sensor],
            sensor_stream_names=['imu'],
            sensor_stream_configs={
                'imu': chunked_writer.DataStreamConfig(
                    codec='ass',
                    metadata={'type': 'imu'},
                ),
            },
        )

        mgr.start_recording()
        time.sleep(0.5)
        mgr.stop_recording()

        mkv_files = list(spool_dirs['output'].glob('*.mkv'))
        assert len(mkv_files) == 1

        with av.open(str(mkv_files[0])) as container:
            assert len(container.streams.video) >= 1
            assert len(container.streams.subtitles) >= 1

    def test_on_recording_complete_callback(self, spool_dirs):
        """on_recording_complete should be called with the merged MKV path."""
        callback_paths: list[pathlib.Path] = []
        mgr = _make_manager(
            spool_dirs['spool'],
            spool_dirs['output'],
            on_recording_complete=lambda p: callback_paths.append(p),
        )

        mgr.start_recording()
        time.sleep(0.5)
        mgr.stop_recording()

        assert len(callback_paths) == 1
        assert callback_paths[0].exists()
        assert callback_paths[0].suffix == '.mkv'

    def test_sequential_recordings(self, spool_dirs):
        """Two sequential recordings should produce two separate MKVs."""
        mgr = _make_manager(spool_dirs['spool'], spool_dirs['output'])

        mgr.start_recording()
        first_id = mgr.active_recording_id
        time.sleep(0.5)
        mgr.stop_recording()

        # Small sleep to ensure distinct timestamp-based recording names.
        time.sleep(1.1)

        mgr.start_recording()
        second_id = mgr.active_recording_id
        time.sleep(0.5)
        mgr.stop_recording()

        assert first_id != second_id

        mkv_files = list(spool_dirs['output'].glob('*.mkv'))
        assert len(mkv_files) == 2

        json_files = list(spool_dirs['output'].glob('*.json'))
        assert len(json_files) == 2
