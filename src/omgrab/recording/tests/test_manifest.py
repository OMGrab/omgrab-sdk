"""Tests for the recording manifest module."""
import datetime
import json
import pathlib

from omgrab.recording import manifest
from omgrab.recording import stream_configs


class TestRecordingManifest:

    def test_to_dict_returns_all_fields(self):
        """to_dict should return a dictionary with all manifest fields."""
        m = manifest.RecordingManifest(
            schema_version=1,
            recording_id='2026-03-17T00-47-42Z',
            started_at='2026-03-17T00:47:42+00:00',
            stopped_at='2026-03-17T00:50:00+00:00',
            duration_s=138.0,
            devices=[manifest.DeviceInfo(label='oakd', device_type='oakd_pro_wide')],
            streams={'rgb': {'width': 640, 'height': 480, 'fps': 30}},
            sensor_streams={},
            output_file='2026-03-17T00-47-42Z.mkv',
        )
        d = m.to_dict()
        assert d['schema_version'] == 1
        assert d['recording_id'] == '2026-03-17T00-47-42Z'
        assert d['started_at'] == '2026-03-17T00:47:42+00:00'
        assert d['stopped_at'] == '2026-03-17T00:50:00+00:00'
        assert d['duration_s'] == 138.0
        assert d['devices'] == [{'label': 'oakd', 'device_type': 'oakd_pro_wide'}]
        assert d['streams'] == {'rgb': {'width': 640, 'height': 480, 'fps': 30}}
        assert d['sensor_streams'] == {}
        assert d['output_file'] == '2026-03-17T00-47-42Z.mkv'
        assert d['recovered'] is False

    def test_frozen(self):
        """RecordingManifest should be immutable."""
        m = manifest.RecordingManifest(
            schema_version=1,
            recording_id='test',
            started_at=None,
            stopped_at=None,
            duration_s=None,
            devices=[],
            streams={},
            sensor_streams={},
            output_file='test.mkv',
        )
        try:
            m.recording_id = 'changed'
            raise AssertionError('Should have raised FrozenInstanceError')
        except AttributeError:
            pass

    def test_schema_version_constant(self):
        """SCHEMA_VERSION should be set."""
        assert manifest.SCHEMA_VERSION >= 1


class TestBuildManifest:

    def test_builds_from_session_data(self):
        """build_manifest should create a manifest from typical session data."""
        started = datetime.datetime(2026, 3, 17, 0, 47, 42, tzinfo=datetime.UTC)
        stopped = datetime.datetime(2026, 3, 17, 0, 50, 0, tzinfo=datetime.UTC)
        devices = [
            manifest.DeviceInfo(label='oakd', device_type='oakd_pro_wide'),
            manifest.DeviceInfo(label='left_wrist', device_type=None),
        ]
        video_configs = {
            'rgb': stream_configs.VideoStreamConfig(
                width=1280, height=800, fps=25,
                codec='libx264', bitrate=4_000_000,
                output_pixel_format='yuv420p',
            ),
        }
        sensor_configs = {
            'imu': stream_configs.DataStreamConfig(codec='ass'),
        }

        m = manifest.build_manifest(
            recording_id='2026-03-17T00-47-42Z',
            started_at=started,
            stopped_at=stopped,
            duration_s=138.0,
            devices=devices,
            stream_configs_map=video_configs,
            sensor_stream_configs_map=sensor_configs,
            output_file='2026-03-17T00-47-42Z.mkv',
        )

        assert m.schema_version == manifest.SCHEMA_VERSION
        assert m.recording_id == '2026-03-17T00-47-42Z'
        assert m.started_at == started.isoformat()
        assert m.stopped_at == stopped.isoformat()
        assert m.duration_s == 138.0
        assert len(m.devices) == 2
        assert m.devices[0].label == 'oakd'
        assert m.recovered is False

    def test_recovered_manifest_has_null_times(self):
        """Recovered manifests should have None for timing fields."""
        m = manifest.build_manifest(
            recording_id='2026-03-17T00-47-42Z',
            started_at=None,
            stopped_at=None,
            duration_s=None,
            devices=[],
            stream_configs_map={},
            sensor_stream_configs_map={},
            output_file='2026-03-17T00-47-42Z.mkv',
            recovered=True,
        )
        assert m.started_at is None
        assert m.stopped_at is None
        assert m.duration_s is None
        assert m.recovered is True

    def test_serializes_video_stream_config(self):
        """Video stream configs should be serialized with curated fields."""
        video_configs = {
            'rgb': stream_configs.VideoStreamConfig(
                width=1280, height=800, fps=25,
                codec='libx264', bitrate=4_000_000,
                input_pixel_format='rgb24',
                output_pixel_format='yuv420p',
                stream_options={'preset': 'ultrafast'},
                metadata={'type': 'rgb'},
            ),
        }

        m = manifest.build_manifest(
            recording_id='test',
            started_at=None,
            stopped_at=None,
            duration_s=None,
            devices=[],
            stream_configs_map=video_configs,
            sensor_stream_configs_map={},
            output_file='test.mkv',
        )

        rgb = m.streams['rgb']
        assert rgb['width'] == 1280
        assert rgb['height'] == 800
        assert rgb['fps'] == 25
        assert rgb['codec'] == 'libx264'
        assert rgb['bitrate'] == 4_000_000
        assert rgb['pixel_format'] == 'yuv420p'
        assert rgb['metadata'] == {'type': 'rgb'}
        # Internal fields should not be present.
        assert 'input_pixel_format' not in rgb
        assert 'stream_options' not in rgb

    def test_serializes_data_stream_config(self):
        """Data stream configs should be serialized with curated fields."""
        sensor_configs = {
            'imu': stream_configs.DataStreamConfig(
                codec='ass',
                metadata={'format': 'json'},
            ),
        }

        m = manifest.build_manifest(
            recording_id='test',
            started_at=None,
            stopped_at=None,
            duration_s=None,
            devices=[],
            stream_configs_map={},
            sensor_stream_configs_map=sensor_configs,
            output_file='test.mkv',
        )

        imu = m.sensor_streams['imu']
        assert imu['codec'] == 'ass'
        assert imu['metadata'] == {'format': 'json'}


class TestWriteManifest:

    def test_writes_json_file(self, tmp_path: pathlib.Path):
        """write_manifest should create a parseable JSON file."""
        m = manifest.RecordingManifest(
            schema_version=1,
            recording_id='2026-03-17T00-47-42Z',
            started_at='2026-03-17T00:47:42+00:00',
            stopped_at='2026-03-17T00:50:00+00:00',
            duration_s=138.0,
            devices=[manifest.DeviceInfo(label='oakd', device_type='oakd_pro_wide')],
            streams={'rgb': {'width': 640}},
            sensor_streams={},
            output_file='2026-03-17T00-47-42Z.mkv',
        )

        manifest.write_manifest(m, tmp_path)
        manifest_path = tmp_path / '2026-03-17T00-47-42Z.json'

        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text())
        assert data['recording_id'] == '2026-03-17T00-47-42Z'
        assert data['schema_version'] == 1

    def test_filename_matches_recording_id(self, tmp_path: pathlib.Path):
        """Manifest filename should be {recording_id}.json."""
        m = manifest.RecordingManifest(
            schema_version=1,
            recording_id='my-recording',
            started_at=None,
            stopped_at=None,
            duration_s=None,
            devices=[],
            streams={},
            sensor_streams={},
            output_file='my-recording.mkv',
        )

        manifest.write_manifest(m, tmp_path)
        assert (tmp_path / 'my-recording.json').exists()

    def test_no_tmp_file_after_write(self, tmp_path: pathlib.Path):
        """Temporary .json.tmp file should not remain after successful write."""
        m = manifest.RecordingManifest(
            schema_version=1,
            recording_id='test',
            started_at=None,
            stopped_at=None,
            duration_s=None,
            devices=[],
            streams={},
            sensor_streams={},
            output_file='test.mkv',
        )

        manifest.write_manifest(m, tmp_path)
        tmp_files = list(tmp_path.glob('*.json.tmp'))
        assert len(tmp_files) == 0
