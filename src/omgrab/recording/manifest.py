"""Recording manifest: metadata sidecar for MKV recordings."""
from typing import Any
from typing import Optional

import dataclasses
import datetime
import json
import logging
import os
import pathlib

from omgrab.recording import stream_configs

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class DeviceInfo:
    """Capture device descriptor for the manifest.

    Attributes:
        label: Human-readable device label (e.g. 'oakd', 'left_wrist').
        device_type: Device type identifier, or None if not detected.
    """

    label: str
    device_type: Optional[str]


@dataclasses.dataclass(frozen=True)
class RecordingManifest:
    """Metadata sidecar for a merged recording.

    Attributes:
        schema_version: Manifest format version.
        recording_id: Timestamp-based recording identifier.
        started_at: UTC ISO 8601 timestamp when capture began.
        stopped_at: UTC ISO 8601 timestamp when capture ended.
        duration_s: Recording duration in seconds.
        devices: Capture devices used in this recording.
        streams: Video stream configurations, keyed by stream name.
        sensor_streams: Data stream configurations, keyed by stream name.
        output_file: Filename of the merged MKV.
        recovered: True if this was an orphaned recording merged at boot.
    """

    schema_version: int
    recording_id: str
    started_at: Optional[str]
    stopped_at: Optional[str]
    duration_s: Optional[float]
    devices: list[DeviceInfo]
    streams: dict[str, dict[str, Any]]
    sensor_streams: dict[str, dict[str, Any]]
    output_file: str
    recovered: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return dataclasses.asdict(self)


def _serialize_video_stream_config(
        config: stream_configs.VideoStreamConfig) -> dict[str, Any]:
    """Serialize a VideoStreamConfig to a manifest-safe dict.

    Args:
        config: The video stream configuration.

    Returns:
        A dictionary with consumer-relevant fields.
    """
    return {
        'width': config.width,
        'height': config.height,
        'fps': float(config.fps),
        'codec': config.codec,
        'bitrate': config.bitrate,
        'pixel_format': config.output_pixel_format,
        'metadata': config.metadata,
    }


def _serialize_data_stream_config(
        config: stream_configs.DataStreamConfig) -> dict[str, Any]:
    """Serialize a DataStreamConfig to a manifest-safe dict.

    Args:
        config: The data stream configuration.

    Returns:
        A dictionary with consumer-relevant fields.
    """
    return {
        'codec': config.codec,
        'metadata': config.metadata,
    }


def build_manifest(
        recording_id: str,
        started_at: Optional[datetime.datetime],
        stopped_at: Optional[datetime.datetime],
        duration_s: Optional[float],
        devices: list[DeviceInfo],
        stream_configs_map: dict[str, stream_configs.VideoStreamConfig],
        sensor_stream_configs_map: dict[str, stream_configs.DataStreamConfig],
        output_file: str,
        recovered: bool = False) -> RecordingManifest:
    """Build a RecordingManifest from recording session data.

    Args:
        recording_id: Timestamp-based recording identifier.
        started_at: UTC datetime when capture began, or None for recovered.
        stopped_at: UTC datetime when capture ended, or None for recovered.
        duration_s: Recording duration in seconds, or None.
        devices: Capture devices used in this recording.
        stream_configs_map: Video stream configurations.
        sensor_stream_configs_map: Sensor/data stream configurations.
        output_file: Filename of the merged MKV.
        recovered: True if this was an orphaned recording.

    Returns:
        A frozen RecordingManifest instance.
    """
    return RecordingManifest(
        schema_version=SCHEMA_VERSION,
        recording_id=recording_id,
        started_at=started_at.isoformat() if started_at else None,
        stopped_at=stopped_at.isoformat() if stopped_at else None,
        duration_s=duration_s,
        devices=devices,
        streams={
            name: _serialize_video_stream_config(config)
            for name, config in stream_configs_map.items()
        },
        sensor_streams={
            name: _serialize_data_stream_config(config)
            for name, config in sensor_stream_configs_map.items()
        },
        output_file=output_file,
        recovered=recovered,
    )


def write_manifest(recording_manifest: RecordingManifest, output_dir: pathlib.Path):
    """Write a manifest JSON file alongside the recording MKV.

    Uses atomic write (tmp + rename + fsync) to prevent partial reads.

    Args:
        recording_manifest: The manifest to write.
        output_dir: Directory containing the merged MKV.
    """
    manifest_path = output_dir / f'{recording_manifest.recording_id}.json'
    tmp_path = manifest_path.with_suffix('.json.tmp')

    with open(tmp_path, 'w') as f:
        json.dump(recording_manifest.to_dict(), f, indent=2)
    tmp_path.rename(manifest_path)
    try:
        with open(manifest_path, 'rb+') as f:
            os.fsync(f)
    except FileNotFoundError:
        logger.warning('Manifest %s not found for fsync', manifest_path)

    logger.info('Manifest written: %s', manifest_path)
