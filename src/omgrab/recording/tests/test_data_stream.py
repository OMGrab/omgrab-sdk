"""Prototype test: validate PyAV subtitle packet muxing + ffmpeg concat.

Tests:
1. Write an MKV with a video stream and a subtitle (data) stream using PyAV
2. Read back the subtitle packets and verify byte-for-byte preservation
3. Concatenate two such MKVs with ffmpeg
4. Read back from the concatenated file and verify all data is preserved
"""

import json
import pathlib
import subprocess

import av
import numpy as np
import pytest

from omgrab.recording import py_av_writer


def _write_test_mkv(
        output_path: pathlib.Path, imu_readings: list[dict],
        start_timestamp_s: float = 1000.0):
    """Write a test MKV with one video stream and one data (subtitle) stream.

    Args:
        output_path: Path to write the MKV file.
        imu_readings: List of dicts, each written as a separate subtitle packet.
        start_timestamp_s: Starting timestamp in seconds since epoch.
    """
    container = av.open(str(output_path), mode='w', format='matroska')

    video_encoder = py_av_writer.VideoStreamEncoder(
        container=container,
        width=64,
        height=64,
        fps=25,
        codec='libx264',
        bitrate=500_000,
        input_pixel_format='rgb24',
        output_pixel_format='yuv420p',
        options={'preset': 'ultrafast'},
        timestamp_origin_s=start_timestamp_s,
        metadata={'type': 'rgb'},
    )

    data_encoder = py_av_writer.DataStreamEncoder(
        container=container,
        timestamp_origin_s=start_timestamp_s,
        codec='ass',
        metadata={'type': 'imu', 'format': 'json', 'version': '1'},
    )

    # Write interleaved video frames and IMU data.
    # 4 IMU readings per video frame (simulating 100Hz IMU with 25fps video).
    frame_period_s = 1.0 / 25.0
    imu_period_s = 1.0 / 100.0
    imu_idx = 0

    for frame_i in range(len(imu_readings) // 4):
        frame_ts = start_timestamp_s + frame_i * frame_period_s

        # Write 4 IMU readings for this frame
        for _ in range(4):
            if imu_idx >= len(imu_readings):
                break
            imu_ts = start_timestamp_s + imu_idx * imu_period_s
            payload = json.dumps(imu_readings[imu_idx], separators=(',', ':')).encode('utf-8')
            for pkt in data_encoder.encode(payload, imu_ts):
                container.mux(pkt)
            imu_idx += 1

        # Write a video frame (solid color)
        frame = np.full((64, 64, 3), fill_value=(frame_i * 10) % 256, dtype=np.uint8)
        for pkt in video_encoder.encode(frame, frame_ts):
            container.mux(pkt)

    for pkt in video_encoder.flush():
        container.mux(pkt)
    container.close()


def _read_subtitle_packets(mkv_path: pathlib.Path) -> list[tuple[float, bytes]]:
    """Read all subtitle packets from an MKV file.

    Returns:
        List of (timestamp_s, payload_bytes) tuples.
    """
    container = av.open(str(mkv_path), mode='r')
    results: list[tuple[float, bytes]] = []

    try:
        if not container.streams.subtitles:
            return results

        stream = container.streams.subtitles[0]
        assert stream.time_base is not None
        time_base = float(stream.time_base)

        for packet in container.demux(stream):
            if packet.dts is None:
                continue
            assert packet.pts is not None
            timestamp_s = packet.pts * time_base
            results.append((timestamp_s, bytes(packet)))
    finally:
        container.close()

    return results


def _read_subtitle_metadata(mkv_path: pathlib.Path) -> dict[str, str]:
    """Read metadata from the first subtitle stream."""
    container = av.open(str(mkv_path), mode='r')
    try:
        if not container.streams.subtitles:
            return {}
        return dict(container.streams.subtitles[0].metadata)
    finally:
        container.close()


def _make_imu_readings(count: int) -> list[dict]:
    """Generate test IMU readings."""
    readings = []
    for i in range(count):
        readings.append(
            {
                'a': [round(0.01 * i, 4), -9.81, round(0.02 * i, 4)],
                'g': [round(0.001 * i, 5), round(-0.002 * i, 5), round(0.003 * i, 5)],
            }
        )
    return readings


class TestDataStreamWriteRead:
    """Test writing and reading data streams in MKV."""

    def test_write_and_read_back(self, tmp_path):
        """Write IMU data to MKV subtitle track and read it back byte-for-byte."""
        mkv_path = tmp_path / 'test.mkv'
        readings = _make_imu_readings(20)

        _write_test_mkv(mkv_path, readings)

        # Read back
        packets = _read_subtitle_packets(mkv_path)
        assert len(packets) == 20

        for i, (_, payload) in enumerate(packets):
            decoded = json.loads(payload.decode('utf-8'))
            assert decoded == readings[i], f'Mismatch at packet {i}: {decoded} != {readings[i]}'

    def test_stream_metadata_preserved(self, tmp_path):
        """Verify that stream metadata (type, format, version) is preserved."""
        mkv_path = tmp_path / 'test.mkv'
        readings = _make_imu_readings(4)

        _write_test_mkv(mkv_path, readings)

        metadata = _read_subtitle_metadata(mkv_path)
        # MKV stores metadata keys in uppercase
        assert metadata.get('TYPE') == 'imu'
        assert metadata.get('FORMAT') == 'json'
        assert metadata.get('VERSION') == '1'

    def test_timestamps_increase_monotonically(self, tmp_path):
        """Verify that PTS timestamps in the subtitle stream are monotonically increasing."""
        mkv_path = tmp_path / 'test.mkv'
        readings = _make_imu_readings(20)

        _write_test_mkv(mkv_path, readings)

        packets = _read_subtitle_packets(mkv_path)
        timestamps = [ts for ts, _ in packets]

        for i in range(1, len(timestamps)):
            assert timestamps[i] > timestamps[i - 1], (
                f'Timestamps not monotonic at index {i}: {timestamps[i]} <= {timestamps[i - 1]}'
            )


class TestDataStreamConcat:
    """Test that ffmpeg concat preserves subtitle data streams."""

    @pytest.fixture
    def ffmpeg_available(self):
        """Skip test if ffmpeg is not available."""
        try:
            subprocess.run(
                ['ffmpeg', '-version'],
                capture_output=True,
                check=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip('ffmpeg not available')

    def test_concat_preserves_data(self, tmp_path, ffmpeg_available):
        """Concatenate two MKVs with ffmpeg and verify all subtitle data is preserved."""
        # Write two MKV files
        mkv1 = tmp_path / 'clip1.mkv'
        mkv2 = tmp_path / 'clip2.mkv'
        readings1 = _make_imu_readings(20)
        readings2 = _make_imu_readings(20)

        _write_test_mkv(mkv1, readings1, start_timestamp_s=1000.0)
        _write_test_mkv(mkv2, readings2, start_timestamp_s=2000.0)

        # Concatenate with ffmpeg (same command as concat_mkv job)
        concat_list = tmp_path / 'concat.txt'
        concat_list.write_text(f"file '{mkv1.absolute()}'\nfile '{mkv2.absolute()}'\n")

        output_mkv = tmp_path / 'concatenated.mkv'
        subprocess.run(
            [
                'ffmpeg',
                '-hide_banner',
                '-loglevel',
                'error',
                '-f',
                'concat',
                '-safe',
                '0',
                '-i',
                str(concat_list),
                '-map',
                '0',
                '-c',
                'copy',
                '-y',
                str(output_mkv),
            ],
            check=True,
        )

        assert output_mkv.exists()

        # Read back all subtitle packets from concatenated file
        packets = _read_subtitle_packets(output_mkv)
        assert len(packets) == 40, f'Expected 40 packets, got {len(packets)}'

        # Verify all payloads are preserved (first 20 from clip1, next 20 from clip2)
        all_readings = readings1 + readings2
        for i, (_, payload) in enumerate(packets):
            decoded = json.loads(payload.decode('utf-8'))
            assert decoded == all_readings[i], (
                f'Mismatch at packet {i}: {decoded} != {all_readings[i]}'
            )

    def test_concat_preserves_metadata(self, tmp_path, ffmpeg_available):
        """Verify stream metadata survives ffmpeg concat."""
        mkv1 = tmp_path / 'clip1.mkv'
        mkv2 = tmp_path / 'clip2.mkv'
        readings = _make_imu_readings(4)

        _write_test_mkv(mkv1, readings, start_timestamp_s=1000.0)
        _write_test_mkv(mkv2, readings, start_timestamp_s=2000.0)

        concat_list = tmp_path / 'concat.txt'
        concat_list.write_text(f"file '{mkv1.absolute()}'\nfile '{mkv2.absolute()}'\n")

        output_mkv = tmp_path / 'concatenated.mkv'
        subprocess.run(
            [
                'ffmpeg',
                '-hide_banner',
                '-loglevel',
                'error',
                '-f',
                'concat',
                '-safe',
                '0',
                '-i',
                str(concat_list),
                '-map',
                '0',
                '-c',
                'copy',
                '-y',
                str(output_mkv),
            ],
            check=True,
        )

        metadata = _read_subtitle_metadata(output_mkv)
        assert metadata.get('TYPE') == 'imu'
        assert metadata.get('FORMAT') == 'json'
        assert metadata.get('VERSION') == '1'
