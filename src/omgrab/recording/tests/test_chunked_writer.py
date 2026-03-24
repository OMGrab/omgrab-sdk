"""Tests for chunked_writer.py — ChunkedWriter, pure helpers, config dataclasses."""
import datetime
import fractions
import pathlib
import queue
import shutil
import threading
import time

import numpy as np
import pytest

from omgrab.recording import chunked_writer

_has_ffmpeg = shutil.which('ffmpeg') is not None



def _make_rgb_frame(width: int = 64, height: int = 64, value: int = 128) -> np.ndarray:
    """Create a solid-color RGB frame."""
    return np.full((height, width, 3), fill_value=value, dtype=np.uint8)


def _ts(year: int = 2025, month: int = 1, day: int = 1,
         hour: int = 12, minute: int = 0, second: int = 0,
         microsecond: int = 0) -> datetime.datetime:
    """Shorthand for creating a datetime."""
    return datetime.datetime(year, month, day, hour, minute, second, microsecond)



class TestVerifyMetadata:

    def test_valid_metadata_passes(self):
        """Metadata with 'type' key should not raise."""
        chunked_writer._verify_metadata({'type': 'rgb'})  # Should not raise.

    def test_missing_type_raises(self):
        """Metadata without 'type' key should raise ValueError."""
        with pytest.raises(ValueError, match='type'):
            chunked_writer._verify_metadata({'codec': 'libx264'})

    def test_empty_metadata_raises(self):
        """Empty metadata should raise ValueError."""
        with pytest.raises(ValueError, match='type'):
            chunked_writer._verify_metadata({})

    def test_extra_keys_are_fine(self):
        """Extra keys beyond the expected ones should be allowed."""
        chunked_writer._verify_metadata({
            'type': 'depth',
            'extra': 'value',
            'another': 123,
        })



class TestVideoStreamConfig:

    def test_default_values(self):
        """VideoStreamConfig should have sensible defaults."""
        config = chunked_writer.VideoStreamConfig(width=640, height=480, fps=30.0)

        assert config.codec == 'libx264'
        assert config.bitrate == 2_000_000
        assert config.input_pixel_format == 'rgb24'
        assert config.output_pixel_format == 'yuv420p'
        assert config.stream_options == {}
        assert config.metadata == {}

    def test_custom_values(self):
        """Custom values should override defaults."""
        config = chunked_writer.VideoStreamConfig(
            width=320,
            height=240,
            fps=15.0,
            codec='ffv1',
            bitrate=0,
            input_pixel_format='gray16le',
            output_pixel_format='gray16le',
            stream_options={'level': '3'},
            metadata={'type': 'depth'},
        )

        assert config.width == 320
        assert config.codec == 'ffv1'
        assert config.metadata == {'type': 'depth'}


class TestDataStreamConfig:

    def test_default_values(self):
        """DataStreamConfig should default to ASS codec."""
        config = chunked_writer.DataStreamConfig()

        assert config.codec == 'ass'
        assert config.metadata == {}

    def test_custom_values(self):
        """Custom values should be stored."""
        config = chunked_writer.DataStreamConfig(
            codec='ass', metadata={'type': 'imu', 'rate_hz': 100})

        assert config.metadata['type'] == 'imu'



def _parallel_stream_configs():
    """Small stream configs for ChunkedWriter tests."""
    return {
        'rgb': chunked_writer.VideoStreamConfig(
            width=64, height=64, fps=fractions.Fraction(25),
            codec='libx264',
            stream_options={'preset': 'ultrafast'},
            metadata={'type': 'rgb'},
        ),
    }


def _parallel_multi_stream_configs():
    """Multiple stream configs for ChunkedWriter tests.

    Uses bf=0 to match production config and avoid B-frame reordering
    issues that cause negative DTS in multi-stream MKV containers.
    """
    return {
        'rgb': chunked_writer.VideoStreamConfig(
            width=64, height=64, fps=fractions.Fraction(25),
            codec='libx264',
            stream_options={'preset': 'ultrafast', 'bf': '0'},
            metadata={'type': 'rgb'},
        ),
        'depth': chunked_writer.VideoStreamConfig(
            width=64, height=64, fps=fractions.Fraction(25),
            codec='ffv1',
            input_pixel_format='gray16le',
            output_pixel_format='gray16le',
            metadata={'type': 'depth'},
        ),
    }


def _make_depth_frame(width: int = 64, height: int = 64, value: int = 1000) -> np.ndarray:
    """Create a depth frame (uint16, 2D)."""
    return np.full((height, width), fill_value=value, dtype=np.uint16)


def _make_chunked_writer(
        tmp_path: pathlib.Path,
        stream_configs: dict[str, chunked_writer.VideoStreamConfig] | None = None,
        sensor_stream_configs: dict[str, chunked_writer.DataStreamConfig] | None = None,
        chunk_ids: list[str] | None = None,
        name: str = 'rec-1',
        chunk_length_s: float = 60.0,
        max_encoder_queue_size: int = 200,
) -> tuple[chunked_writer.ChunkedWriter, list[str]]:
    """Create a ChunkedWriter and return it with the chunk IDs list."""
    if stream_configs is None:
        stream_configs = _parallel_stream_configs()

    chunk_ids_returned: list[str] = []
    if chunk_ids is None:
        chunk_ids = [f'chunk-{i}' for i in range(100)]
    chunk_iter = iter(chunk_ids)

    def callback(writer_name, started_at, ext):
        cid = next(chunk_iter)
        chunk_ids_returned.append(cid)
        return cid

    writer = chunked_writer.ChunkedWriter(
        name=name,
        output_directory=tmp_path,
        stream_configs=stream_configs,
        start_chunk_callback=callback,
        sensor_stream_configs=sensor_stream_configs,
        chunk_length_s=chunk_length_s,
        max_encoder_queue_size=max_encoder_queue_size,
    )
    return writer, chunk_ids_returned


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestChunkedWriterBasic:
    """Basic lifecycle tests for ChunkedWriter."""

    def test_start_stop_without_frames(self, tmp_path: pathlib.Path):
        """Starting and stopping with no frames should not crash."""
        writer, chunk_ids = _make_chunked_writer(tmp_path)
        writer.start()
        writer.stop()

        # No chunk created since no frames arrived (lazy init).
        assert len(chunk_ids) == 0

    def test_single_frame_produces_mkv(self, tmp_path: pathlib.Path):
        """Writing a single frame should produce a readable MKV file."""
        writer, chunk_ids = _make_chunked_writer(tmp_path)
        writer.start()

        eq = writer.get_encoder_queue('rgb')
        eq.put((_make_rgb_frame(), _ts(minute=10, second=0)))

        import time
        time.sleep(0.5)
        writer.stop()

        assert len(chunk_ids) == 1
        mkv_path = tmp_path / 'chunk-0.mkv'
        assert mkv_path.exists()

    def test_multiple_frames_within_chunk(self, tmp_path: pathlib.Path):
        """Multiple frames within chunk_length_s should produce exactly one chunk."""
        writer, chunk_ids = _make_chunked_writer(tmp_path, chunk_length_s=60)
        writer.start()

        eq = writer.get_encoder_queue('rgb')
        for i in range(10):
            eq.put((_make_rgb_frame(value=i * 20), _ts(minute=10, second=i * 5)))

        import time
        time.sleep(0.5)
        writer.stop()

        assert len(chunk_ids) == 1

    def test_stop_is_clean_no_exceptions(self, tmp_path: pathlib.Path):
        """Stopping after writing frames should not raise any exceptions."""
        writer, _ = _make_chunked_writer(tmp_path)
        writer.start()

        eq = writer.get_encoder_queue('rgb')
        for i in range(20):
            eq.put((_make_rgb_frame(), _ts(minute=10, second=i * 2)))

        import time
        time.sleep(0.5)

        # stop() should not raise
        writer.stop()


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestChunkedWriterRotation:
    """Tests for duration-based file rotation.

    All tests use deterministic fixed timestamps (year=2030) and short
    chunk_length_s to avoid flakiness.
    """

    def test_boundary_creates_exactly_two_chunks(self, tmp_path: pathlib.Path):
        """Frames spanning one chunk boundary should produce exactly 2 chunks."""
        base = _ts(year=2030, minute=10, second=0)
        writer, chunk_ids = _make_chunked_writer(tmp_path, chunk_length_s=10)
        writer.start()

        eq = writer.get_encoder_queue('rgb')
        # 5 frames in first 10s
        for i in range(5):
            ts = base + datetime.timedelta(seconds=i * 2)
            eq.put((_make_rgb_frame(), ts))
        # 5 frames after the 10s boundary
        for i in range(5):
            ts = base + datetime.timedelta(seconds=10 + i * 2)
            eq.put((_make_rgb_frame(), ts))

        import time
        time.sleep(2.0)
        writer.stop()

        assert len(chunk_ids) == 2, (
            f'Expected exactly 2 chunks but got {len(chunk_ids)}.')

    def test_multi_stream_rotation_creates_exactly_two_chunks(self, tmp_path: pathlib.Path):
        """Multiple streams crossing a chunk boundary should produce exactly 2 chunks.

        This tests the parallel rotation protocol: all encoder threads must
        coordinate to produce exactly one rotation at the boundary.
        """
        base = _ts(year=2030, minute=10, second=0)
        writer, chunk_ids = _make_chunked_writer(
            tmp_path, stream_configs=_parallel_multi_stream_configs(),
            chunk_length_s=10)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')

        # Feed frames to both streams: 5 in first chunk, 5 in second
        for i in range(5):
            ts = base + datetime.timedelta(seconds=i * 2)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))
        for i in range(5):
            ts = base + datetime.timedelta(seconds=10 + i * 2)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        import time
        time.sleep(2.0)
        writer.stop()

        assert len(chunk_ids) == 2, (
            f'Expected 2 chunks with multi-stream rotation but got {len(chunk_ids)}')

    def test_multi_stream_with_data_rotation_creates_exactly_two_chunks(
            self, tmp_path: pathlib.Path):
        """Video + data streams crossing a chunk boundary should produce exactly 2 chunks.

        Data streams (e.g. IMU at 100Hz) are particularly prone to triggering
        repeated rotations because they produce items much faster than video.
        """
        base = _ts(year=2030, minute=10, second=0)
        data_configs = {
            'imu': chunked_writer.DataStreamConfig(metadata={'type': 'imu'}),
        }
        writer, chunk_ids = _make_chunked_writer(
            tmp_path, sensor_stream_configs=data_configs, chunk_length_s=10)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        imu_q = writer.get_encoder_queue('imu')

        # Video frames: 3 before boundary, 3 after
        for i in range(3):
            ts = base + datetime.timedelta(seconds=i * 3)
            rgb_q.put((_make_rgb_frame(), ts))
        for i in range(3):
            ts = base + datetime.timedelta(seconds=10 + i * 3)
            rgb_q.put((_make_rgb_frame(), ts))

        # IMU data: many items spanning the boundary (simulating 100Hz)
        for i in range(200):
            ts = base + datetime.timedelta(milliseconds=i * 100)
            imu_q.put((b'{"ax":0.1}', ts))

        import time
        time.sleep(2.0)
        writer.stop()

        assert len(chunk_ids) == 2, (
            f'Expected 2 chunks with video+data rotation but got {len(chunk_ids)}. '
            f'High-rate data streams may be triggering repeated rotations.')

    def test_three_chunks_creates_exactly_three_chunks(self, tmp_path: pathlib.Path):
        """Frames spanning three consecutive chunks should produce exactly 3 chunks."""
        base = _ts(year=2030, minute=10, second=0)
        writer, chunk_ids = _make_chunked_writer(tmp_path, chunk_length_s=10)
        writer.start()

        eq = writer.get_encoder_queue('rgb')
        # 3 frames per 10s chunk, spanning 30s total
        for chunk in range(3):
            for i in range(3):
                ts = base + datetime.timedelta(seconds=chunk * 10 + i * 3)
                eq.put((_make_rgb_frame(), ts))

        import time
        time.sleep(3.0)
        writer.stop()

        assert len(chunk_ids) == 3, (
            f'Expected 3 chunks for 3 chunks but got {len(chunk_ids)}')

    def test_shared_timestamp_origin(self, tmp_path: pathlib.Path):
        """All streams in a chunk should share the same timestamp origin.

        Source 2's first frame arriving 1s after source 1's first frame
        should get PTS corresponding to 1s, not PTS=0. We verify this
        by reading the per-stream .part files directly (before ffmpeg
        merge normalizes stream start times).
        """
        base = _ts(year=2030, minute=10, second=0)
        writer, chunk_ids = _make_chunked_writer(
            tmp_path, stream_configs=_parallel_multi_stream_configs(),
            chunk_length_s=60)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')

        # RGB frame arrives first at base
        rgb_q.put((_make_rgb_frame(), base))
        import time
        time.sleep(0.2)
        # Depth frame arrives 1s later
        depth_q.put((_make_depth_frame(), base + datetime.timedelta(seconds=1)))

        # A few more frames for both streams
        for i in range(1, 5):
            ts = base + datetime.timedelta(seconds=i)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts + datetime.timedelta(seconds=1)))

        time.sleep(1.0)

        # Stop encoder threads but DON'T merge yet — we need the .part files.
        writer._stop_event.set()
        with writer._rotation_condition:
            writer._rotation_generation += 1
            writer._rotation_condition.notify_all()
        for t in writer._encoder_threads:
            t.join()

        # Read the depth .part file BEFORE merge deletes it.
        import av
        depth_part = tmp_path / f'{chunk_ids[0]}.depth.mkv.part'
        assert depth_part.exists(), 'Depth .part file not found'

        container = av.open(str(depth_part), mode='r')
        try:
            stream = container.streams[0]
            first_pkt = next(container.demux(stream))
            # Convert PTS to seconds using the stream's time_base (MKV uses
            # millisecond resolution, so PTS=1000 ≈ 1s).
            assert first_pkt.pts is not None
            assert stream.time_base is not None
            pts_seconds = float(first_pkt.pts * stream.time_base)
            assert pts_seconds > 0.5, (
                f'Depth .part first PTS={first_pkt.pts} '
                f'(time_base={stream.time_base}, {pts_seconds:.3f}s) '
                f'is too close to 0. '
                f'Streams are not sharing a common timestamp origin.')
        finally:
            container.close()

        # Now do the merge for cleanup
        writer._merge_current_chunk()


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestChunkedWriterStopSafety:
    """Tests that stop() is safe and doesn't crash encoder/mux threads."""

    def test_stop_during_active_encoding(self, tmp_path: pathlib.Path):
        """Stopping while frames are actively being encoded should not crash."""
        writer, _ = _make_chunked_writer(tmp_path)
        writer.start()

        eq = writer.get_encoder_queue('rgb')
        # Feed a burst of frames
        for i in range(50):
            try:
                eq.put((_make_rgb_frame(), _ts(minute=10, second=i % 60)),
                       timeout=0.1)
            except queue.Full:
                break

        # Stop immediately while encoding is in progress
        writer.stop()

    def test_stop_during_rotation(self, tmp_path: pathlib.Path):
        """Stopping during an active rotation should not crash.

        This specifically tests the scenario where stop() advances the
        rotation generation while encoders are blocked in _participate_in_rotation.
        The encoder must not try to encode on an already-closed container.
        """
        base = _ts(year=2030, minute=10, second=0)
        writer, _ = _make_chunked_writer(
            tmp_path, stream_configs=_parallel_multi_stream_configs(),
            chunk_length_s=5)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')

        # Push frames right up to the boundary on RGB, but starve depth.
        # This means RGB will request rotation and wait for depth to participate,
        # but depth never will because its queue is empty. stop() must unblock.
        for i in range(5):
            ts = base + datetime.timedelta(seconds=i)
            rgb_q.put((_make_rgb_frame(), ts))
        rgb_q.put((_make_rgb_frame(), base + datetime.timedelta(seconds=5)))

        import time
        time.sleep(0.5)

        # stop() should unblock the stuck rotation and exit cleanly
        writer.stop()

    def test_stop_multi_stream_no_thread_exceptions(self, tmp_path: pathlib.Path):
        """Stopping a multi-stream writer should not produce thread exceptions."""
        import threading

        thread_exceptions: list[tuple[str, Exception]] = []
        original_excepthook = threading.excepthook

        def capture_thread_exception(args):
            thread_exceptions.append((args.thread.name, args.exc_value))

        threading.excepthook = capture_thread_exception
        try:
            base = _ts(year=2030, minute=10, second=0)
            writer, _ = _make_chunked_writer(
                tmp_path, stream_configs=_parallel_multi_stream_configs())
            writer.start()

            rgb_q = writer.get_encoder_queue('rgb')
            depth_q = writer.get_encoder_queue('depth')

            for i in range(20):
                ts = base + datetime.timedelta(seconds=i)
                rgb_q.put((_make_rgb_frame(), ts))
                depth_q.put((_make_depth_frame(), ts))

            import time
            time.sleep(0.5)
            writer.stop()

            # No thread should have raised an unhandled exception
            assert thread_exceptions == [], (
                f'Encoder/mux threads raised exceptions: {thread_exceptions}')
        finally:
            threading.excepthook = original_excepthook

    def test_stop_with_rotation_and_data_stream_no_crash(self, tmp_path: pathlib.Path):
        """Stop during rotation with video + data streams should not crash."""
        import threading

        base = _ts(year=2030, minute=10, second=0)
        thread_exceptions: list[tuple[str, Exception]] = []
        original_excepthook = threading.excepthook

        def capture_thread_exception(args):
            thread_exceptions.append((args.thread.name, args.exc_value))

        threading.excepthook = capture_thread_exception
        try:
            data_configs = {
                'imu': chunked_writer.DataStreamConfig(metadata={'type': 'imu'}),
            }
            writer, _ = _make_chunked_writer(
                tmp_path,
                stream_configs=_parallel_multi_stream_configs(),
                sensor_stream_configs=data_configs,
                chunk_length_s=5)
            writer.start()

            rgb_q = writer.get_encoder_queue('rgb')
            depth_q = writer.get_encoder_queue('depth')
            imu_q = writer.get_encoder_queue('imu')

            # Push frames crossing boundary
            for i in range(5):
                ts = base + datetime.timedelta(seconds=i)
                rgb_q.put((_make_rgb_frame(), ts))
                depth_q.put((_make_depth_frame(), ts))
                imu_q.put((b'{"ax":0.1}', ts))
            for i in range(3):
                ts = base + datetime.timedelta(seconds=5 + i)
                rgb_q.put((_make_rgb_frame(), ts))
                depth_q.put((_make_depth_frame(), ts))
                imu_q.put((b'{"ax":0.2}', ts))

            import time
            time.sleep(1.0)
            writer.stop()

            assert thread_exceptions == [], (
                f'Thread exceptions during stop: {thread_exceptions}')
        finally:
            threading.excepthook = original_excepthook


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestChunkedWriterMuxOrdering:
    """Tests for correct muxing when data streams produce packets before video.

    In production, IMU data at 100Hz reaches the mux thread before video
    packets (H.264 has encoder delay, buffering several frames before
    producing the first packet). If data stream packets are muxed before the
    MKV header is properly finalized with video codec extradata, the mux
    call fails with EINVAL.
    """

    def test_data_before_video_no_mux_errors(self, tmp_path: pathlib.Path, caplog):
        """Data stream packets arriving before video should not cause mux errors."""
        import time

        base = _ts(year=2030, minute=30, second=10)
        data_configs = {
            'imu': chunked_writer.DataStreamConfig(metadata={'type': 'imu'}),
        }
        writer, _ = _make_chunked_writer(
            tmp_path, sensor_stream_configs=data_configs)
        writer.start()

        imu_q = writer.get_encoder_queue('imu')
        rgb_q = writer.get_encoder_queue('rgb')

        # Feed IMU data first (simulates 100Hz sensor outpacing video encoder)
        for i in range(50):
            ts = base + datetime.timedelta(milliseconds=i * 10)
            imu_q.put((b'{"ax":0.1,"ay":0.2}', ts))

        # Small delay to let IMU packets reach the mux thread before video
        time.sleep(0.2)

        # Then feed video frames
        for i in range(10):
            ts = base + datetime.timedelta(milliseconds=i * 40)
            rgb_q.put((_make_rgb_frame(), ts))

        time.sleep(1.0)
        writer.stop()

        mux_errors = [r for r in caplog.records if 'Mux error' in r.message]
        assert mux_errors == [], (
            f'Got {len(mux_errors)} mux errors when data arrived before video: '
            f'{[r.message for r in mux_errors[:5]]}')

    def test_multi_video_plus_data_no_mux_errors(self, tmp_path: pathlib.Path, caplog):
        """Multiple video streams + data stream should not produce mux errors."""
        import time

        base = _ts(year=2030, minute=30, second=10)
        data_configs = {
            'imu': chunked_writer.DataStreamConfig(metadata={'type': 'imu'}),
        }
        writer, _ = _make_chunked_writer(
            tmp_path,
            stream_configs=_parallel_multi_stream_configs(),
            sensor_stream_configs=data_configs)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')
        imu_q = writer.get_encoder_queue('imu')

        # Feed IMU data rapidly first
        for i in range(100):
            ts = base + datetime.timedelta(milliseconds=i * 10)
            imu_q.put((b'{"ax":0.1}', ts))

        time.sleep(0.1)

        # Then feed video frames
        for i in range(10):
            ts = base + datetime.timedelta(milliseconds=i * 40)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(1.0)
        writer.stop()

        mux_errors = [r for r in caplog.records if 'Mux error' in r.message]
        assert mux_errors == [], (
            f'Got {len(mux_errors)} mux errors with multi-stream: '
            f'{[r.message for r in mux_errors[:5]]}')

    def test_depth_before_rgb_with_data_no_mux_errors(
            self, tmp_path: pathlib.Path, caplog):
        """Depth (ffv1) + IMU arriving before H.264 rgb should not cause mux errors."""
        import time

        base = _ts(year=2030, minute=30, second=10)
        data_configs = {
            'imu': chunked_writer.DataStreamConfig(metadata={'type': 'imu'}),
        }
        writer, _ = _make_chunked_writer(
            tmp_path,
            stream_configs=_parallel_multi_stream_configs(),
            sensor_stream_configs=data_configs)
        writer.start()

        depth_q = writer.get_encoder_queue('depth')
        imu_q = writer.get_encoder_queue('imu')
        rgb_q = writer.get_encoder_queue('rgb')

        # Feed depth and IMU first (both produce output instantly).
        for i in range(20):
            ts = base + datetime.timedelta(milliseconds=i * 40)
            depth_q.put((_make_depth_frame(), ts))
        for i in range(50):
            ts = base + datetime.timedelta(milliseconds=i * 10)
            imu_q.put((b'{"ax":0.1}', ts))

        # Delay before feeding rgb to give depth+IMU time to reach mux thread
        time.sleep(0.3)

        for i in range(20):
            ts = base + datetime.timedelta(milliseconds=i * 40)
            rgb_q.put((_make_rgb_frame(), ts))

        time.sleep(1.0)
        writer.stop()

        mux_errors = [r for r in caplog.records if 'Mux error' in r.message]
        assert mux_errors == [], (
            f'Got {len(mux_errors)} mux errors when depth arrived before rgb: '
            f'{[r.message for r in mux_errors[:5]]}')


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestChunkedWriterTimestampDomain:
    """Tests for camera timestamp domain mismatch.

    OAK-D timestamps are computed as:
        pipeline_start_reference_time + getTimestamp()
    where getTimestamp() returns timedelta since device boot. If the device
    was powered on before the pipeline started (e.g. at system boot, 50+
    seconds earlier), frame timestamps are offset forward from wall clock.

    The writer handles this because the first frame's timestamp (in whatever
    domain) sets the chunk origin. No wall-clock dependency.
    """

    def test_offset_timestamps_no_spurious_rotation(
            self, tmp_path: pathlib.Path):
        """Frame timestamps in a different time domain must not cause rotation.

        Spurious rotation on the first chunk must not happen.
        """
        import time

        # Use fixed timestamps far from wall clock. Because the chunk
        # origin is set from the first frame, all 10 frames (spanning
        # 9 seconds) fit within a 60s chunk.
        frame_base = _ts(year=2030, minute=15, second=10)

        writer, chunk_ids = _make_chunked_writer(
            tmp_path, stream_configs=_parallel_multi_stream_configs())
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')

        for i in range(10):
            ts = frame_base + datetime.timedelta(seconds=i)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(1.0)
        writer.stop()

        # Should be exactly 1 chunk (no spurious rotation)
        assert len(chunk_ids) == 1, (
            f'Expected 1 chunk but got {len(chunk_ids)}. '
            f'Spurious rotation from timestamp domain mismatch.')

    def test_offset_timestamps_with_imu_no_errors(
            self, tmp_path: pathlib.Path, caplog):
        """Offset timestamps with IMU data should produce no mux errors."""
        import time

        frame_base = _ts(year=2030, minute=15, second=10)

        data_configs = {
            'imu': chunked_writer.DataStreamConfig(metadata={'type': 'imu'}),
        }
        writer, chunk_ids = _make_chunked_writer(
            tmp_path,
            stream_configs=_parallel_multi_stream_configs(),
            sensor_stream_configs=data_configs)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')
        imu_q = writer.get_encoder_queue('imu')

        # IMU arrives first at 100Hz
        for i in range(50):
            ts = frame_base + datetime.timedelta(milliseconds=i * 10)
            imu_q.put((b'{"ax":0.1}', ts))

        time.sleep(0.1)

        # Then video frames
        for i in range(10):
            ts = frame_base + datetime.timedelta(milliseconds=i * 40)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(1.0)
        writer.stop()

        assert len(chunk_ids) == 1, (
            f'Expected 1 chunk but got {len(chunk_ids)}')

        mux_errors = [r for r in caplog.records if 'Mux error' in r.message]
        assert mux_errors == [], (
            f'Got {len(mux_errors)} mux errors with offset timestamps: '
            f'{[r.message for r in mux_errors[:5]]}')

    def test_rotation_after_offset_timestamps_no_errors(
            self, tmp_path: pathlib.Path, caplog):
        """Rotation should work correctly with offset timestamps."""
        import time

        # Start at an arbitrary offset time; chunk at 10s.
        frame_base = _ts(year=2030, minute=15, second=10)

        data_configs = {
            'imu': chunked_writer.DataStreamConfig(metadata={'type': 'imu'}),
        }
        writer, chunk_ids = _make_chunked_writer(
            tmp_path,
            stream_configs=_parallel_multi_stream_configs(),
            sensor_stream_configs=data_configs,
            chunk_length_s=10)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')
        imu_q = writer.get_encoder_queue('imu')

        # Frames in first chunk (0-9s from base)
        for i in range(5):
            ts = frame_base + datetime.timedelta(seconds=i * 2)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))
            imu_q.put((b'{"ax":0.1}', ts))

        # Frames in second chunk (10-19s from base)
        for i in range(5):
            ts = frame_base + datetime.timedelta(seconds=10 + i * 2)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))
            imu_q.put((b'{"ax":0.2}', ts))

        time.sleep(2.0)
        writer.stop()

        assert len(chunk_ids) == 2, (
            f'Expected 2 chunks (one rotation) but got {len(chunk_ids)}')

        mux_errors = [r for r in caplog.records if 'Mux error' in r.message]
        assert mux_errors == [], (
            f'Got {len(mux_errors)} mux errors after rotation with offset '
            f'timestamps: {[r.message for r in mux_errors[:5]]}')


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestEncoderCrash:

    def test_encoder_crash_stops_writer_cleanly(
            self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
        """If an encoder thread crashes, stop() should still complete."""
        base = _ts(year=2030, minute=10, second=0)
        writer, _ = _make_chunked_writer(
            tmp_path, stream_configs=_parallel_multi_stream_configs())
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')

        # Feed enough frames to create encoders
        for i in range(3):
            ts = base + datetime.timedelta(seconds=i)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(0.5)
        assert 'rgb' in writer._encoders, 'Encoder was not created in time'

        # Bomb the rgb encoder so it crashes on the next encode
        original_encode = writer._encoders['rgb'].encode
        call_count = [0]

        def bombing_encode(data, timestamp_s):
            call_count[0] += 1
            if call_count[0] > 1:
                raise RuntimeError('simulated encoder crash')
            return original_encode(data, timestamp_s)

        monkeypatch.setattr(writer._encoders['rgb'], 'encode', bombing_encode)

        # Feed more frames to trigger the crash
        for i in range(5):
            ts = base + datetime.timedelta(seconds=3 + i)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(1.0)

        # stop() should complete without hanging
        writer.stop()

    def test_encoder_crash_sets_stop_event(
            self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
        """An encoder crash should signal the stop event for other threads."""
        base = _ts(year=2030, minute=10, second=0)
        writer, _ = _make_chunked_writer(
            tmp_path, stream_configs=_parallel_multi_stream_configs())
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')

        for i in range(3):
            ts = base + datetime.timedelta(seconds=i)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(0.5)
        assert 'rgb' in writer._encoders

        def bombing_encode(data, timestamp_s):
            raise RuntimeError('simulated encoder crash')

        monkeypatch.setattr(writer._encoders['rgb'], 'encode', bombing_encode)

        # Trigger the crash
        ts = base + datetime.timedelta(seconds=10)
        rgb_q.put((_make_rgb_frame(), ts))

        time.sleep(1.0)

        assert writer._stop_event.is_set(), (
            'Stop event should be set after encoder crash')
        writer.stop()

    def test_encoder_crash_during_rotation_no_deadlock(
            self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
        """Encoder crash during rotation should not deadlock other threads."""
        base = _ts(year=2030, minute=10, second=0)
        writer, _ = _make_chunked_writer(
            tmp_path,
            stream_configs=_parallel_multi_stream_configs(),
            chunk_length_s=5)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')

        # Feed frames before the rotation boundary
        for i in range(3):
            ts = base + datetime.timedelta(seconds=i * 2)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(0.5)
        assert 'rgb' in writer._encoders

        # Bomb the rgb encoder's flush so it crashes during rotation
        def bombing_flush():
            raise RuntimeError('simulated flush crash during rotation')

        monkeypatch.setattr(writer._encoders['rgb'], 'flush', bombing_flush)

        # Feed frames past the boundary to trigger rotation
        for i in range(3):
            ts = base + datetime.timedelta(seconds=5 + i * 2)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(2.0)

        # stop() should complete without hanging (no deadlock)
        writer.stop()

    def test_encoder_crash_no_unhandled_thread_exceptions(
            self, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
        """Encoder crash should be caught, not leak as unhandled thread exception."""
        thread_exceptions: list[tuple[str, Exception]] = []

        def capture_thread_exception(args):
            thread_exceptions.append((args.thread.name, args.exc_value))

        monkeypatch.setattr(threading, 'excepthook', capture_thread_exception)

        base = _ts(year=2030, minute=10, second=0)
        writer, _ = _make_chunked_writer(
            tmp_path, stream_configs=_parallel_multi_stream_configs())
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')

        for i in range(3):
            ts = base + datetime.timedelta(seconds=i)
            rgb_q.put((_make_rgb_frame(), ts))
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(0.5)

        def bombing_encode(data, timestamp_s):
            raise RuntimeError('simulated encoder crash')

        monkeypatch.setattr(writer._encoders['rgb'], 'encode', bombing_encode)

        ts = base + datetime.timedelta(seconds=10)
        rgb_q.put((_make_rgb_frame(), ts))

        time.sleep(1.0)
        writer.stop()

        assert thread_exceptions == [], (
            f'Encoder crash leaked as unhandled exception: '
            f'{thread_exceptions}')


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestChunkedWriterSlowStreamRotation:
    """Tests for rotation when streams cross the boundary at different times.

    Reproduces the race where a fast stream crosses the chunk boundary
    while a slower stream still has items before the boundary. Without
    the per-stream boundary check, the slow stream enters rotation early,
    and its next item gets a negative PTS in the new chunk.
    """

    def test_slow_stream_catches_up_before_rotating(self, tmp_path: pathlib.Path):
        """Slow stream finishes pre-boundary items before entering rotation."""
        base = _ts(year=2030, minute=10, second=0)
        data_configs = {
            'imu': chunked_writer.DataStreamConfig(metadata={'type': 'imu'}),
        }
        writer, chunk_ids = _make_chunked_writer(
            tmp_path, sensor_stream_configs=data_configs, chunk_length_s=10)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        imu_q = writer.get_encoder_queue('imu')

        # RGB: 5 frames in chunk 1, then 3 past the boundary.
        for i in range(5):
            ts = base + datetime.timedelta(seconds=i * 2)
            rgb_q.put((_make_rgb_frame(), ts))
        for i in range(3):
            ts = base + datetime.timedelta(seconds=10 + i * 2)
            rgb_q.put((_make_rgb_frame(), ts))

        # IMU: items that lag behind — all before the boundary at first.
        for i in range(80):
            ts = base + datetime.timedelta(milliseconds=i * 100)
            imu_q.put((b'{"ax":0.1}', ts))

        # Then IMU items past the boundary.
        for i in range(30):
            ts = base + datetime.timedelta(seconds=10, milliseconds=i * 100)
            imu_q.put((b'{"ax":0.2}', ts))

        time.sleep(3.0)
        writer.stop()

        assert len(chunk_ids) == 2, (
            f'Expected 2 chunks but got {len(chunk_ids)}')

    def test_slow_stream_no_negative_pts_crash(self, tmp_path: pathlib.Path):
        """A slow stream behind the boundary must not crash with negative PTS."""
        thread_exceptions: list[tuple[str, Exception]] = []
        original_excepthook = threading.excepthook

        def capture_thread_exception(args):
            thread_exceptions.append((args.thread.name, args.exc_value))

        threading.excepthook = capture_thread_exception
        try:
            base = _ts(year=2030, minute=10, second=0)
            data_configs = {
                'imu': chunked_writer.DataStreamConfig(metadata={'type': 'imu'}),
            }
            writer, chunk_ids = _make_chunked_writer(
                tmp_path, sensor_stream_configs=data_configs, chunk_length_s=5)
            writer.start()

            rgb_q = writer.get_encoder_queue('rgb')
            imu_q = writer.get_encoder_queue('imu')

            # RGB crosses boundary immediately.
            for i in range(3):
                ts = base + datetime.timedelta(seconds=i * 2)
                rgb_q.put((_make_rgb_frame(), ts))
            for i in range(3):
                ts = base + datetime.timedelta(seconds=5 + i * 2)
                rgb_q.put((_make_rgb_frame(), ts))

            # IMU has many items BEFORE the boundary, then some after.
            for i in range(50):
                ts = base + datetime.timedelta(milliseconds=i * 100)
                imu_q.put((b'{"ax":0.1}', ts))
            for i in range(20):
                ts = base + datetime.timedelta(seconds=5, milliseconds=i * 100)
                imu_q.put((b'{"ax":0.2}', ts))

            time.sleep(3.0)
            writer.stop()

            assert len(chunk_ids) == 2
            assert thread_exceptions == [], (
                f'Thread exceptions (likely negative PTS): {thread_exceptions}')
        finally:
            threading.excepthook = original_excepthook

    def test_multi_video_slow_stream_rotation(self, tmp_path: pathlib.Path):
        """Two video streams where depth lags behind RGB at the boundary."""
        base = _ts(year=2030, minute=10, second=0)
        writer, chunk_ids = _make_chunked_writer(
            tmp_path, stream_configs=_parallel_multi_stream_configs(),
            chunk_length_s=10)
        writer.start()

        rgb_q = writer.get_encoder_queue('rgb')
        depth_q = writer.get_encoder_queue('depth')

        # RGB: crosses the boundary
        for i in range(5):
            ts = base + datetime.timedelta(seconds=i * 2)
            rgb_q.put((_make_rgb_frame(), ts))
        for i in range(5):
            ts = base + datetime.timedelta(seconds=10 + i * 2)
            rgb_q.put((_make_rgb_frame(), ts))

        # Depth: lags behind — items before boundary first, then past
        for i in range(5):
            ts = base + datetime.timedelta(seconds=i * 2)
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(0.5)

        for i in range(5):
            ts = base + datetime.timedelta(seconds=10 + i * 2)
            depth_q.put((_make_depth_frame(), ts))

        time.sleep(3.0)
        writer.stop()

        assert len(chunk_ids) == 2, (
            f'Expected 2 chunks but got {len(chunk_ids)}')
