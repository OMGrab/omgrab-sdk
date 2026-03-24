"""Tests for py_av_writer.py — VideoStreamEncoder, DataStreamEncoder, merge functions."""
import fractions
import json
import pathlib
import shutil

import av
import numpy as np
import pytest

from omgrab.recording import py_av_writer

_has_ffmpeg = shutil.which('ffmpeg') is not None

# PyAV requires rate to be int or Fraction, not float.
FPS = fractions.Fraction(25)



def _make_rgb_frame(width: int = 64, height: int = 64, value: int = 128) -> np.ndarray:
    """Create a solid-color RGB frame."""
    return np.full((height, width, 3), fill_value=value, dtype=np.uint8)


def _make_gray16_frame(width: int = 64, height: int = 64, value: int = 1000) -> np.ndarray:
    """Create a solid-value 16-bit grayscale frame (like a depth map)."""
    return np.full((height, width), fill_value=value, dtype=np.uint16)


def _read_video_stream_metadata(mkv_path: pathlib.Path, stream_index: int = 0) -> dict[str, str]:
    """Read metadata from a video stream."""
    container = av.open(str(mkv_path), mode='r')
    try:
        return dict(container.streams.video[stream_index].metadata)
    finally:
        container.close()


def _count_video_frames(mkv_path: pathlib.Path, stream_index: int = 0) -> int:
    """Count the number of decoded video frames in an MKV file."""
    container = av.open(str(mkv_path), mode='r')
    try:
        stream = container.streams.video[stream_index]
        count = 0
        for _ in container.decode(stream):
            count += 1
        return count
    finally:
        container.close()


def _make_encoder(container, **overrides):
    """Create a StreamEncoder with sensible defaults."""
    defaults = dict(
        container=container,
        width=64,
        height=64,
        fps=FPS,
        codec='libx264',
        bitrate=2_000_000,
        input_pixel_format='rgb24',
        output_pixel_format='yuv420p',
        options={'preset': 'ultrafast', 'bf': '0'},
        timestamp_origin_s=1000.0,
    )
    defaults.update(overrides)
    return py_av_writer.VideoStreamEncoder(**defaults)


def _encode_and_close(output_path, encoder_kwargs, frames_and_timestamps):
    """Create container, encode frames, close, return path."""
    container = av.open(str(output_path), mode='w', format='matroska')
    encoder = _make_encoder(container, **encoder_kwargs)
    for frame, ts in frames_and_timestamps:
        for pkt in encoder.encode(frame, ts):
            container.mux(pkt)
    for pkt in encoder.flush():
        container.mux(pkt)
    container.close()
    return output_path



class TestStreamEncoderEncoding:

    def test_writes_expected_number_of_frames(self, tmp_path: pathlib.Path):
        """All encoded frames should be decodable from the output."""
        output = tmp_path / 'output.mkv'
        frames = [(_make_rgb_frame(value=i * 25), 1000.0 + i * 0.04) for i in range(10)]
        _encode_and_close(output, {}, frames)

        assert _count_video_frames(output) == 10

    def test_vfr_timestamps_are_relative_to_origin(self, tmp_path: pathlib.Path):
        """Frame PTS should be relative to timestamp_origin_s.

        Tolerance is 1ms (MKV uses millisecond mux time_base).
        """
        output = tmp_path / 'output.mkv'
        # origin=1000.0, frames at 1000.0, 1000.5, 1001.0
        frames = [
            (_make_rgb_frame(), 1000.0),
            (_make_rgb_frame(), 1000.5),
            (_make_rgb_frame(), 1001.0),
        ]
        _encode_and_close(output, {}, frames)

        container = av.open(str(output), mode='r')
        try:
            time_s_values = []
            for frame in container.decode(video=0):
                assert frame.pts is not None
                assert frame.time_base is not None
                time_s_values.append(float(frame.pts * frame.time_base))
        finally:
            container.close()

        assert len(time_s_values) == 3
        assert time_s_values[0] == pytest.approx(0.0, abs=0.001)
        assert time_s_values[1] == pytest.approx(0.5, abs=0.001)
        assert time_s_values[2] == pytest.approx(1.0, abs=0.001)

    def test_sub_frame_timestamp_precision(self, tmp_path: pathlib.Path):
        """Timestamps not aligned to frame boundaries should be preserved.

        At 25fps, frame boundaries are every 40ms. This test writes frames
        at 33ms intervals (not a multiple of 40ms) and verifies the encoder
        does not quantize them to the nearest frame boundary.
        """
        output = tmp_path / 'output.mkv'
        expected_offsets = [0.0, 0.033, 0.066, 0.099, 0.5, 0.503]
        frames = [(_make_rgb_frame(), 1000.0 + offset) for offset in expected_offsets]
        _encode_and_close(output, {}, frames)

        container = av.open(str(output), mode='r')
        try:
            time_s_values = []
            for frame in container.decode(video=0):
                assert frame.pts is not None
                assert frame.time_base is not None
                time_s_values.append(float(frame.pts * frame.time_base))
        finally:
            container.close()

        assert len(time_s_values) == len(expected_offsets)
        for actual, expected in zip(time_s_values, expected_offsets, strict=True):
            assert actual == pytest.approx(expected, abs=0.001)

    def test_grayscale_2d_frame_expanded(self, tmp_path: pathlib.Path):
        """A 2D (H, W) frame should be automatically expanded to (H, W, 1)."""
        output = tmp_path / 'output.mkv'
        container = av.open(str(output), mode='w', format='matroska')
        encoder = _make_encoder(
            container, codec='ffv1',
            input_pixel_format='gray16le',
            output_pixel_format='gray16le',
            options={})

        frame_2d = _make_gray16_frame()
        assert frame_2d.ndim == 2

        for pkt in encoder.encode(frame_2d, 1000.0):
            container.mux(pkt)
        for pkt in encoder.flush():
            container.mux(pkt)
        container.close()

        assert _count_video_frames(output) == 1

    def test_stream_metadata_preserved(self, tmp_path: pathlib.Path):
        """Video stream metadata should survive write -> read round trip."""
        output = tmp_path / 'output.mkv'
        container = av.open(str(output), mode='w', format='matroska')
        encoder = _make_encoder(
            container,
            metadata={'type': 'rgb', 'custom_key': 'custom_value'})

        for pkt in encoder.encode(_make_rgb_frame(), 1000.0):
            container.mux(pkt)
        for pkt in encoder.flush():
            container.mux(pkt)
        container.close()

        metadata = _read_video_stream_metadata(output)
        assert metadata.get('TYPE') == 'rgb'
        assert metadata.get('CUSTOM_KEY') == 'custom_value'

    def test_dict_metadata_serialized_as_json(self, tmp_path: pathlib.Path):
        """Non-string metadata values should be JSON-encoded."""
        output = tmp_path / 'output.mkv'
        container = av.open(str(output), mode='w', format='matroska')
        encoder = _make_encoder(
            container,
            metadata={'type': 'rgb', 'resolution': [64, 64]})

        for pkt in encoder.encode(_make_rgb_frame(), 1000.0):
            container.mux(pkt)
        for pkt in encoder.flush():
            container.mux(pkt)
        container.close()

        metadata = _read_video_stream_metadata(output)
        assert json.loads(metadata.get('RESOLUTION', '')) == [64, 64]



class TestPTSClamping:

    def test_video_encoder_clamps_negative_pts_to_zero(self, tmp_path: pathlib.Path):
        """VideoStreamEncoder should clamp PTS to 0 for tiny negative relative time."""
        origin = 1774604451.123456 + 60.0
        timestamp_s = origin - 1e-10

        output = tmp_path / 'clamp.mkv'
        container = av.open(str(output), mode='w', format='matroska')
        encoder = _make_encoder(container, timestamp_origin_s=origin)

        pkts = encoder.encode(_make_rgb_frame(), timestamp_s)
        for pkt in pkts:
            assert pkt.pts >= 0, f'Video PTS should be >= 0, got {pkt.pts}'
            container.mux(pkt)

        for pkt in encoder.flush():
            container.mux(pkt)
        container.close()


class TestTimeBase:

    def test_time_base_is_microsecond(self, tmp_path: pathlib.Path):
        """The stream time_base should be 1/1_000_000 for microsecond precision."""
        output = tmp_path / 'output.mkv'
        container = av.open(str(output), mode='w', format='matroska')
        encoder = _make_encoder(container)

        assert encoder._stream.time_base == py_av_writer.TIME_BASE

        for pkt in encoder.encode(_make_rgb_frame(), 1000.0):
            container.mux(pkt)
        for pkt in encoder.flush():
            container.mux(pkt)
        container.close()



class TestDataStreamEncoder:

    def test_flush_is_noop(self, tmp_path: pathlib.Path):
        """flush() should return an empty list."""
        output = tmp_path / 'output.mkv'
        container = av.open(str(output), mode='w', format='matroska')
        # Need a video stream for valid MKV.
        video = _make_encoder(container)
        data = py_av_writer.DataStreamEncoder(
            container=container, timestamp_origin_s=1000.0, codec='ass')

        for pkt in video.encode(_make_rgb_frame(), 1000.0):
            container.mux(pkt)
        for pkt in data.encode(b'payload', 1000.0):
            container.mux(pkt)

        assert data.flush() == []

        for pkt in video.flush():
            container.mux(pkt)
        container.close()

    def test_data_stream_metadata_preserved(self, tmp_path: pathlib.Path):
        """Data stream metadata should survive round-trip."""
        output = tmp_path / 'output.mkv'
        container = av.open(str(output), mode='w', format='matroska')
        video = _make_encoder(container)
        data = py_av_writer.DataStreamEncoder(
            container=container,
            timestamp_origin_s=1000.0,
            codec='ass',
            metadata={'type': 'sensor', 'rate_hz': 100})

        for pkt in video.encode(_make_rgb_frame(), 1000.0):
            container.mux(pkt)
        for pkt in data.encode(b'packet', 1000.0):
            container.mux(pkt)
        for pkt in video.flush():
            container.mux(pkt)
        container.close()

        read_container = av.open(str(output), mode='r')
        try:
            sub_meta = dict(read_container.streams.subtitles[0].metadata)
            assert sub_meta.get('TYPE') == 'sensor'
            assert sub_meta.get('RATE_HZ') == '100'
        finally:
            read_container.close()

    def test_pts_relative_to_origin(self, tmp_path: pathlib.Path):
        """PTS should be relative to timestamp_origin_s, not first write."""
        output = tmp_path / 'output.mkv'
        container = av.open(str(output), mode='w', format='matroska')
        video = _make_encoder(container)
        data = py_av_writer.DataStreamEncoder(
            container=container,
            timestamp_origin_s=1000.0,
            codec='ass')

        # Write data at 1001.0 (1s after origin)
        pkts = data.encode(b'hello', 1001.0)
        assert pkts[0].pts == 1_000_000  # 1s in microseconds

        for pkt in video.encode(_make_rgb_frame(), 1000.0):
            container.mux(pkt)
        for pkt in data.encode(b'hello', 1001.0):
            container.mux(pkt)
        for pkt in video.flush():
            container.mux(pkt)
        container.close()

    def test_slightly_negative_relative_time_clamped_to_zero(self):
        """PTS must be clamped to 0 when floating point gives tiny negative."""
        # Simulate the chunk rotation edge case: origin is computed as
        # old_origin + chunk_length, but due to IEEE 754 rounding,
        # timestamp - new_origin can be slightly negative.
        origin = 1774604451.123456 + 60.0  # new chunk origin after rotation
        # Construct a timestamp that passes the boundary check
        # (timestamp - old_origin >= 60.0) but produces negative
        # relative_time when subtracted from new_origin.
        timestamp_s = origin - 1e-10  # tiny negative relative time

        import io
        buf = io.BytesIO()
        container = av.open(buf, mode='w', format='matroska')
        data = py_av_writer.DataStreamEncoder(
            container=container,
            timestamp_origin_s=origin,
            codec='ass')

        pkts = data.encode(b'{"ax":0.1}', timestamp_s)
        assert pkts[0].pts == 0, (
            f'Expected PTS=0 for slightly-before-origin timestamp, got {pkts[0].pts}')
        container.close()



class TestMultiStreamEncoding:

    def test_rgb_plus_depth_streams(self, tmp_path: pathlib.Path):
        """Two video streams (RGB + depth) should both be decodable."""
        output = tmp_path / 'output.mkv'
        container = av.open(str(output), mode='w', format='matroska')

        rgb = _make_encoder(container, metadata={'type': 'rgb'})
        depth = _make_encoder(
            container, codec='ffv1',
            input_pixel_format='gray16le',
            output_pixel_format='gray16le',
            options={},
            metadata={'type': 'depth'})

        num_frames = 5
        for i in range(num_frames):
            ts = 1000.0 + i * 0.04
            for pkt in rgb.encode(_make_rgb_frame(value=i * 50), ts):
                container.mux(pkt)
            for pkt in depth.encode(_make_gray16_frame(value=i * 100), ts):
                container.mux(pkt)
        for pkt in rgb.flush():
            container.mux(pkt)
        for pkt in depth.flush():
            container.mux(pkt)
        container.close()

        read_container = av.open(str(output), mode='r')
        try:
            assert len(read_container.streams.video) == 2
        finally:
            read_container.close()

        assert _count_video_frames(output, stream_index=0) == num_frames
        assert _count_video_frames(output, stream_index=1) == num_frames

    def test_rgb_plus_depth_plus_data_stream(self, tmp_path: pathlib.Path):
        """Container with video + depth + subtitle data should be readable."""
        output = tmp_path / 'output.mkv'
        container = av.open(str(output), mode='w', format='matroska')

        rgb = _make_encoder(container, metadata={'type': 'rgb'})
        depth = _make_encoder(
            container, codec='ffv1',
            input_pixel_format='gray16le',
            output_pixel_format='gray16le',
            options={},
            metadata={'type': 'depth'})
        data = py_av_writer.DataStreamEncoder(
            container=container,
            timestamp_origin_s=1000.0,
            codec='ass',
            metadata={'type': 'imu'})

        ts = 1000.0
        for pkt in rgb.encode(_make_rgb_frame(), ts):
            container.mux(pkt)
        for pkt in depth.encode(_make_gray16_frame(), ts):
            container.mux(pkt)
        for pkt in data.encode(json.dumps({'ax': 0.1}).encode(), ts):
            container.mux(pkt)

        for pkt in rgb.flush():
            container.mux(pkt)
        for pkt in depth.flush():
            container.mux(pkt)
        container.close()

        read_container = av.open(str(output), mode='r')
        try:
            assert len(read_container.streams.video) == 2
            assert len(read_container.streams.subtitles) == 1
        finally:
            read_container.close()


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestMergeStreamFiles:
    """Tests for merge_stream_files (ffmpeg -c copy remux)."""

    def _write_h264_part(self, path: pathlib.Path, n_frames: int = 5):
        """Write an H.264 single-stream MKV part file."""
        container = av.open(str(path), mode='w', format='matroska')
        stream = container.add_stream(
            'libx264', rate=25, height=64, width=64,
            pix_fmt='yuv420p',
            options={'preset': 'ultrafast', 'bf': '0'})
        stream.codec_context.time_base = py_av_writer.TIME_BASE
        stream.time_base = py_av_writer.TIME_BASE
        stream.metadata['type'] = 'rgb'

        for i in range(n_frames):
            f = av.VideoFrame.from_ndarray(
                _make_rgb_frame(value=i * 40), format='rgb24')
            f.pts = i * 40000
            f.time_base = py_av_writer.TIME_BASE
            for pkt in stream.encode(f):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

    def _write_ffv1_part(self, path: pathlib.Path, n_frames: int = 5):
        """Write an ffv1 single-stream MKV part file."""
        container = av.open(str(path), mode='w', format='matroska')
        stream = container.add_stream(
            'ffv1', rate=25, height=64, width=64, pix_fmt='gray16le')
        assert isinstance(stream, av.VideoStream)
        stream.codec_context.time_base = py_av_writer.TIME_BASE
        stream.time_base = py_av_writer.TIME_BASE
        stream.metadata['type'] = 'depth'

        for i in range(n_frames):
            f = av.VideoFrame.from_ndarray(
                _make_gray16_frame(value=i * 100), format='gray16le')
            f.pts = i * 40000
            f.time_base = py_av_writer.TIME_BASE
            for pkt in stream.encode(f):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

    def _write_subtitle_part(self, path: pathlib.Path, n_packets: int = 10):
        """Write an ASS subtitle single-stream MKV part file."""
        container = av.open(str(path), mode='w', format='matroska')
        stream = container.add_stream('ass')
        stream.time_base = py_av_writer.TIME_BASE
        stream.metadata['type'] = 'imu'

        for i in range(n_packets):
            data = json.dumps({'ax': 0.1 * i}).encode()
            pkt = av.Packet(data)
            pkt.stream = stream
            pkt.pts = i * 10000
            pkt.dts = i * 10000
            container.mux(pkt)
        container.close()

    def test_merge_two_video_streams(self, tmp_path: pathlib.Path):
        """Merging H.264 + ffv1 should produce a multi-stream MKV."""
        rgb_part = tmp_path / 'clip.rgb.mkv.part'
        depth_part = tmp_path / 'clip.depth.mkv.part'
        self._write_h264_part(rgb_part)
        self._write_ffv1_part(depth_part)

        output = tmp_path / 'clip.mkv'
        py_av_writer.merge_stream_files([rgb_part, depth_part], output)

        assert output.exists()
        container = av.open(str(output), mode='r')
        try:
            assert len(container.streams.video) == 2
        finally:
            container.close()

        assert _count_video_frames(output, stream_index=0) == 5
        assert _count_video_frames(output, stream_index=1) == 5

    def test_merge_video_plus_subtitle(self, tmp_path: pathlib.Path):
        """Merging video + subtitle should produce a mixed-stream MKV."""
        rgb_part = tmp_path / 'clip.rgb.mkv.part'
        imu_part = tmp_path / 'clip.imu.mkv.part'
        self._write_h264_part(rgb_part)
        self._write_subtitle_part(imu_part)

        output = tmp_path / 'clip.mkv'
        py_av_writer.merge_stream_files([rgb_part, imu_part], output)

        container = av.open(str(output), mode='r')
        try:
            assert len(container.streams.video) == 1
            assert len(container.streams.subtitles) == 1
        finally:
            container.close()

    def test_merge_preserves_metadata(self, tmp_path: pathlib.Path):
        """Stream metadata should survive the ffmpeg remux."""
        rgb_part = tmp_path / 'clip.rgb.mkv.part'
        depth_part = tmp_path / 'clip.depth.mkv.part'
        self._write_h264_part(rgb_part)
        self._write_ffv1_part(depth_part)

        output = tmp_path / 'clip.mkv'
        py_av_writer.merge_stream_files([rgb_part, depth_part], output)

        container = av.open(str(output), mode='r')
        try:
            assert container.streams.video[0].metadata.get('TYPE') == 'rgb'
            assert container.streams.video[1].metadata.get('TYPE') == 'depth'
        finally:
            container.close()

    def test_merge_deletes_part_files(self, tmp_path: pathlib.Path):
        """Part files should be deleted after successful merge."""
        rgb_part = tmp_path / 'clip.rgb.mkv.part'
        depth_part = tmp_path / 'clip.depth.mkv.part'
        self._write_h264_part(rgb_part)
        self._write_ffv1_part(depth_part)

        output = tmp_path / 'clip.mkv'
        py_av_writer.merge_stream_files([rgb_part, depth_part], output)

        assert not rgb_part.exists()
        assert not depth_part.exists()

    def test_merge_empty_list_is_noop(self, tmp_path: pathlib.Path):
        """Merging an empty list should do nothing."""
        output = tmp_path / 'clip.mkv'
        py_av_writer.merge_stream_files([], output)
        assert not output.exists()

    def test_merge_single_stream(self, tmp_path: pathlib.Path):
        """Merging a single stream should still produce a valid MKV."""
        rgb_part = tmp_path / 'clip.rgb.mkv.part'
        self._write_h264_part(rgb_part, n_frames=3)

        output = tmp_path / 'clip.mkv'
        py_av_writer.merge_stream_files([rgb_part], output)

        assert output.exists()
        assert _count_video_frames(output, stream_index=0) == 3

    def test_merge_three_streams(self, tmp_path: pathlib.Path):
        """Merging H.264 + ffv1 + subtitle should produce 3-stream MKV."""
        rgb_part = tmp_path / 'clip.rgb.mkv.part'
        depth_part = tmp_path / 'clip.depth.mkv.part'
        imu_part = tmp_path / 'clip.imu.mkv.part'
        self._write_h264_part(rgb_part)
        self._write_ffv1_part(depth_part)
        self._write_subtitle_part(imu_part)

        output = tmp_path / 'clip.mkv'
        py_av_writer.merge_stream_files(
            [rgb_part, depth_part, imu_part], output)

        container = av.open(str(output), mode='r')
        try:
            assert len(container.streams.video) == 2
            assert len(container.streams.subtitles) == 1
        finally:
            container.close()


@pytest.mark.skipif(not _has_ffmpeg, reason='ffmpeg not installed')
class TestMergeRecordingChunks:
    """Tests for merge_recording_chunks (ffmpeg concat of multi-stream chunks)."""

    def _write_h264_part(self, path: pathlib.Path, n_frames: int = 5,
                         pts_offset: int = 0):
        container = av.open(str(path), mode='w', format='matroska')
        stream = container.add_stream(
            'libx264', rate=25, height=64, width=64,
            pix_fmt='yuv420p',
            options={'preset': 'ultrafast', 'bf': '0'})
        stream.codec_context.time_base = py_av_writer.TIME_BASE
        stream.time_base = py_av_writer.TIME_BASE
        stream.metadata['type'] = 'rgb'
        for i in range(n_frames):
            f = av.VideoFrame.from_ndarray(
                _make_rgb_frame(value=(i * 40) % 256), format='rgb24')
            f.pts = pts_offset + i * 40000
            f.time_base = py_av_writer.TIME_BASE
            for pkt in stream.encode(f):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

    def _write_ffv1_part(self, path: pathlib.Path, n_frames: int = 5,
                         pts_offset: int = 0):
        container = av.open(str(path), mode='w', format='matroska')
        stream = container.add_stream(
            'ffv1', rate=25, height=64, width=64, pix_fmt='gray16le')
        assert isinstance(stream, av.VideoStream)
        stream.codec_context.time_base = py_av_writer.TIME_BASE
        stream.time_base = py_av_writer.TIME_BASE
        stream.metadata['type'] = 'depth'
        for i in range(n_frames):
            f = av.VideoFrame.from_ndarray(
                _make_gray16_frame(value=i * 100), format='gray16le')
            f.pts = pts_offset + i * 40000
            f.time_base = py_av_writer.TIME_BASE
            for pkt in stream.encode(f):
                container.mux(pkt)
        for pkt in stream.encode(None):
            container.mux(pkt)
        container.close()

    def _write_subtitle_part(self, path: pathlib.Path, n_packets: int = 10,
                             pts_offset: int = 0):
        container = av.open(str(path), mode='w', format='matroska')
        stream = container.add_stream('ass')
        stream.time_base = py_av_writer.TIME_BASE
        stream.metadata['type'] = 'imu'
        for i in range(n_packets):
            data = json.dumps({'ax': 0.1 * i}).encode()
            pkt = av.Packet(data)
            pkt.stream = stream
            pkt.pts = pts_offset + i * 10000
            pkt.dts = pts_offset + i * 10000
            container.mux(pkt)
        container.close()

    def _make_multi_stream_chunk(self, path: pathlib.Path, n_frames: int = 5,
                                 pts_offset: int = 0):
        """Create a multi-stream MKV chunk (h264 + ffv1 + subtitle)."""
        rgb_part = path.with_suffix('.rgb.mkv.part')
        depth_part = path.with_suffix('.depth.mkv.part')
        imu_part = path.with_suffix('.imu.mkv.part')
        self._write_h264_part(rgb_part, n_frames, pts_offset)
        self._write_ffv1_part(depth_part, n_frames, pts_offset)
        self._write_subtitle_part(imu_part, n_frames * 2, pts_offset)
        py_av_writer.merge_stream_files(
            [rgb_part, depth_part, imu_part], path)

    def test_concat_preserves_all_streams(self, tmp_path: pathlib.Path):
        """Concatenating multi-stream chunks must keep h264 + ffv1 + subtitle."""
        rec_dir = tmp_path / 'recording'
        rec_dir.mkdir()
        self._make_multi_stream_chunk(rec_dir / '00001.mkv', n_frames=5, pts_offset=0)
        self._make_multi_stream_chunk(rec_dir / '00002.mkv', n_frames=5, pts_offset=200000)

        output = tmp_path / 'output.mkv'
        py_av_writer.merge_recording_chunks(rec_dir, output)

        assert output.exists()
        container = av.open(str(output), mode='r')
        try:
            assert len(container.streams.video) == 2, (
                f'Expected 2 video streams (h264 + ffv1), got {len(container.streams.video)}')
            assert len(container.streams.subtitles) == 1, (
                f'Expected 1 subtitle stream, got {len(container.streams.subtitles)}')
            assert container.streams.video[0].codec_context.name == 'h264'
            assert container.streams.video[1].codec_context.name == 'ffv1'
        finally:
            container.close()

    def test_concat_preserves_frame_count(self, tmp_path: pathlib.Path):
        """Total frame count after concat should equal sum of chunks."""
        rec_dir = tmp_path / 'recording'
        rec_dir.mkdir()
        self._make_multi_stream_chunk(rec_dir / '00001.mkv', n_frames=5, pts_offset=0)
        self._make_multi_stream_chunk(rec_dir / '00002.mkv', n_frames=3, pts_offset=200000)

        output = tmp_path / 'output.mkv'
        py_av_writer.merge_recording_chunks(rec_dir, output)

        assert _count_video_frames(output, stream_index=0) == 8
        assert _count_video_frames(output, stream_index=1) == 8

    def test_concat_preserves_metadata(self, tmp_path: pathlib.Path):
        """Stream metadata should survive concat merge."""
        rec_dir = tmp_path / 'recording'
        rec_dir.mkdir()
        self._make_multi_stream_chunk(rec_dir / '00001.mkv', n_frames=3, pts_offset=0)
        self._make_multi_stream_chunk(rec_dir / '00002.mkv', n_frames=3, pts_offset=120000)

        output = tmp_path / 'output.mkv'
        py_av_writer.merge_recording_chunks(rec_dir, output)

        container = av.open(str(output), mode='r')
        try:
            assert container.streams.video[0].metadata.get('TYPE') == 'rgb'
            assert container.streams.video[1].metadata.get('TYPE') == 'depth'
            assert container.streams.subtitles[0].metadata.get('TYPE') == 'imu'
        finally:
            container.close()

    def test_concat_cleans_up_recording_dir(self, tmp_path: pathlib.Path):
        """Recording directory should be removed after successful concat."""
        rec_dir = tmp_path / 'recording'
        rec_dir.mkdir()
        self._make_multi_stream_chunk(rec_dir / '00001.mkv', n_frames=3, pts_offset=0)
        self._make_multi_stream_chunk(rec_dir / '00002.mkv', n_frames=3, pts_offset=120000)

        output = tmp_path / 'output.mkv'
        py_av_writer.merge_recording_chunks(rec_dir, output)

        assert not rec_dir.exists()

    def test_single_chunk_preserves_all_streams(self, tmp_path: pathlib.Path):
        """Single-chunk shortcut (rename) must also preserve all streams."""
        rec_dir = tmp_path / 'recording'
        rec_dir.mkdir()
        self._make_multi_stream_chunk(rec_dir / '00001.mkv', n_frames=5)

        output = tmp_path / 'output.mkv'
        py_av_writer.merge_recording_chunks(rec_dir, output)

        container = av.open(str(output), mode='r')
        try:
            assert len(container.streams.video) == 2
            assert len(container.streams.subtitles) == 1
        finally:
            container.close()
