from typing import Any
from typing import Optional
from typing import Protocol
from typing import runtime_checkable

import fractions
import json
import logging
import os
import pathlib
import subprocess

import av
import av.container
import numpy as np

from omgrab.cameras import cameras

logger = logging.getLogger(__name__)

TIME_BASE = fractions.Fraction(1, 1_000_000)


def merge_stream_files(
        part_paths: list[pathlib.Path],
        output_path: pathlib.Path):
    """Merge per-stream MKV part files into a single multi-stream MKV.

    Uses ffmpeg -c copy for a fast packet-level remux (no re-encoding).
    The merged file is written atomically via a .tmp rename + fsync.

    Args:
        part_paths: Paths to single-stream .mkv.part files to merge.
        output_path: Final output path (must end in .mkv).

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    if not part_paths:
        return

    tmp_path = output_path.with_suffix('.mkv.tmp')

    cmd = ['ffmpeg', '-y']
    for p in part_paths:
        cmd.extend(['-i', str(p)])
    cmd.extend(['-c', 'copy'])
    for i in range(len(part_paths)):
        cmd.extend(['-map', str(i)])
    cmd.extend(['-f', 'matroska'])
    cmd.append(str(tmp_path))

    logger.debug('Merging %d stream parts into %s', len(part_paths), output_path.name)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=60)
    if result.stderr:
        logger.debug('ffmpeg stderr: %s', result.stderr[-1000:])
    if result.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f'ffmpeg merge failed (exit {result.returncode}): '
            f'{result.stderr[-500:]}')

    tmp_path.rename(output_path)
    try:
        with open(output_path, 'rb+') as f:
            os.fsync(f)
    except FileNotFoundError:
        logger.warning('File %s not found for fsync', output_path)

    for p in part_paths:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def merge_recording_chunks(
        recording_dir: pathlib.Path,
        output_path: pathlib.Path):
    """Merge sequential MKV chunks from a recording directory into one file.

    Uses ffmpeg concat demuxer for a lossless stream copy. The merged file
    is written atomically via a .tmp rename + fsync, then the chunk
    directory is cleaned up.

    Args:
        recording_dir: Directory containing numbered .mkv chunk files.
        output_path: Final output path (e.g. /data/output/2026-03-17T00:47:42Z.mkv).

    Raises:
        RuntimeError: If ffmpeg fails.
    """
    chunks = sorted(recording_dir.glob('*.mkv'))
    if not chunks:
        logger.warning('No chunks found in %s, nothing to merge', recording_dir)
        return

    if len(chunks) == 1:
        # Single chunk: just move it directly (no concat needed).
        tmp_path = output_path.with_suffix('.mkv.tmp')
        chunks[0].rename(tmp_path)
        tmp_path.rename(output_path)
        try:
            with open(output_path, 'rb+') as f:
                os.fsync(f)
        except FileNotFoundError:
            logger.warning('File %s not found for fsync', output_path)
        _cleanup_recording_dir(recording_dir)
        return

    filelist_path = recording_dir / 'filelist.txt'
    with open(filelist_path, 'w') as f:
        for chunk in chunks:
            escaped = str(chunk).replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    tmp_path = output_path.with_suffix('.mkv.tmp')
    cmd = [
        'ffmpeg', '-y',
        '-f', 'concat', '-safe', '0',
        '-i', str(filelist_path),
        '-map', '0',
        '-c', 'copy',
        '-f', 'matroska',
        str(tmp_path),
    ]

    logger.info('Merging %d chunks into %s', len(chunks), output_path.name)
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300)
    if result.stderr:
        logger.debug('ffmpeg stderr: %s', result.stderr[-1000:])
    if result.returncode != 0:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(
            f'ffmpeg concat merge failed (exit {result.returncode}): '
            f'{result.stderr[-500:]}')

    tmp_path.rename(output_path)
    try:
        with open(output_path, 'rb+') as f:
            os.fsync(f)
    except FileNotFoundError:
        logger.warning('File %s not found for fsync', output_path)

    # Clean up chunks and recording directory.
    _cleanup_recording_dir(recording_dir)


def _cleanup_recording_dir(recording_dir: pathlib.Path):
    """Remove all files in a recording directory and the directory itself."""
    try:
        for f in recording_dir.iterdir():
            try:
                f.unlink()
            except FileNotFoundError:
                pass
        recording_dir.rmdir()
    except Exception as e:
        logger.warning('Failed to clean up recording dir %s: %s', recording_dir, e)


@runtime_checkable
class StreamEncoder(Protocol):
    """Common interface for all stream encoders (video and data)."""

    def encode(self, data: Any, timestamp_s: float) -> list[av.Packet]:
        """Encode data and return the resulting packets."""
        ...

    def flush(self) -> list[av.Packet]:
        """Flush any buffered data and return remaining packets."""
        ...


class VideoStreamEncoder:
    """Encodes video frames into packets without muxing.

    All PTS values are calculated relative to a shared timestamp origin,
    ensuring synchronization across streams within the same clip.
    """

    def __init__(
            self,
            container: av.container.OutputContainer,
            width: int,
            height: int,
            fps: float | fractions.Fraction,
            codec: str,
            bitrate: int,
            input_pixel_format: str,
            output_pixel_format: str,
            options: dict[str, str],
            timestamp_origin_s: float,
            metadata: Optional[dict[str, Any]] = None):
        """Initialize the stream encoder.

        Args:
            container: The PyAV container (used only to add the stream).
            width: Width of the video stream.
            height: Height of the video stream.
            fps: Nominal frames per second.
            codec: Codec name (e.g. 'libx264', 'ffv1').
            bitrate: Target bitrate.
            input_pixel_format: Input pixel format (e.g. 'rgb24', 'gray16le').
            output_pixel_format: Output pixel format (e.g. 'yuv420p', 'gray16le').
            options: Codec options.
            timestamp_origin_s: Shared timestamp origin in seconds since epoch.
                All PTS values are calculated relative to this origin.
            metadata: Optional metadata dictionary to add to the stream.
        """
        self._input_pixel_format = input_pixel_format
        self._timestamp_origin_s = timestamp_origin_s

        self._stream: av.VideoStream = container.add_stream(  # type: ignore[assignment,call-overload]
            codec_name=codec,
            rate=fractions.Fraction(fps),
            height=height,
            width=width,
            pix_fmt=output_pixel_format,
            bit_rate=bitrate,
            options=options,
        )
        # Use a fine-grained time_base for precise VFR timestamps.
        # Both the codec context and the stream (mux-side) must be set:
        # add_stream(rate=fps) defaults the codec time_base to 1/fps,
        # which quantizes PTS to frame boundaries and destroys sub-frame
        # precision. Setting codec_context.time_base forces the encoder
        # to preserve our microsecond-resolution PTS values.
        self._stream.codec_context.time_base = TIME_BASE
        self._stream.time_base = TIME_BASE

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    self._stream.metadata[key] = value
                else:
                    self._stream.metadata[key] = json.dumps(value)

    def encode(self, frame: cameras.Frame, timestamp_s: float) -> list[av.Packet]:
        """Encode a frame and return the resulting packets.

        Args:
            frame: Frame data to encode.
            timestamp_s: Timestamp in seconds since epoch.

        Returns:
            List of encoded packets (may be empty due to encoder buffering).
        """
        if frame.ndim == 2:
            frame = np.expand_dims(frame, axis=2)

        relative_time_s = timestamp_s - self._timestamp_origin_s
        assert self._stream.time_base is not None
        pts = max(0, int(relative_time_s * self._stream.time_base.denominator))

        video_frame = av.VideoFrame.from_ndarray(
            frame, format=self._input_pixel_format, channel_last=True)
        video_frame.pts = pts
        video_frame.time_base = self._stream.time_base

        return list(self._stream.encode(video_frame))

    def flush(self) -> list[av.Packet]:
        """Flush buffered frames from the encoder.

        Returns:
            List of remaining encoded packets.
        """
        return list(self._stream.encode(None))


class DataStreamEncoder:
    """Constructs data packets for a subtitle stream without muxing.

    All PTS values are calculated relative to a shared timestamp origin,
    ensuring synchronization across streams within the same clip.
    """

    def __init__(
            self,
            container: av.container.OutputContainer,
            timestamp_origin_s: float,
            codec: str = 'ass',
            metadata: Optional[dict[str, Any]] = None):
        """Initialize the data stream encoder.

        Args:
            container: The PyAV container (used only to add the stream).
            timestamp_origin_s: Shared timestamp origin in seconds since epoch.
            codec: Subtitle codec name.
            metadata: Optional metadata dictionary to add to the stream.
        """
        self._stream = container.add_stream(codec)
        self._stream.time_base = TIME_BASE
        self._timestamp_origin_s = timestamp_origin_s

        if metadata:
            for key, value in metadata.items():
                if isinstance(value, str):
                    self._stream.metadata[key] = value
                else:
                    self._stream.metadata[key] = json.dumps(value)

    def encode(self, data: bytes, timestamp_s: float) -> list[av.Packet]:
        """Construct a data packet.

        Args:
            data: Raw bytes to store.
            timestamp_s: Timestamp in seconds since epoch.

        Returns:
            A single-element list containing the packet.
        """
        relative_time_s = timestamp_s - self._timestamp_origin_s
        assert self._stream.time_base is not None
        pts = max(0, int(relative_time_s * self._stream.time_base.denominator))

        packet = av.Packet(data)
        packet.stream = self._stream
        packet.pts = pts
        packet.dts = pts
        return [packet]

    def flush(self) -> list[av.Packet]:
        """No-op: raw packet muxing has no encoder to flush."""
        return []
