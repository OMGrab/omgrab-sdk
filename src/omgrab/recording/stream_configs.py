"""Stream configuration dataclasses for video and data streams."""
from typing import Any

import dataclasses
import fractions


@dataclasses.dataclass
class VideoStreamConfig:
    """Configuration for a video stream.

    Attributes:
        width: Frame width in pixels.
        height: Frame height in pixels.
        fps: Target frames per second.
        codec: Video codec name (e.g. 'libx264', 'libx265').
        bitrate: Target bitrate in bits per second.
        input_pixel_format: Pixel format of incoming frames (e.g. 'rgb24').
        output_pixel_format: Pixel format for the encoded stream (e.g. 'yuv420p').
        stream_options: Additional codec options passed to the encoder.
        metadata: Arbitrary metadata stored in the MKV stream header.
    """

    width: int
    height: int
    fps: float | fractions.Fraction
    codec: str = 'libx264'
    bitrate: int = 2_000_000
    input_pixel_format: str = 'rgb24'
    output_pixel_format: str = 'yuv420p'
    stream_options: dict[str, str] = dataclasses.field(default_factory=dict)
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass
class DataStreamConfig:
    """Configuration for a non-video data stream stored as an MKV subtitle track.

    Attributes:
        codec: Subtitle codec name (e.g. 'ass').
        metadata: Arbitrary metadata stored in the MKV stream header.
    """

    codec: str = 'ass'
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)
