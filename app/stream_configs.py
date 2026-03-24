"""Stream configuration builders for RGB and depth cameras."""
from typing import Optional

from omgrab.cameras import cameras
from omgrab.recording import stream_configs


def rgb(config: cameras.CameraConfig,
        title: Optional[str] = None) -> stream_configs.VideoStreamConfig:
    """Build a VideoStreamConfig for an RGB camera."""
    metadata: dict = {'type': 'rgb'}
    if title is not None:
        metadata['TITLE'] = title
    return stream_configs.VideoStreamConfig(
        width=config.width,
        height=config.height,
        fps=config.fps,
        codec='libx264',
        bitrate=4_000_000,
        input_pixel_format='rgb24',
        output_pixel_format='yuv420p',
        stream_options={
            'preset': 'superfast',
            'profile': 'baseline',
            'bf': '0',
            'g': str(int(config.fps)),
        },
        metadata=metadata)


def depth(config: cameras.CameraConfig,
          title: Optional[str] = None) -> stream_configs.VideoStreamConfig:
    """Build a VideoStreamConfig for a depth camera."""
    metadata: dict = {'type': 'depth'}
    if title is not None:
        metadata['TITLE'] = title
    return stream_configs.VideoStreamConfig(
        width=config.width,
        height=config.height,
        fps=config.fps,
        codec='ffv1',
        input_pixel_format='gray16le',
        output_pixel_format='gray16le',
        metadata=metadata)
