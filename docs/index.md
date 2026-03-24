# OMGrab SDK

SDK for egocentric data collection with OMGrab hardware.

## Installation

```bash
pip install omgrab
```

## Overview

The omgrab SDK provides the core runtime for egocentric data collection on the OMGrab streaming device. The SDK is used to build the core runtime application, which supports:

- User activated recording start/stop on physical button press
- Minute-chunked clip recording of various stream types to disk for crash safety
- Automatic recording chunk merging into a standardized MKV container on stop
- OLED status display with camera preview mode
- WiFi hotspot for authentication
- Automatic network and device health monitoring
- Network file share of completed recordings using [Samba](https://www.samba.org/)

### Key components

- **devices** -- hardware capture device abstraction (`CaptureDevice` protocol)
- **cameras** -- frame sources for RGB, depth, and USB cameras
- **sensors** -- non-video data sources (e.g., IMU)
- **recording** -- chunked MKV writing with multi-stream support
- **runtime** -- recording lifecycle, network monitoring, device status
- **workflows** -- state machine coordinating the full device lifecycle
- **gpio** -- LED, buzzer, and button control
- **display** -- OLED status dashboard and live camera preview

## Quick start

This example initializes an OAK-D camera, captures 10 seconds of synchronized
RGB + depth video, and writes the result to a single MKV file on disk.

```python
import pathlib

from omgrab.cameras import CameraConfig
from omgrab.devices import OakDCaptureDevice
from omgrab.recording import ChunkedWriter, VideoStreamConfig

# 1. Configure cameras and streams
rgb_config = CameraConfig(fps=25, width=1280, height=800)
depth_config = CameraConfig(fps=25, width=640, height=400)

output_dir = pathlib.Path('./my_recording')
output_dir.mkdir(exist_ok=True)

video_streams = {
    'rgb': VideoStreamConfig(
        width=rgb_config.width,
        height=rgb_config.height,
        fps=rgb_config.fps,
        codec='libx264',
        bitrate=4_000_000,
        input_pixel_format='rgb24',
        output_pixel_format='yuv420p',
        metadata={'type': 'rgb'},
    ),
    'depth': VideoStreamConfig(
        width=depth_config.width,
        height=depth_config.height,
        fps=depth_config.fps,
        codec='ffv1',
        input_pixel_format='gray16le',
        output_pixel_format='gray16le',
        metadata={'type': 'depth'},
    ),
}

# 2. Initialize the capture device
device = OakDCaptureDevice(rgb_config, depth_config)

# 3. Initialize the writer and get encoder queues
writer = ChunkedWriter(
    name='my_recording',
    output_directory=output_dir,
    stream_configs=video_streams,
    chunk_length_s=60.0,
)
rgb_queue = writer.get_encoder_queue('rgb')
depth_queue = writer.get_encoder_queue('depth')

# 4. Record frames from the capture device to disk
with device:
    device.wait_until_ready(timeout=10.0)
    rgb_camera = device.get_rgb_camera()
    depth_camera = device.get_depth_camera()

    with writer:
        for _ in range(25 * 10):  # 10 seconds at 25 fps
            rgb_frame, rgb_ts = rgb_camera.get_next_frame(timeout_s=1.0)
            depth_frame, depth_ts = depth_camera.get_next_frame(timeout_s=1.0)

            rgb_queue.put((rgb_frame, rgb_ts), block=False)
            depth_queue.put((depth_frame, depth_ts), block=False)
```

## Device runtime

The OMGrab device runtime application built using the OMGrab SDK is included under [Device Runtime](runtime-app.md).
