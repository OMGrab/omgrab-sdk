"""Prototype script for camera alignment preview on the OLED.

Uses OakDCaptureDevice.preview() to run a lightweight RGB-only pipeline
and displays dithered frames on the SSD1306 OLED at the maximum rate
the I2C bus allows.

Run directly on the Pi with the OAK-D and OLED connected:

    python -m omgrab.display.test_alignment

Press Ctrl-C to quit.
"""
import signal
import sys
import time

import PIL.Image
import PIL.ImageOps

from omgrab.cameras import cameras
from omgrab.devices import oakd_capture_device
from omgrab.display import screen_writer as screen_writer_module

# Target display dimensions.
DISPLAY_W = 128
DISPLAY_H = 64

# Preview camera config (produce faster than we can display so the
# queue always has a fresh frame ready; display runs as fast as I2C allows).
PREVIEW_FPS = 10
PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 400


def _frame_to_display_image(frame: cameras.RGBFrame) -> PIL.Image.Image:
    """Convert an RGB frame to a 1-bit dithered image for the OLED.

    Steps:
    1. Convert numpy array to PIL Image.
    2. Crop to the display aspect ratio (2:1) if needed.
    3. Resize to 128x64 with high-quality downsampling.
    4. Convert to grayscale.
    5. Apply Floyd-Steinberg dithering to 1-bit.

    Args:
        frame: RGB numpy array (H, W, 3).

    Returns:
        1-bit PIL Image ready for the display.
    """
    image = PIL.Image.fromarray(frame)

    # Crop to 2:1 aspect ratio (display is 128x64 = 2:1).
    src_w, src_h = image.size
    target_ratio = DISPLAY_W / DISPLAY_H  # 2.0
    src_ratio = src_w / src_h

    if src_ratio > target_ratio:
        # Source is wider: crop sides.
        new_w = int(src_h * target_ratio)
        offset = (src_w - new_w) // 2
        image = image.crop((offset, 0, offset + new_w, src_h))
    elif src_ratio < target_ratio:
        # Source is taller: crop top/bottom.
        new_h = int(src_w / target_ratio)
        offset = (src_h - new_h) // 2
        image = image.crop((0, offset, src_w, offset + new_h))

    # Resize -> grayscale -> equalize -> dither to 1-bit.
    image = image.resize((DISPLAY_W, DISPLAY_H), PIL.Image.Resampling.LANCZOS)
    image = image.convert('L')
    image = PIL.ImageOps.equalize(image)
    image = image.convert('1')  # Floyd-Steinberg dithering by default.
    return image


def main():
    """Capture and display dithered camera preview on the OLED."""
    print('Initializing screen...')
    writer = screen_writer_module.ScreenWriter()
    if not writer.available:
        print('ERROR: Screen hardware not available.')
        return

    print('Initializing OAK-D capture device (preview mode)...')
    # Recording configs are unused but required by the constructor.
    rgb_config = cameras.CameraConfig(fps=25, width=1280, height=800)
    depth_config = cameras.CameraConfig(fps=25, width=640, height=400)
    preview_config = cameras.CameraConfig(
        fps=PREVIEW_FPS, width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)

    device = oakd_capture_device.OakDCaptureDevice(
        rgb_config, depth_config,
        max_queue_size=2,
        preview_config=preview_config,
    )
    preview_cam = device.get_preview_camera()

    display_count = 0
    start_time = time.monotonic()

    def _cleanup_and_exit():
        elapsed = time.monotonic() - start_time
        if elapsed > 0 and display_count > 0:
            print(f'\n{display_count} frames displayed in {elapsed:.1f}s '
                  f'({display_count / elapsed:.1f} fps)')
        print('Clearing screen...')
        writer.clear()
        writer.cleanup()

    # Graceful shutdown on Ctrl-C.
    def _signal_handler(signum, frame):
        _cleanup_and_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)

    print('Starting preview pipeline...')
    with device.preview():
        # Flush any stale frames from startup.
        preview_cam.flush_queue()

        print('Streaming as fast as display allows. Press Ctrl-C to quit.\n')

        while True:
            try:
                frame, _timestamp = preview_cam.get_next_frame(timeout_s=2.0)
            except cameras.FrameUnavailableError:
                print('  Warning: frame timeout')
                continue

            # Drain any older frames so we always display the freshest one.
            while True:
                try:
                    frame, _timestamp = preview_cam.get_next_frame(timeout_s=0)
                except cameras.FrameUnavailableError:
                    break

            display_image = _frame_to_display_image(frame)
            writer.display(display_image)

            display_count += 1
            if display_count % 25 == 0:
                elapsed = time.monotonic() - start_time
                print(f'  {display_count} frames displayed, '
                      f'{display_count / elapsed:.1f} fps')


if __name__ == '__main__':
    main()
