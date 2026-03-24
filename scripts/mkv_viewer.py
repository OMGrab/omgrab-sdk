#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["av", "opencv-python", "numpy"]
# ///
"""Playback tool for viewing recorded MKV files (RGB + depth streams)."""

import argparse
import sys
import time

import av
import cv2
import numpy as np


def main():
    """Main entry point for the MKV viewer."""
    parser = argparse.ArgumentParser(description='View MKV recordings with RGB and depth streams.')
    parser.add_argument('filename', help='Path to the .mkv file')
    args = parser.parse_args()

    try:
        container = av.open(args.filename)
    except Exception as e:
        print(f'Error opening file: {e}')
        sys.exit(1)

    if len(container.streams.video) < 2:
        print(f'Expected at least 2 video streams, found {len(container.streams.video)}')
        sys.exit(1)

    stream_rgb = container.streams.video[0]
    stream_depth = container.streams.video[1]

    for i, (stream, name) in enumerate([(stream_rgb, 'RGB'), (stream_depth, 'Depth')]):
        print(
            f'Stream {i} ({name}): '
            f'{stream.width}x{stream.height} '
            f'@ {stream.average_rate} fps ({stream.codec_context.name})'
        )

    target_w = max(stream_rgb.width, stream_depth.width)
    target_h = max(stream_rgb.height, stream_depth.height)

    rgb_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    depth_img = np.zeros((target_h, target_w), dtype=np.uint8)

    start_wall = None
    first_pts = None

    stats = {
        'rgb': {'count': 0, 'first_pts': None, 'last_pts': None, 'time_base': stream_rgb.time_base},
        'depth': {
            'count': 0,
            'first_pts': None,
            'last_pts': None,
            'time_base': stream_depth.time_base,
        },
    }

    print('Starting playback... Press [Esc] to quit.')

    try:
        for packet in container.demux(stream_rgb, stream_depth):
            if packet.dts is None:
                continue

            for frame in packet.decode():
                if start_wall is None:
                    start_wall = time.time()
                    first_pts = frame.time

                if frame.time is not None:
                    wait = (frame.time - first_pts) - (time.time() - start_wall)
                    if wait > 0:
                        time.sleep(wait)

                is_rgb = packet.stream == stream_rgb
                key = 'rgb' if is_rgb else 'depth'
                stats[key]['count'] += 1
                if stats[key]['first_pts'] is None:
                    stats[key]['first_pts'] = frame.pts
                stats[key]['last_pts'] = frame.pts

                if is_rgb:
                    img = frame.to_ndarray(format='bgr24')
                    if img.shape[:2] != (target_h, target_w):
                        img = cv2.resize(img, (target_w, target_h))
                    rgb_img = img
                else:
                    img = frame.to_ndarray().astype(np.float32)
                    if img.shape[:2] != (target_h, target_w):
                        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                    inv = np.where(img > 0, 200.0 / img, 0.0)
                    p95 = np.percentile(inv[inv > 0], 95) if np.any(inv > 0) else 1.0
                    clipped = np.clip(inv, 0, p95)
                    depth_img = (clipped * 255.0 / p95).astype(np.uint8)

                depth_colored = cv2.applyColorMap(depth_img, cv2.COLORMAP_MAGMA)
                composite = cv2.hconcat([rgb_img, depth_colored])
                cv2.imshow('RGB | Depth', composite)

                if cv2.waitKey(1) & 0xFF == 27:
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error during playback: {e}')
    finally:
        container.close()
        cv2.destroyAllWindows()

        print('\nPlayback statistics:')
        for name, s in stats.items():
            duration = 0.0
            if (
                s['last_pts'] is not None
                and s['first_pts'] is not None
                and s['time_base'] is not None
            ):
                duration = float((s['last_pts'] - s['first_pts']) * s['time_base'])
            fps = s['count'] / duration if duration > 0 else 0.0
            print(f'  {name.upper()}: {s["count"]} frames, {duration:.2f}s, {fps:.1f} fps')


if __name__ == '__main__':
    main()
