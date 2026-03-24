"""Chunked video writer with per-stream parallel encoding."""
from typing import Any
from typing import Optional

import datetime
import itertools
import logging
import pathlib
import queue
import threading
import time
from collections.abc import Callable

import av

from omgrab.recording import py_av_writer
from omgrab.recording.stream_configs import DataStreamConfig
from omgrab.recording.stream_configs import VideoStreamConfig

logger = logging.getLogger(__name__)


# Type aliases
ChunkID = str

# Callback type for starting a chunk: (name, started_at, file_extension) -> chunk_id
StartChunkCallback = Callable[[str, datetime.datetime, str], ChunkID]

EXPECTED_METADATA_KEYS = ['type']
STATUS_LOG_INTERVAL_S = 5.0
_ROTATION_BARRIER_TIMEOUT_S = 5.0
_ENCODER_JOIN_TIMEOUT_S = 5.0


def _verify_metadata(metadata: dict[str, Any]):
    """Verify the metadata contains expected keys."""
    for expected_key in EXPECTED_METADATA_KEYS:
        if expected_key not in metadata:
            raise ValueError(f'Metadata: {metadata}\nMissing expected key: {expected_key}')


class ChunkedWriter:
    """Chunked video writer with per-stream parallel encoding.

    Each stream gets its own encoder thread AND its own single-stream MKV
    container. Chunks are rotated after chunk_length_s seconds of recording
    time, measured from the first frame's timestamp.

    All streams share a common timestamp origin per chunk so PTS values are
    directly comparable across .part files. When the first frame arrives on
    any stream, its timestamp becomes the origin for the first chunk. On
    rotation, the origin advances by exactly chunk_length_s.

    At finalization (chunk boundary or stop), per-stream .mkv.part files
    are merged into a single multi-stream MKV via ffmpeg -c copy.

    Rotation protocol (barrier with generation counter):
    1. Any encoder detects a frame whose timestamp exceeds the current
       chunk's end boundary and sets rotation_requested.
    2. All encoders flush their encoder, close their container, and enter
       the rotation barrier.
    3. The last encoder to arrive merges the per-stream files, advances
       the chunk origin, creates new per-stream containers, and advances
       the generation.
    4. All encoders resume with new encoder/container objects.
    """

    def __init__(
            self,
            name: str,
            output_directory: pathlib.Path,
            stream_configs: dict[str, VideoStreamConfig],
            start_chunk_callback: Optional[StartChunkCallback] = None,
            sensor_stream_configs: Optional[dict[str, DataStreamConfig]] = None,
            chunk_length_s: float = 60.0,
            max_encoder_queue_size: int = 200):
        """Initialize the chunked writer.

        Args:
            name: Name for this writer (used in log messages and thread names).
            output_directory: Directory to write video files.
            stream_configs: Dictionary mapping stream names to their config.
            start_chunk_callback: Optional callback to generate chunk IDs.
                Signature: (name, started_at, file_extension) -> chunk_id.
                When None, sequential 5-digit IDs are generated automatically.
            sensor_stream_configs: Optional sensor stream configurations.
            chunk_length_s: Chunk duration in seconds. The first chunk starts
                when the first frame arrives; subsequent chunks advance by
                this amount.
            max_encoder_queue_size: Max queue size per encoder thread.
        """
        for video_cfg in stream_configs.values():
            _verify_metadata(video_cfg.metadata)
        sensor_stream_configs = sensor_stream_configs or {}
        for data_cfg in sensor_stream_configs.values():
            _verify_metadata(data_cfg.metadata)

        if start_chunk_callback is None:
            counter = itertools.count(1)

            def start_chunk_callback(_name: str, _ts: datetime.datetime, _ext: str):
                return f'{next(counter):05d}'

        self._name = name
        self._output_directory = output_directory
        self._stream_configs = stream_configs
        self._sensor_stream_configs = sensor_stream_configs
        self._start_chunk_callback = start_chunk_callback
        self._chunk_length_s = chunk_length_s

        all_names = list(stream_configs.keys()) + list(sensor_stream_configs.keys())
        self._all_stream_names = all_names
        self._num_encoders = len(all_names)

        self._encoder_queues: dict[str, queue.Queue] = {
            name: queue.Queue(maxsize=max_encoder_queue_size)
            for name in all_names
        }

        self._rotation_requested = threading.Event()
        self._rotation_generation = 0
        self._rotation_condition = threading.Condition()
        self._rotation_lock = threading.Lock()
        self._rotation_arrivals = 0

        # Shared timestamp origin for the current chunk (seconds since epoch).
        # Set from the first frame's timestamp; advanced by chunk_length_s on
        # each rotation.
        self._chunk_start_s: Optional[float] = None
        self._chunk_start_lock = threading.Lock()

        self._current_chunk_id: Optional[str] = None
        self._current_chunk_output_path: Optional[pathlib.Path] = None
        self._stream_containers: dict[str, av.container.OutputContainer] = {}
        self._encoders: dict[str, py_av_writer.StreamEncoder] = {}

        self._encoder_threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._started = False

    def get_encoder_queue(self, stream_name: str) -> queue.Queue:
        """Get the input queue for a stream's encoder thread."""
        return self._encoder_queues[stream_name]

    def is_data_stream(self, stream_name: str) -> bool:
        """Check if a stream name refers to a data stream."""
        return stream_name in self._sensor_stream_configs

    def __enter__(self):
        """Start the writer and return self for use as a context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the writer on context manager exit."""
        self.stop()

    def start(self):
        """Start all encoder threads.

        Chunk creation is deferred until the first frame arrives, so the
        shared timestamp origin matches the actual frame domain rather
        than the wall clock.
        """
        if self._started:
            raise RuntimeError('ChunkedWriter already started')
        self._started = True

        first_stream = next(iter(self._stream_configs), None)
        for name in self._all_stream_names:
            t = threading.Thread(
                target=self._encoder_loop,
                args=(name, name == first_stream),
                name=f'encode-{name}-{self._name}',
                daemon=True)
            t.start()
            self._encoder_threads.append(t)

    def stop(self):
        """Signal all encoder threads to stop, join them, and merge the final chunk."""
        self._stop_event.set()
        # Unblock any encoders waiting on rotation.
        with self._rotation_condition:
            self._rotation_generation += 1
            self._rotation_condition.notify_all()

        for t in self._encoder_threads:
            t.join(timeout=_ENCODER_JOIN_TIMEOUT_S)
            if t.is_alive():
                logger.warning(
                    '%s: encoder thread %s did not exit within %.1fs',
                    self._name, t.name, _ENCODER_JOIN_TIMEOUT_S)

        # Merge the final chunk's per-stream files.
        self._merge_current_chunk()

    def _ensure_first_chunk(self, timestamp: datetime.datetime):
        """Initialize the first chunk from the first frame's timestamp.

        Called by encoder threads. Only the first caller creates the file;
        others block on the lock until it is ready.
        """
        with self._chunk_start_lock:
            if self._chunk_start_s is not None:
                return
            self._chunk_start_s = timestamp.timestamp()
            self._create_chunk(timestamp)

    def _create_chunk(self, timestamp: datetime.datetime):
        """Create per-stream MKV part files for a new chunk."""
        assert self._chunk_start_s is not None
        chunk_id = self._start_chunk_callback(
            self._name, timestamp, 'mkv')
        self._current_chunk_id = chunk_id
        self._current_chunk_output_path = (
            self._output_directory / f'{chunk_id}.mkv')

        self._stream_containers.clear()
        self._encoders.clear()

        for name, video_cfg in self._stream_configs.items():
            part_path = (
                self._output_directory / f'{chunk_id}.{name}.mkv.part')
            container = av.open(
                part_path.as_posix(), mode='w', format='matroska')
            self._stream_containers[name] = container
            self._encoders[name] = py_av_writer.VideoStreamEncoder(
                container=container,
                width=video_cfg.width,
                height=video_cfg.height,
                fps=video_cfg.fps,
                codec=video_cfg.codec,
                bitrate=video_cfg.bitrate,
                input_pixel_format=video_cfg.input_pixel_format,
                output_pixel_format=video_cfg.output_pixel_format,
                options=video_cfg.stream_options,
                timestamp_origin_s=self._chunk_start_s,
                metadata=video_cfg.metadata,
            )

        for name, data_cfg in self._sensor_stream_configs.items():
            part_path = (
                self._output_directory / f'{chunk_id}.{name}.mkv.part')
            container = av.open(
                part_path.as_posix(), mode='w', format='matroska')
            self._stream_containers[name] = container
            self._encoders[name] = py_av_writer.DataStreamEncoder(
                container=container,
                timestamp_origin_s=self._chunk_start_s,
                codec=data_cfg.codec,
                metadata=data_cfg.metadata,
            )

        logger.debug('%s: created new chunk %s', self._name, chunk_id)

    def _close_stream_container(self, stream_name: str):
        """Close a single stream's container."""
        container = self._stream_containers.get(stream_name)
        if container:
            try:
                container.close()
            except Exception:
                logger.error('Error closing container for stream %s',
                             stream_name, exc_info=True)

    def _merge_current_chunk(self):
        """Merge per-stream .mkv.part files into a single multi-stream MKV."""
        if self._current_chunk_id is None:
            return

        part_paths = []
        for name in self._all_stream_names:
            part_path = (
                self._output_directory
                / f'{self._current_chunk_id}.{name}.mkv.part')
            if part_path.exists() and part_path.stat().st_size > 0:
                part_paths.append(part_path)
            else:
                logger.warning('Chunk %s: missing or empty part file for stream %s',
                               self._current_chunk_id, name)

        if not part_paths:
            logger.warning('Chunk %s: no part files found, skipping merge', self._current_chunk_id)
            return

        try:
            py_av_writer.merge_stream_files(
                part_paths, self._current_chunk_output_path)
        except Exception as e:
            logger.error('Failed to merge chunk %s: %s. '
                         'Part files left for boot-time cleanup.',
                         self._current_chunk_id, e)
            tmp_path = self._current_chunk_output_path.with_suffix('.mkv.tmp')
            if tmp_path.exists():
                tmp_path.unlink()

    def _check_and_request_rotation(self, timestamp: datetime.datetime) -> bool:
        """Check if this timestamp exceeds the current chunk's boundary.

        Returns:
            True if rotation was requested.
        """
        timestamp_s = timestamp.timestamp()
        with self._chunk_start_lock:
            assert self._chunk_start_s is not None
            if timestamp_s - self._chunk_start_s >= self._chunk_length_s:
                self._rotation_requested.set()
                return True
        return False

    def _participate_in_rotation(self, stream_name: str,
                                 rotation_timestamp: datetime.datetime):
        """Enter rotation barrier. Last encoder to arrive does the merge.

        Args:
            stream_name: Name of the stream entering the barrier.
            rotation_timestamp: Timestamp of the frame that triggered rotation.
        """
        gen = self._rotation_generation
        is_last = False

        with self._rotation_lock:
            self._rotation_arrivals += 1
            is_last = self._rotation_arrivals >= self._num_encoders

        if is_last:
            self._merge_current_chunk()

            with self._chunk_start_lock:
                assert self._chunk_start_s is not None
                self._chunk_start_s += self._chunk_length_s

            chunk_start_dt = datetime.datetime.fromtimestamp(self._chunk_start_s)
            self._create_chunk(chunk_start_dt)

            with self._rotation_lock:
                self._rotation_arrivals = 0
            self._rotation_requested.clear()
            with self._rotation_condition:
                self._rotation_generation += 1
                self._rotation_condition.notify_all()
        else:
            with self._rotation_condition:
                while True:
                    ready = self._rotation_condition.wait_for(
                        lambda: self._rotation_generation > gen
                        or self._stop_event.is_set(),
                        timeout=_ROTATION_BARRIER_TIMEOUT_S)
                    if ready:
                        break
                    alive = sum(1 for t in self._encoder_threads
                                if t.is_alive())
                    if alive < self._num_encoders:
                        logger.warning(
                            '%s: rotation barrier broken — '
                            '%d/%d encoder threads alive',
                            self._name, alive, self._num_encoders)
                        self._stop_event.set()
                        break

    def _encoder_loop(self, stream_name: str, log_status: bool):
        """Encoder thread for any stream type (video or data).

        Each thread owns its own single-stream container and muxes directly
        into it. No shared container or packet queue.

        Args:
            stream_name: Name of the stream this thread encodes.
            log_status: If True, periodically log recording progress.
        """
        eq = self._encoder_queues[stream_name]
        already_flushed = False
        start_time = time.monotonic()
        last_log_time = start_time
        frame_count = 0

        while True:
            try:
                item = eq.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            data, timestamp = item
            frame_count += 1

            if log_status:
                now = time.monotonic()
                if now - last_log_time >= STATUS_LOG_INTERVAL_S:
                    last_log_time = now
                    elapsed = now - start_time
                    minutes = int(elapsed // 60)
                    seconds = elapsed % 60
                    logger.info(
                        '%s: recorded %dm%04.1fs (%d frames)',
                        self._name, minutes, seconds, frame_count)

            try:
                self._ensure_first_chunk(timestamp)

                if not self._rotation_requested.is_set():
                    self._check_and_request_rotation(timestamp)

                if self._rotation_requested.is_set():
                    encoder = self._encoders.get(stream_name)
                    container = self._stream_containers.get(stream_name)
                    if encoder and container:
                        for pkt in encoder.flush():
                            container.mux(pkt)
                    self._close_stream_container(stream_name)
                    self._participate_in_rotation(stream_name, timestamp)
                    if self._stop_event.is_set():
                        already_flushed = True
                        break

                encoder = self._encoders.get(stream_name)
                if encoder is None:
                    continue

                container = self._stream_containers.get(stream_name)
                if container is None:
                    continue
                timestamp_s = timestamp.timestamp()
                for pkt in encoder.encode(data, timestamp_s):
                    container.mux(pkt)
            except Exception:
                logger.error(
                    '%s: encoder thread %s crashed',
                    self._name, stream_name, exc_info=True)
                self._stop_event.set()
                with self._rotation_condition:
                    self._rotation_condition.notify_all()
                self._close_stream_container(stream_name)
                return

        if not already_flushed:
            encoder = self._encoders.get(stream_name)
            container = self._stream_containers.get(stream_name)
            if encoder and container:
                for pkt in encoder.flush():
                    container.mux(pkt)
        self._close_stream_container(stream_name)
