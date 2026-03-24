"""Boot-time recovery: merge orphaned recording chunks."""
import logging
import pathlib

from omgrab.recording import manifest
from omgrab.recording import py_av_writer

logger = logging.getLogger(__name__)


def merge_orphaned_chunks(spool_dir: pathlib.Path, output_dir: pathlib.Path):
    """Scan spool_dir for recording directories with chunks and merge them.

    This is the boot-time recovery equivalent of the old
    ``files.enqueue_orphaned_capture_files()``. It handles:

    1. Recording directories with finalized .mkv chunks but no merged output:
       merge them into output_dir.
    2. Incomplete .mkv.tmp files from crashed writes: delete them.

    Args:
        spool_dir: Directory containing recording subdirectories (each
            with numbered .mkv chunk files).
        output_dir: Directory to write merged output files.
    """
    if not spool_dir.exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    for entry in sorted(spool_dir.iterdir()):
        if not entry.is_dir():
            continue

        recording_name = entry.name

        # Clean up any .tmp files from crashed writes.
        for tmp_file in entry.glob('*.mkv.tmp'):
            logger.info('Removing incomplete chunk: %s', tmp_file)
            try:
                tmp_file.unlink()
            except FileNotFoundError:
                pass

        # Check if there are finalized chunks to merge.
        chunks = sorted(entry.glob('*.mkv'))
        if not chunks:
            # Empty directory (or only had .tmp files) — remove it.
            try:
                entry.rmdir()
            except OSError:
                pass
            continue

        output_path = output_dir / f'{recording_name}.mkv'
        if output_path.exists():
            logger.info('Merged output already exists for %s, cleaning up chunks',
                        recording_name)
            py_av_writer._cleanup_recording_dir(entry)
            continue

        logger.info('Merging orphaned recording: %s (%d chunks)',
                    recording_name, len(chunks))
        try:
            py_av_writer.merge_recording_chunks(entry, output_path)
            logger.info('Orphaned recording merged: %s', output_path)
            recovered_manifest = manifest.build_manifest(
                recording_id=recording_name,
                started_at=None,
                stopped_at=None,
                duration_s=None,
                devices=[],
                stream_configs_map={},
                sensor_stream_configs_map={},
                output_file=output_path.name,
                recovered=True,
            )
            manifest.write_manifest(recovered_manifest, output_dir)
        except Exception as e:
            logger.error('Failed to merge orphaned recording %s: %s',
                         recording_name, e)
