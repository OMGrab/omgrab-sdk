"""Tests for the merge utilities (utils/merge.py)."""
import json
import pathlib

from omgrab.recording import py_av_writer
from omgrab.utils import merge


class TestMergeOrphanedChunks:

    def test_noop_when_spool_dir_missing(self, tmp_path: pathlib.Path):
        """Should not crash when spool_dir doesn't exist."""
        spool = tmp_path / 'nonexistent'
        output = tmp_path / 'output'
        merge.merge_orphaned_chunks(spool, output)

    def test_noop_when_spool_empty(self, tmp_path: pathlib.Path):
        """Should do nothing when spool_dir has no subdirectories."""
        spool = tmp_path / 'spool'
        spool.mkdir()
        output = tmp_path / 'output'
        merge.merge_orphaned_chunks(spool, output)
        assert not output.exists() or len(list(output.iterdir())) == 0

    def test_removes_tmp_files(self, tmp_path: pathlib.Path):
        """Should delete .mkv.tmp files from recording directories."""
        spool = tmp_path / 'spool'
        rec_dir = spool / '2026-03-17T00:00:00Z'
        rec_dir.mkdir(parents=True)
        tmp_file = rec_dir / '00001.mkv.tmp'
        tmp_file.write_bytes(b'incomplete')

        output = tmp_path / 'output'
        merge.merge_orphaned_chunks(spool, output)

        assert not tmp_file.exists()

    def test_removes_empty_recording_dirs(self, tmp_path: pathlib.Path):
        """Should remove empty recording directories."""
        spool = tmp_path / 'spool'
        rec_dir = spool / '2026-03-17T00:00:00Z'
        rec_dir.mkdir(parents=True)

        output = tmp_path / 'output'
        merge.merge_orphaned_chunks(spool, output)

        assert not rec_dir.exists()

    def test_skips_if_merged_output_exists(self, tmp_path: pathlib.Path):
        """Should skip merging if output file already exists."""
        spool = tmp_path / 'spool'
        rec_dir = spool / '2026-03-17T00:00:00Z'
        rec_dir.mkdir(parents=True)
        chunk = rec_dir / '00001.mkv'
        chunk.write_bytes(b'video data')

        output = tmp_path / 'output'
        output.mkdir()
        existing = output / '2026-03-17T00:00:00Z.mkv'
        existing.write_bytes(b'already merged')

        merge.merge_orphaned_chunks(spool, output)

        # The existing merged file should be untouched.
        assert existing.read_bytes() == b'already merged'

    def test_writes_recovered_manifest(self, tmp_path: pathlib.Path):
        """Orphaned merge should write a manifest with recovered=true."""
        spool = tmp_path / 'spool'
        rec_dir = spool / '2026-03-17T00-00-00Z'
        rec_dir.mkdir(parents=True)
        chunk = rec_dir / '00001.mkv'
        chunk.write_bytes(b'video data')

        output = tmp_path / 'output'
        merge.merge_orphaned_chunks(spool, output)

        manifest_path = output / '2026-03-17T00-00-00Z.json'
        assert manifest_path.exists()

        data = json.loads(manifest_path.read_text())
        assert data['recovered'] is True
        assert data['recording_id'] == '2026-03-17T00-00-00Z'
        assert data['started_at'] is None
        assert data['stopped_at'] is None
        assert data['duration_s'] is None
        assert data['devices'] == []
        assert data['output_file'] == '2026-03-17T00-00-00Z.mkv'


class TestMergeRecordingChunks:

    def test_no_chunks_is_noop(self, tmp_path: pathlib.Path):
        """Should not create output when recording dir is empty."""
        rec_dir = tmp_path / 'recording'
        rec_dir.mkdir()
        output = tmp_path / 'output.mkv'

        py_av_writer.merge_recording_chunks(rec_dir, output)

        assert not output.exists()

    def test_single_chunk_is_moved(self, tmp_path: pathlib.Path):
        """Single chunk should be moved directly without ffmpeg."""
        rec_dir = tmp_path / 'recording'
        rec_dir.mkdir()
        chunk = rec_dir / '00001.mkv'
        chunk.write_bytes(b'video data')

        output = tmp_path / 'output.mkv'
        py_av_writer.merge_recording_chunks(rec_dir, output)

        assert output.exists()
        assert output.read_bytes() == b'video data'
        # Recording dir should be cleaned up.
        assert not rec_dir.exists()



class TestMergeOrphanedEdgeCases:

    def test_skips_non_directory_entries(self, tmp_path: pathlib.Path):
        """Files in spool_dir should be skipped (only dirs are recordings)."""
        spool = tmp_path / 'spool'
        spool.mkdir()
        # Place a stray file in spool_dir.
        (spool / 'stray_file.txt').write_text('not a recording')

        output = tmp_path / 'output'
        merge.merge_orphaned_chunks(spool, output)

        # The file should still exist (not deleted).
        assert (spool / 'stray_file.txt').exists()

    def test_merge_failure_logs_and_continues(
            self, tmp_path: pathlib.Path, monkeypatch):
        """Merge failure should be logged without crashing."""
        spool = tmp_path / 'spool'
        rec_dir = spool / 'rec-1'
        rec_dir.mkdir(parents=True)
        (rec_dir / '00001.mkv').write_bytes(b'chunk')

        output = tmp_path / 'output'

        def _raise_merge(recording_dir, output_path):
            raise RuntimeError('ffmpeg failed')

        monkeypatch.setattr(
            py_av_writer, 'merge_recording_chunks', _raise_merge)

        # Should not raise.
        merge.merge_orphaned_chunks(spool, output)

        # Chunks should still be present (not cleaned up on failure).
        assert rec_dir.exists()

    def test_rmdir_failure_on_non_empty_dir(self, tmp_path: pathlib.Path):
        """OSError from rmdir on non-empty dir should be caught."""
        spool = tmp_path / 'spool'
        rec_dir = spool / 'rec-1'
        rec_dir.mkdir(parents=True)
        # No .mkv chunks, but a non-.tmp file that won't be cleaned up.
        (rec_dir / 'notes.txt').write_text('keep me')

        output = tmp_path / 'output'

        # Should not raise (rmdir fails because dir isn't empty).
        merge.merge_orphaned_chunks(spool, output)

        # Directory should still exist since rmdir failed.
        assert rec_dir.exists()
