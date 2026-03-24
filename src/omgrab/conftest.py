"""Shared pytest fixtures for omgrab unit tests."""
import pathlib

import pytest

from omgrab import testing


@pytest.fixture
def fake_gpio() -> testing.FakeGPIOController:
    """Provide a FakeGPIOController instance."""
    return testing.FakeGPIOController()


@pytest.fixture
def fake_capture_device() -> testing.FakeCaptureDevice:
    """Provide a FakeCaptureDevice instance."""
    return testing.FakeCaptureDevice()


@pytest.fixture
def spool_dirs(tmp_path: pathlib.Path) -> dict[str, pathlib.Path]:
    """Create the standard spool directory hierarchy.

    Returns:
        Dict with keys: spool, output.
    """
    dirs = {
        'spool': tmp_path / 'spool',
        'output': tmp_path / 'output',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
