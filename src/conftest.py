"""Root conftest — pytest hooks that must live at the testpaths root."""
import pytest


def pytest_addoption(parser: pytest.Parser):
    """Register custom CLI flags."""
    parser.addoption(
        '--hardware',
        action='store_true',
        default=False,
        help='Run tests that require real GPIO hardware.',
    )
    parser.addoption(
        '--slow',
        action='store_true',
        default=False,
        help='Run tests that take >5s (e.g. long-press timing tests).',
    )


def pytest_collection_modifyitems(
        config: pytest.Config, items: list[pytest.Item]):
    """Skip @pytest.mark.hardware and @pytest.mark.slow tests unless opted in."""
    if not config.getoption('--hardware'):
        skip_hw = pytest.mark.skip(reason='needs --hardware flag to run')
        for item in items:
            if 'hardware' in item.keywords:
                item.add_marker(skip_hw)

    if not config.getoption('--slow'):
        skip_slow = pytest.mark.skip(reason='needs --slow flag to run')
        for item in items:
            if 'slow' in item.keywords:
                item.add_marker(skip_slow)
