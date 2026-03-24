"""Local conftest for GPIO tests.

Excludes legacy standalone scripts that are not pytest-compatible (they
run GPIO hardware at module-import time and hang when collected on
machines without GPIO access).
"""

collect_ignore = ['test_gpio.py', 'gpio_cleanup.py']
