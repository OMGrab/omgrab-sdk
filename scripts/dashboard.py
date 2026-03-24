#!/usr/bin/env python3
"""TUI dashboard for monitoring device status.

Reads from /data/device_status.json and displays a live-updating dashboard.
"""
from typing import Optional

import curses
import json
import pathlib
import time

STATUS_FILE = pathlib.Path('/opt/omgrab/data/device_status.json')
REFRESH_INTERVAL = 1.0  # seconds

# EMA smoothing for battery percentage
BATTERY_EMA_ALPHA = 0.05  # Lower = smoother, higher = more responsive


class DashboardState:
    """Tracks smoothed values across refreshes."""

    def __init__(self):
        self.battery_pct_ema: Optional[float] = None

    def update_battery_ema(self, raw_pct: float) -> float:
        """Update and return smoothed battery percentage."""
        if self.battery_pct_ema is None:
            # Initialize with first reading
            self.battery_pct_ema = raw_pct
        else:
            # Exponential moving average
            self.battery_pct_ema = (
                BATTERY_EMA_ALPHA * raw_pct +
                (1 - BATTERY_EMA_ALPHA) * self.battery_pct_ema
            )
        return self.battery_pct_ema


def format_bytes(b: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(b) < 1024:
            return f'{b:.1f} {unit}'
        b /= 1024
    return f'{b:.1f} PB'


def format_uptime(seconds: float) -> str:
    """Format uptime as human-readable string."""
    days = int(seconds // 86400)
    hours = int((seconds % 86400) // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if days > 0:
        return f'{days}d {hours}h {minutes}m'
    if hours > 0:
        return f'{hours}h {minutes}m {secs}s'
    return f'{minutes}m {secs}s'


def read_status() -> dict:
    """Read status from JSON file."""
    try:
        with open(STATUS_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def draw_box(win, y: int, x: int, h: int, w: int, title: str = ''):
    """Draw a box with optional title."""
    win.attron(curses.color_pair(1))
    # Top border
    win.addstr(y, x, '┌' + '─' * (w - 2) + '┐')
    # Side borders
    for i in range(1, h - 1):
        win.addstr(y + i, x, '│')
        win.addstr(y + i, x + w - 1, '│')
    # Bottom border
    win.addstr(y + h - 1, x, '└' + '─' * (w - 2) + '┘')
    win.attroff(curses.color_pair(1))
    # Title (in cyan/bold)
    if title:
        win.attron(curses.color_pair(1) | curses.A_BOLD)
        win.addstr(y, x + 2, f' {title} ')
        win.attroff(curses.color_pair(1) | curses.A_BOLD)


def draw_dashboard(stdscr, status: dict, state: DashboardState):
    """Draw the dashboard."""
    stdscr.clear()
    height, width = stdscr.getmaxyx()

    # Title with gradient effect
    title_left = '═══'
    title_text = ' OMGrab Dashboard '
    title_right = '═══'
    title_start = (width - len(title_left) - len(title_text) - len(title_right)) // 2
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(0, title_start, title_left)
    stdscr.attroff(curses.color_pair(1))
    stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
    stdscr.addstr(title_text)
    stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(title_right)
    stdscr.attroff(curses.color_pair(1))

    if not status:
        stdscr.attron(curses.color_pair(2))
        stdscr.addstr(2, 2, 'Waiting for status data...')
        stdscr.attroff(curses.color_pair(2))
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(3, 2, f'Watching: {STATUS_FILE}')
        stdscr.attroff(curses.color_pair(1))
        stdscr.refresh()
        return

    # Health indicator in top right
    healthy = status.get('device_healthy', False)
    health_str = '● HEALTHY' if healthy else '● UNHEALTHY'
    health_color = curses.color_pair(3) if healthy else curses.color_pair(4)
    stdscr.attron(health_color | curses.A_BOLD)
    stdscr.addstr(0, width - len(health_str) - 2, health_str)
    stdscr.attroff(health_color | curses.A_BOLD)

    y = 2
    col1_x = 1
    col2_x = width // 2

    # Device info
    draw_box(stdscr, y, col1_x, 5, width // 2 - 1, 'Device')
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 1, col1_x + 2, 'ID: ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(curses.color_pair(7) | curses.A_BOLD)
    stdscr.addstr(f"{status.get('device_id', '?')}")
    stdscr.attroff(curses.color_pair(7) | curses.A_BOLD)

    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 2, col1_x + 2, 'Version: ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(curses.color_pair(6))
    stdscr.addstr(f"{status.get('software_version', '?')}")
    stdscr.attroff(curses.color_pair(6))

    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 3, col1_x + 2, 'Uptime: ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(curses.color_pair(6))
    stdscr.addstr(f"{format_uptime(status.get('uptime_seconds', 0))}")
    stdscr.attroff(curses.color_pair(6))

    # State
    draw_box(stdscr, y, col2_x, 5, width // 2 - 1, 'State')
    machine_state = status.get('state_machine_state', '?').upper()
    state_color = curses.color_pair(3) if machine_state == 'RECORDING' else curses.color_pair(6)
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 1, col2_x + 2, 'State: ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(state_color | curses.A_BOLD)
    stdscr.addstr(f'{machine_state}')
    stdscr.attroff(state_color | curses.A_BOLD)

    # Recording indicator
    if machine_state == 'RECORDING':
        stdscr.attron(curses.color_pair(4) | curses.A_BLINK)
        stdscr.addstr(' ●')
        stdscr.attroff(curses.color_pair(4) | curses.A_BLINK)

    recording = status.get('recording', {})
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 2, col2_x + 2, 'Recording: ')
    stdscr.attroff(curses.A_DIM)
    if recording.get('is_recording'):
        stdscr.attron(curses.color_pair(3))
        rec_id = recording.get('recording_id', '?')
        stdscr.addstr(f'{rec_id[:20]}')
        stdscr.attroff(curses.color_pair(3))
    else:
        stdscr.addstr('-')

    network = status.get('network', {})
    net_status = network.get('status', '?').upper()
    net_color = (
        curses.color_pair(3)
        if net_status == 'ONLINE'
        else (
            curses.color_pair(2)
            if net_status == 'NETWORK_ONLY'
            else (
                curses.color_pair(4)
            )
        )
    )
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 3, col2_x + 2, 'Network: ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(net_color | curses.A_BOLD)
    stdscr.addstr(f'{net_status}')
    stdscr.attroff(net_color | curses.A_BOLD)

    y += 6

    # System resources
    draw_box(stdscr, y, col1_x, 7, width // 2 - 1, 'System')
    cpu = status.get('cpu', {})
    mem = status.get('memory', {})
    storage = status.get('storage', {})

    temp = cpu.get('temperature_celsius')
    temp_str = f'{temp:.1f}°C' if temp else 'N/A'
    temp_color = (
        curses.color_pair(4)
        if temp and temp > 70
        else (
            curses.color_pair(3)
            if temp and temp < 50
            else curses.color_pair(2)
        )
    )
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 1, col1_x + 2, 'CPU Temp: ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(temp_color | curses.A_BOLD)
    stdscr.addstr(temp_str)
    stdscr.attroff(temp_color | curses.A_BOLD)

    # Show per-core CPU usage with color coding
    cores = cpu.get('usage_per_core', [])
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 2, col1_x + 2, 'CPU Cores: ')
    stdscr.attroff(curses.A_DIM)
    if cores:
        stdscr.addstr('[')
        for i, c in enumerate(cores[:4]):
            core_color = (
                curses.color_pair(4)
                if c > 80 else (
                    curses.color_pair(2)
                    if c > 50 else (
                        curses.color_pair(3)
                    )
                )
            )
            stdscr.attron(core_color)
            stdscr.addstr(f'{c:.0f}%')
            stdscr.attroff(core_color)
            if i < min(len(cores), 4) - 1:
                stdscr.addstr(', ')
        stdscr.addstr(']')
    else:
        stdscr.addstr('N/A')

    mem_used = format_bytes(mem.get('used_bytes', 0))
    mem_total = format_bytes(mem.get('total_bytes', 0))
    mem_pct = mem.get('used_percent', 0)
    mem_color = (
        curses.color_pair(4) if mem_pct > 80 else (
            curses.color_pair(2) if mem_pct > 60 else curses.color_pair(3)
        )
    )
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 3, col1_x + 2, 'Memory: ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(mem_color)
    stdscr.addstr(f'{mem_used} / {mem_total}')
    stdscr.attroff(mem_color)

    storage_pct = storage.get('used_percent', 0)
    storage_color = (
        curses.color_pair(4) if storage_pct > 80 else (
            curses.color_pair(2) if storage_pct > 60 else curses.color_pair(3)
        )
    )
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 4, col1_x + 2, 'Storage: ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(storage_color)
    stdscr.addstr(f'{storage_pct:.1f}%')
    stdscr.attroff(storage_color)
    stdscr.attron(curses.color_pair(6))
    stdscr.addstr(f" ({format_bytes(storage.get('available_bytes', 0))} free)")
    stdscr.attroff(curses.color_pair(6))

    # Battery
    draw_box(stdscr, y, col2_x, 7, width // 2 - 1, 'Battery')
    battery = status.get('battery')
    if battery:
        raw_pct = battery.get('percent', 0)
        pct = state.update_battery_ema(raw_pct)  # Smoothed value
        bar_width = 20
        filled = int(pct / 100 * bar_width)
        pct_color = (
            curses.color_pair(4) if pct < 20 else (
                curses.color_pair(3) if pct > 50 else curses.color_pair(2)
            )
        )

        stdscr.attron(curses.A_DIM)
        stdscr.addstr(y + 1, col2_x + 2, 'Charge: ')
        stdscr.attroff(curses.A_DIM)
        stdscr.attron(pct_color | curses.A_BOLD)
        stdscr.addstr(f'{pct:.1f}%')
        stdscr.attroff(pct_color | curses.A_BOLD)

        # Colored battery bar
        stdscr.addstr(y + 2, col2_x + 2, '[')
        stdscr.attron(pct_color)
        stdscr.addstr('█' * filled)
        stdscr.attroff(pct_color)
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr('░' * (bar_width - filled))
        stdscr.attroff(curses.color_pair(1))
        stdscr.addstr(']')

        current = battery.get('current_a', 0)
        charging = current > 0
        stdscr.attron(curses.A_DIM)
        stdscr.addstr(y + 3, col2_x + 2, 'Voltage: ')
        stdscr.attroff(curses.A_DIM)
        stdscr.attron(curses.color_pair(6))
        stdscr.addstr(f"{battery.get('voltage_v', 0):.2f}V")
        stdscr.attroff(curses.color_pair(6))
        if charging:
            stdscr.attron(curses.color_pair(3))
            stdscr.addstr(' ⚡ CHARGING')
            stdscr.attroff(curses.color_pair(3))

        stdscr.attron(curses.A_DIM)
        stdscr.addstr(y + 4, col2_x + 2, 'Power: ')
        stdscr.attroff(curses.A_DIM)
        stdscr.attron(curses.color_pair(6))
        stdscr.addstr(f"{battery.get('power_w', 0):.2f}W @ {abs(current):.3f}A")
        stdscr.attroff(curses.color_pair(6))
    else:
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr(y + 2, col2_x + 2, 'No battery detected')
        stdscr.attroff(curses.color_pair(1))

    y += 8

    # Network details
    draw_box(stdscr, y, col1_x, 5, width - 2, 'Network')
    ssid = network.get('wifi_ssid')
    signal = network.get('wifi_signal_strength')

    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 1, col1_x + 2, 'WiFi: ')
    stdscr.attroff(curses.A_DIM)
    if ssid:
        stdscr.attron(curses.color_pair(3))
        stdscr.addstr(f'{ssid}')
        stdscr.attroff(curses.color_pair(3))
    else:
        stdscr.attron(curses.color_pair(4))
        stdscr.addstr('Not connected')
        stdscr.attroff(curses.color_pair(4))

    stdscr.attron(curses.A_DIM)
    stdscr.addstr(y + 2, col1_x + 2, 'Signal: ')
    stdscr.attroff(curses.A_DIM)
    if signal:
        sig_color = (
            curses.color_pair(3) if signal > -60 else (
                curses.color_pair(2) if signal > -70 else curses.color_pair(4)
            )
        )
        stdscr.attron(sig_color)
        stdscr.addstr(f'{signal} dBm')
        stdscr.attroff(sig_color)
        # Signal bars
        bars = 4 if signal > -50 else 3 if signal > -60 else 2 if signal > -70 else 1
        stdscr.addstr(' ')
        stdscr.attron(sig_color)
        stdscr.addstr('▂▄▆█'[:bars])
        stdscr.attroff(sig_color)
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr('▂▄▆█'[bars:])
        stdscr.attroff(curses.color_pair(1))
    else:
        stdscr.attron(curses.color_pair(1))
        stdscr.addstr('N/A')
        stdscr.attroff(curses.color_pair(1))

    # Footer
    updated_at = status.get('_updated_at')
    age = time.time() - updated_at if updated_at else 0
    age_color = (
        curses.color_pair(3) if age < 2 else (
            curses.color_pair(2) if age < 5 else curses.color_pair(4)
        )
    )
    footer_start = (width - 35) // 2
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(height - 1, footer_start, 'Updated ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(age_color)
    stdscr.addstr(f'{age:.1f}s ago')
    stdscr.attroff(age_color)
    stdscr.attron(curses.color_pair(1))
    stdscr.addstr(' │ ')
    stdscr.attroff(curses.color_pair(1))
    stdscr.attron(curses.A_DIM)
    stdscr.addstr('Press ')
    stdscr.attroff(curses.A_DIM)
    stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
    stdscr.addstr('q')
    stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
    stdscr.attron(curses.A_DIM)
    stdscr.addstr(' to quit')
    stdscr.attroff(curses.A_DIM)

    stdscr.refresh()


def main(stdscr):
    """Main dashboard loop."""
    # Setup colors
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)     # Box borders, section titles
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # Main title, highlights
    curses.init_pair(3, curses.COLOR_GREEN, -1)    # Good status
    curses.init_pair(4, curses.COLOR_RED, -1)      # Warning/bad status
    curses.init_pair(5, curses.COLOR_WHITE, -1)    # Labels (dimmed)
    curses.init_pair(6, curses.COLOR_BLUE, -1)     # Values
    curses.init_pair(7, curses.COLOR_WHITE, -1)    # Bright white

    curses.curs_set(0)  # Hide cursor
    stdscr.timeout(int(REFRESH_INTERVAL * 1000))  # Non-blocking getch

    state = DashboardState()

    while True:
        status = read_status()
        try:
            draw_dashboard(stdscr, status, state)
        except curses.error:
            pass  # Terminal too small

        key = stdscr.getch()
        if key == ord('q') or key == ord('Q'):
            break


if __name__ == '__main__':
    curses.wrapper(main)
