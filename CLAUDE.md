# Style
## Imports
Imports should be ordered alphabetically and split into three sections: standard libraries, external libraries, internal libraries. Only import modules, not classes or methods, e.g.
```
# Okay
import enum

class Foo(enum.Enum):
...
```

```
# Not okay
from enum import Enum

class Foo(Enum):
...
```

The exception is the typing module. Typing imports should be placed above all other imports, and importing types directly from typing is permitted.

## Quotes and docstrings
Prefer single quotes, except in docstrings or where nesting makes double quotes required.

Docstrings should be either one line, or if multiple lines should include args, returns, and raises sections.
Always start the docstring on the line immediately below the function definition.

Do not include -> None for functions that return nothing.

E.g.
```
def foo(a: int) -> int:
  """Do foo."""
```
or
```
def foo(a: int) -> int:
  """Do foo.

    Args:
        a: The a parameter.

    Returns:
        The result of foo.

    Raises:
        FooError: When an error occurs.
  """
```

## Typehints
Use typehints where possible. Prefer python 3.10 style `list[Foo]`, `dict[Foo]` etc. except for Optional[Foo], which should be used instead of Foo | None for readability.

# GitHub
List issues with `gh issue list`, and open PRs with `gh pr create`. Each PR should be composed of atomic, sequential commits. PR descriptions should contain a concise summary of how the changes solved the issue, and reference issues with "closes #id". When asked about a specific issue like "#26", run `gh issue list` to see an overview of all open issues, and `gh issue view 26` to view that particular issue. PRs should show commits that make sense to a future reader, so collapse commits that involve multiple tweaks of the same code from feedback during the PR into a single commit.

# Linting
Use `ruff check` to lint your changes.

---

## Project description

This repo contains the **omgrab SDK** — a PyPI-publishable Python package (`pip install omgrab`, `import omgrab`) that provides the core runtime for **egocentric data collection** on the **OMGrab streaming device**. A user presses a physical **button** to start/stop recording; the device writes **minute-chunked clips** to disk and merges them into a single MKV on stop. We support multiple stream types; by default we use an **OAK-D Wide** (or **OAK-D Pro Wide**) and capture **time-synced RGB + depth** frames, plus **IMU data** encoded as a subtitle track.

The `app/` directory is the **runtime application** built on the SDK that adds Docker-based WiFi setup. The SDK itself has no dependency on Docker, cloud APIs, or upload infrastructure.

## Docker / Docker Compose (runtime)

- **Container entrypoint**: `Dockerfile` runs `python -u -m app`.
- **Compose stack**: `docker-compose.yml` defines `omgrab-runtime` (privileged + device mounts + `/opt/omgrab/data:/data`) and a `wifi-connect` helper container (host networking) that the runtime can start via `/var/run/docker.sock` during long-press WiFi setup.
- **System service**: `service/omgrab.service` starts the stack via `docker compose up -d` in `/opt/omgrab` (where the installed compose file + persistent volumes live).
- **Samba share**: The installer sets up a Samba share (`smb://omgrab.local/omgrab`) exposing the recordings output directory. Runs on the host (not in Docker). Configured via `service/smb-omgrab.conf`.

## Top-level architecture (objects + ownership)

At runtime, `app/__main__.py` wires together a small set of "owners" and background loops:

- **`StateMachine` (central coordinator)** (`src/omgrab/workflows/state_machine.py`)
  - Owns the high-level workflow state: `BOOT → IDLE ↔ RECORDING`, plus `WIFI_SETUP` and `SHUTDOWN`.
  - Owns GPIO side-effects (LED/buzzer) via `GPIOController` (`src/omgrab/gpio/gpio.py`).
  - Delegates recording to `RecordingManager`.

- **Capture devices (physical hardware)** (`src/omgrab/devices/`)
  - `CaptureDevice` protocol (`capture_device.py`) defines the interface for all physical devices.
  - `OakDCaptureDevice` (`oakd_capture_device.py`) — primary egocentric device. Owns the DepthAI pipeline and produces synchronized RGB+depth frames. Provides IMU and preview support.
  - `USBCaptureDevice` (`usb_capture_device.py`) — wraps a single USB camera identified by port path. Used for auxiliary wrist cameras.
  - Devices represent the physical hardware and can be powered on/off independently of recordings.

- **Cameras (per-recording session)** (`src/omgrab/cameras/`)
  - `Camera` base class (`cameras.py`) defines the frame-reading interface.
  - `OakDRGBCamera` / `OakDDepthCamera` (`oakd_camera.py`) — lightweight queue readers created per recording from `OakDCaptureDevice`.
  - `USBCamera` (`usb_camera.py`) — reads frames from a USB camera via OpenCV.
  - Cameras are started and stopped each time the user begins/ends a recording; the underlying device may outlive many camera instances.

- **`RecordingManager` (recording lifecycle)** (`src/omgrab/runtime/recording_manager.py`)
  - Owns the capture device (opens/closes per recording).
  - Owns active and finished `RecordingSession` instances.
  - Generates timestamp-based recording names (e.g., `2026-03-17T00-47-42Z`).
  - Produces sequential chunk names (`00001`, `00002`, etc.).
  - Merges chunks into a single MKV on stop via ffmpeg concat.
  - Reports device-unhealthy events to state machine via `set_on_device_unhealthy` callback.

- **`RecordingSession` (per-recording threads)** (`src/omgrab/runtime/recording_session.py`)
  - Owns an isolated frame queue for one recording.
  - Owns capture threads (one per camera) and writer thread.
  - Uses `ChunkedWriter` for duration-chunked MKV output with optional data streams (IMU).

- **Network monitor** (`src/omgrab/runtime/network_monitor.py`)
  - Runs a background thread that probes local connectivity and internet reachability.
  - States: `OFFLINE` / `NETWORK_ONLY` (local network, no internet) / `ONLINE` (internet reachable).
  - Broadcasts stable state changes to registered callbacks.

- **Device status manager** (`src/omgrab/runtime/device_status.py`)
  - Centralized aggregator for all device state, system metrics, and operational status.
  - Queries system resources (CPU temp/usage, memory, disk space) from `/proc` and `/sys`.
  - Queries operational state from state machine, recording manager.
  - Provides structured status snapshots for the screen manager.

- **Screen manager** (`src/omgrab/display/screen_manager.py`)
  - Drives the OLED status dashboard: polls `DeviceStatusManager` and renders device state to a 1-bit I2C display.
  - Manages live camera preview: receives frames from the capture device's preview camera, dithers to 1-bit, and renders to the OLED.
  - Receives push notifications (alerts, WiFi status) from the state machine with priority ordering.

- **Button/WiFi loop** (`src/omgrab/runtime/gpio_manager.py`, `src/omgrab/runtime/wifi_connect.py`)
  - Polls the GPIO button and calls into `StateMachine` for start/stop or WiFi provisioning.
  - `WifiManager` class takes injectable container management functions (Docker helpers in `app/docker_wifi.py`).

## Recording flow (button → frames → chunks → merged MKV)

### 1) User input → state transition

- The **button monitor** polls GPIO; on short press it calls `state_machine.handle_button_press()`.
- `StateMachine` toggles between IDLE → RECORDING → IDLE.

### 2) RecordingManager starts a recording session

1. Opens the capture device and waits until ready
2. Flushes any stale frames from camera queues
3. Generates a timestamp-based recording name (e.g., `2026-03-17T00-47-42Z`)
4. Creates a recording subdirectory `{spool_dir}/{recording_name}/`
5. Creates a new `RecordingSession` with sequential chunk naming
6. Starts capture threads (one per camera) and writer thread

### 3) Stop recording → merge chunks

1. Capture threads stop, device closes
2. All chunks in `{spool_dir}/{recording_name}/` are merged into `{output_dir}/{recording_name}.mkv`
3. Merge uses `ffmpeg -f concat -safe 0 -c copy` (lossless, fast)
4. Chunk directory is cleaned up after successful merge

### 4) Boot-time recovery

On startup, `merge_orphaned_chunks(spool_dir, output_dir)`:
- Scans for recording directories with finalized chunks but no merged output → merges them
- Cleans up `.tmp` files from crashed writes
- Removes empty recording directories
