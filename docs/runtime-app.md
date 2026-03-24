# Device Runtime

The `app/` directory contains the runtime application that wires the SDK components together into the full omgrab device runtime. It is deployed as a Docker container and is built using the [OMGrab SDK](index.md).

## Architecture

The application entry point (`app/__main__.py`) creates and connects:

1. **Capture device** -- an `OakDCaptureDevice` producing synchronized RGB + depth frames and IMU data
2. **Recording manager** -- manages recording sessions, chunk naming, and merge-on-stop
3. **State machine** -- coordinates BOOT → IDLE ↔ RECORDING transitions, plus WIFI_SETUP and SHUTDOWN
4. **GPIO controller** -- drives LEDs and buzzer based on state machine transitions
5. **Button monitor** -- polls the physical button and dispatches short-press (record toggle) and long-press (WiFi setup) events
6. **Network monitor** -- probes connectivity and broadcasts OFFLINE / NETWORK_ONLY / ONLINE state changes
7. **Device status manager** -- aggregates system metrics (CPU, memory, disk, temperature) and operational state
8. **Screen manager** -- renders status and live camera preview to the OLED display

On startup, the app also runs `merge_orphaned_chunks()` to recover any recordings left incomplete by a prior crash.

## Configuration

All configuration is via environment variables, set in `docker-compose.yml`:

| Variable | Default | Description |
|---|---|---|
| `DEVICE_ID` | `unknown-device` | Unique identifier for this device |
| `AGENT_VERSION` | `v0.0.1` | Software version (set at build time) |
| `WRIST_CAMERAS_ENABLED` | `false` | Enable auxiliary USB wrist cameras |
| `WIFI_CONNECT_IMAGE` | `wifi-connect:latest` | Docker image for the WiFi setup container |
| `DEPTHAI_LEVEL` | `warn` | DepthAI SDK log level |

### Paths

| Path (in container) | Host mount | Purpose |
|---|---|---|
| `/data/spool/capture/` | `/opt/omgrab/data/spool/capture/` | In-progress recording chunks |
| `/data/spool/output/` | `/opt/omgrab/data/spool/output/` | Merged MKV output files |
| `/var/run/docker.sock` | `/var/run/docker.sock` | Docker socket for WiFi container management |
| `/button-trigger` | `/opt/omgrab/button-trigger` | Remote button simulation (touch a file to trigger) |

## Stream configuration

The app configures the following streams per recording:

| Stream | Codec | Resolution | FPS | Notes |
|---|---|---|---|---|
| `egocentric_rgb` | H.264 (libx264) | 1280x800 | 25 | Baseline profile, superfast preset, 4 Mbps |
| `egocentric_depth` | FFV1 (lossless) | 640x400 | 25 | 16-bit grayscale |
| `imu` | ASS subtitle | -- | 100 Hz | JSON-encoded IMU samples |
| `left_wrist` | H.264 | 1280x720 | 15 | Only when `WRIST_CAMERAS_ENABLED=true` |
| `right_wrist` | H.264 | 1280x720 | 15 | Only when `WRIST_CAMERAS_ENABLED=true` |

## Installation

The installer (`setup/install.sh`) provisions a Raspberry Pi from scratch. It must be run as root on an ARM64 device:

```bash
sudo ./setup/install.sh --device-id my-device-001
```

The installer performs the following steps:

1. Installs Docker and Docker Compose if not present
2. Creates `/opt/omgrab/` with data directories and a `.env` file containing the device ID
3. Configures persistent journald logging (200 MB cap)
4. Enables I2C (required for the battery monitor on `/dev/i2c-1`)
5. Installs the early-led boot indicator service (see [Services](#services) below)
6. Installs WiFi stability fixes (NetworkManager dispatcher script)
7. Sets up a Samba file share at `smb://<hostname>.local/omgrab` exposing the recordings output directory (user: `omgrab`, password: `omgrab`)
8. Builds the Docker images locally from the current source tree
9. Installs and enables the `omgrab.service` systemd unit
10. Optionally starts the service immediately

### Uninstallation

The runtime can be uninstalled using:

```bash
sudo ./setup/uninstall.sh
```

This stops the service, removes systemd units, tears down Docker containers and images, and removes the Samba share. Optionally deletes `/opt/omgrab/` (including all recorded data) if confirmed interactively.

## Services

Several systemd services are used to power the runtime:

### omgrab.service

The primary systemd unit. Runs the Docker Compose stack and restarts automatically on failure (up to 5 times per 60 seconds).

- **Starts after**: `docker.service`, `systemd-udev-settle.service`, `NetworkManager.service`
- **Working directory**: `/opt/omgrab`
- **Start command**: `docker compose up --abort-on-container-exit --remove-orphans`
- **Stop command**: `docker compose down`
- **Restart policy**: always, 5 second delay

Before starting, the service stops the early-led indicator and loads the `uvcvideo` kernel module (for USB cameras).

Useful commands:

```bash
sudo systemctl start omgrab        # Start the runtime
sudo systemctl stop omgrab         # Stop the runtime
sudo systemctl status omgrab       # Check status
sudo journalctl -u omgrab -f       # Follow logs
sudo journalctl -u omgrab --boot=-1  # Logs from previous boot
```

### early-led.service

A minimal boot indicator that blinks the green LED (GPIO 17) at ~5 Hz immediately after the root filesystem is mounted. This gives the user visual feedback that the device is booting before Docker and the main application are ready. The omgrab service stops it on startup and takes over LED control via the GPIO controller.

- **Starts at**: `sysinit.target` (very early boot, before most services)
- **Stopped by**: `omgrab.service` via `ExecStartPre`

### Samba share

Not a custom service -- uses the system `smbd`. The installer adds an `[omgrab]` share definition pointing to `/opt/omgrab/data/spool/output/` so recordings can be accessed over the local network at `smb://<hostname>.local/omgrab`.

## Deployment

The compose stack defines two containers:

- **omgrab-runtime** -- the main application, running privileged with host networking, GPIO chip access (`/dev/gpiochip0`), USB device access, and the Docker socket mounted for managing the WiFi container
- **wifi-connect** -- a separate Rust-based container (built from `wifi-connect/`) that creates a captive portal hotspot for WiFi provisioning. Not started by default -- launched on-demand by the runtime via Docker socket when the user long-presses the button.
This container is based on the [wifi-connect](https://github.com/balena-os/wifi-connect) project by BalenaOS.

## Shutdown

On `SIGINT` or `SIGTERM`, the app performs a graceful shutdown sequence:

1. State machine transitions to SHUTDOWN (LEDs indicate shutdown)
2. Active recording is stopped and chunks are merged
3. Network monitor, screen manager, and device status manager are stopped
4. Process exits
