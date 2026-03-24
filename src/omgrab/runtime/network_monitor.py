"""Network connectivity monitor with hysteresis and backoff.

Simplified for SDK use: checks local network and internet reachability.
No heartbeats or API probes.
"""
from typing import Optional

import dataclasses
import enum
import logging
import random
import socket
import subprocess
import threading
import time
from collections.abc import Callable

logger = logging.getLogger(__name__)


class Status(enum.Enum):
    """Discrete network health states in increasing order of quality."""
    OFFLINE = 'offline'
    NETWORK_ONLY = 'network_only'
    ONLINE = 'online'


RANK = {
    Status.OFFLINE: 0,
    Status.NETWORK_ONLY: 1,
    Status.ONLINE: 2,
}


@dataclasses.dataclass(frozen=True)
class Config:
    """Configuration values controlling probes, hysteresis, and backoff.

    Attributes:
        dns_timeout: Timeout in seconds for DNS resolution probes.
        tcp_timeout: Timeout in seconds for TCP connection probes.
        internet_check_host: Hostname used for internet reachability checks.
        internet_check_port: TCP port used for internet reachability checks.
        down_after_failures: Consecutive probe failures before transitioning down.
        up_after_successes: Consecutive probe successes before transitioning up.
        poll_ok_s: Polling interval in seconds when the network is healthy.
        poll_min_s: Minimum polling interval in seconds during backoff.
        poll_max_s: Maximum polling interval in seconds during backoff.
        jitter_frac: Fraction of the polling interval added as random jitter.
    """

    dns_timeout: float = 1.0
    tcp_timeout: float = 1.0
    internet_check_host: str = 'dns.google'
    internet_check_port: int = 443
    down_after_failures: int = 3
    up_after_successes: int = 1
    poll_ok_s: float = 5.0
    poll_min_s: float = 1.0
    poll_max_s: float = 20.0
    jitter_frac: float = 0.1


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """Immutable snapshot of the last stable network state.

    Attributes:
        status: Current network health level.
        detail: Human-readable description of the last probe result.
        changed_at: Monotonic timestamp when the status last changed.
    """

    status: Status
    detail: str
    changed_at: float


def _run(cmd: list[str]) -> tuple[int, str]:
    """Run a subprocess command and return (returncode, combined_output)."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
        out = ((p.stdout or '') + (p.stderr or '')).strip()
        return p.returncode, out
    except FileNotFoundError:
        return 127, f'not_found:{cmd[0]}'
    except OSError as e:
        return 126, f'os_error:{type(e).__name__}'


def _has_default_route() -> bool:
    """Return True if a default route exists."""
    rc, out = _run(['ip', 'route', 'show', 'default'])
    return rc == 0 and bool(out)


def _has_ipv4_addr(iface: Optional[str] = None) -> bool:
    """Return True if a non-loopback, non-link-local IPv4 address is present."""
    cmd = ['ip', '-4', 'addr', 'show']
    if iface:
        cmd += ['dev', iface]
    rc, out = _run(cmd)
    if rc != 0:
        return False

    for line in out.splitlines():
        line = line.strip()
        if line.startswith('inet '):
            ip = line.split()[1].split('/')[0]
            if ip.startswith(('127.', '169.254.')):
                continue
            return True
    return False


def _local_ok(iface: Optional[str]) -> bool:
    """Return True if the host has an IP and a default route."""
    return _has_default_route() and _has_ipv4_addr(iface)


def _resolve(host: str, port: int, timeout_s: float) -> list[tuple]:
    """Resolve a hostname to socket addresses with a temporary timeout.

    Args:
        host: Hostname to resolve.
        port: Port number for resolution.
        timeout_s: DNS resolution timeout in seconds.

    Returns:
        A list of socket address tuples.

    Raises:
        socket.gaierror: If DNS resolution fails.
    """
    prev = socket.getdefaulttimeout()
    try:
        socket.setdefaulttimeout(timeout_s)
        return socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    finally:
        socket.setdefaulttimeout(prev)


def _tcp_ok(addrs: list[tuple], timeout_s: float) -> bool:
    """Attempt a TCP connect to any resolved address.

    Args:
        addrs: Address tuples from getaddrinfo.
        timeout_s: TCP connect timeout in seconds.

    Returns:
        True if any address connects successfully.

    Raises:
        OSError: If all connection attempts fail.
    """
    last_err: Optional[OSError] = None
    for family, socktype, proto, _canon, sockaddr in addrs:
        try:
            with socket.socket(family, socktype, proto) as s:
                s.settimeout(timeout_s)
                s.connect(sockaddr)
                return True
        except OSError as e:
            last_err = e
    if last_err:
        raise last_err
    return False


class NetworkMonitor:
    """Stateful network monitor with hysteresis and backoff."""

    def __init__(self, cfg: Config, iface: Optional[str] = None):
        """Initialize the monitor with configuration and optional interface pinning.

        Args:
            cfg: Configuration for network monitoring.
            iface: Optional network interface to monitor.
        """
        self.cfg = cfg
        self.iface = iface

        self._stable = Snapshot(Status.OFFLINE, 'init', time.monotonic())
        self._candidate: Optional[Status] = None
        self._cand_succ = 0
        self._cand_fail = 0
        self._backoff_s = cfg.poll_min_s
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._on_change_callbacks: list[Callable[[Snapshot], None]] = []
        self._started = False

    def wake(self):
        """Wake the monitor loop early."""
        self._wake_event.set()

    def register_on_change_callback(self, on_network_change: Callable[[Snapshot], None]):
        """Register a callback to be called on stable network state changes.

        Note: Callbacks must be registered before the monitor is started.

        Args:
            on_network_change: Callback function to register.

        Raises:
            RuntimeError: If the monitor has already started.
        """
        if self._started:
            raise RuntimeError('Cannot register callbacks after NetworkMonitor has started')
        self._on_change_callbacks.append(on_network_change)

    def start(self):
        """Start the monitor thread."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def shutdown(self):
        """Stop the monitor thread."""
        self._stop_event.set()
        self._wake_event.set()  # Wake the loop so it can observe stop_event quickly.
        if self._thread:
            self._thread.join(timeout=5.0)

    @property
    def snapshot(self) -> Snapshot:
        """Return the current stable snapshot."""
        return self._stable

    def _set_stable(self, status: Status, detail: str):
        """Update the stable snapshot if the state or detail changed."""
        if status != self._stable.status or detail != self._stable.detail:
            self._stable = Snapshot(status, detail, time.monotonic())

    def _reset_candidate(self):
        """Reset candidate transition tracking."""
        self._candidate = None
        self._cand_succ = 0
        self._cand_fail = 0

    def _consider(self, observed: Status, detail: str):
        """Apply hysteresis and backoff to an observed state."""
        cfg = self.cfg
        stable = self._stable.status

        if observed == stable:
            self._reset_candidate()
            self._backoff_s = cfg.poll_ok_s if observed == Status.ONLINE else cfg.poll_min_s
            self._set_stable(observed, detail)
            return

        moving_up = RANK[observed] > RANK[stable]
        thresh = cfg.up_after_successes if moving_up else cfg.down_after_failures

        if self._candidate != observed:
            self._candidate = observed
            self._cand_succ = 0
            self._cand_fail = 0

        if moving_up:
            self._cand_succ += 1
            if self._cand_succ >= thresh:
                self._set_stable(observed, detail)
                self._reset_candidate()
                self._backoff_s = cfg.poll_ok_s if observed == Status.ONLINE else cfg.poll_min_s
        else:
            self._cand_fail += 1
            if self._cand_fail >= thresh:
                self._set_stable(observed, detail)
                self._reset_candidate()
                self._backoff_s = min(cfg.poll_max_s, max(cfg.poll_min_s, self._backoff_s * 2))

    def check_once(self) -> tuple[Status, str]:
        """Run a connectivity check: local network, then internet reachability."""
        cfg = self.cfg

        # Check if the local network is OK.
        if not _local_ok(self.iface):
            return Status.OFFLINE, 'no_ip_or_default_route'

        # Try to resolve and connect to the internet check host.
        try:
            addrs = _resolve(cfg.internet_check_host, cfg.internet_check_port, cfg.dns_timeout)
        except Exception as e:
            return Status.NETWORK_ONLY, f'dns_fail:{type(e).__name__}'

        try:
            _tcp_ok(addrs, cfg.tcp_timeout)
        except Exception as e:
            return Status.NETWORK_ONLY, f'tcp_fail:{type(e).__name__}'

        return Status.ONLINE, 'ok'

    def _run_loop(self):
        """Continuously monitor and invoke registered callbacks on stable state changes."""
        last = self._stable.status
        while not self._stop_event.is_set():
            self._wake_event.clear()

            observed, detail = self.check_once()
            self._consider(observed, detail)

            snap = self.snapshot
            if snap.status != last:
                last = snap.status
                for callback in self._on_change_callbacks:
                    try:
                        callback(snap)
                    except Exception as e:
                        logger.error('Callback error: %s', e)

            base_delay_s = self._backoff_s
            jitter_magnitude_s = base_delay_s * self.cfg.jitter_frac
            delay_s = base_delay_s + random.uniform(-jitter_magnitude_s, jitter_magnitude_s)
            delay_s = max(0.5, delay_s)  # Minimum delay of 0.5 seconds.
            # Wait for either a wake signal or the calculated delay.
            self._wake_event.wait(timeout=delay_s)
