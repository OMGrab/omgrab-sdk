"""Tests for the network monitor (runtime/network_monitor.py)."""
import time

import pytest

from omgrab.runtime import network_monitor


class TestStatusEnum:

    def test_all_statuses_have_ranks(self):
        """Every Status enum value must have a rank."""
        for status in network_monitor.Status:
            assert status in network_monitor.RANK

    def test_rank_ordering(self):
        """Ranks should increase: OFFLINE < NETWORK_ONLY < ONLINE."""
        assert (network_monitor.RANK[network_monitor.Status.OFFLINE]
                < network_monitor.RANK[network_monitor.Status.NETWORK_ONLY]
                < network_monitor.RANK[network_monitor.Status.ONLINE])

    @pytest.mark.parametrize('status,value', [
        (network_monitor.Status.OFFLINE, 'offline'),
        (network_monitor.Status.NETWORK_ONLY, 'network_only'),
        (network_monitor.Status.ONLINE, 'online'),
    ])
    def test_status_values(self, status, value):
        """Status enum string values should match expected names."""
        assert status.value == value



class TestSnapshot:

    def test_snapshot_is_frozen(self):
        """Snapshot should be immutable (frozen dataclass)."""
        snap = network_monitor.Snapshot(
            status=network_monitor.Status.ONLINE,
            detail='test',
            changed_at=time.monotonic(),
        )

        with pytest.raises(AttributeError):
            snap.status = network_monitor.Status.OFFLINE

    def test_snapshot_stores_all_fields(self):
        """Snapshot should expose status, detail, and changed_at."""
        now = time.monotonic()
        snap = network_monitor.Snapshot(
            status=network_monitor.Status.NETWORK_ONLY,
            detail='dns_fail',
            changed_at=now,
        )

        assert snap.status == network_monitor.Status.NETWORK_ONLY
        assert snap.detail == 'dns_fail'
        assert snap.changed_at == now



class TestConfig:

    def test_default_config_values(self):
        """Config defaults should be reasonable."""
        cfg = network_monitor.Config()

        assert cfg.down_after_failures == 3
        assert cfg.up_after_successes == 1
        assert cfg.poll_ok_s == 5.0
        assert cfg.poll_min_s == 1.0
        assert cfg.poll_max_s == 20.0
        assert cfg.internet_check_host == 'dns.google'
        assert cfg.internet_check_port == 443



class TestHasDefaultRoute:

    def test_returns_truthy_when_route_exists(self, monkeypatch):
        """Should return truthy when 'ip route show default' has output."""
        monkeypatch.setattr(
            network_monitor, '_run',
            lambda cmd: (0, 'default via 192.168.1.1 dev wlan0'),
        )

        assert network_monitor._has_default_route()

    def test_returns_falsy_when_no_route(self, monkeypatch):
        """Should return falsy when there is no default route."""
        monkeypatch.setattr(network_monitor, '_run', lambda cmd: (0, ''))

        assert not network_monitor._has_default_route()

    def test_returns_falsy_on_command_failure(self, monkeypatch):
        """Should return falsy when the ip command fails."""
        monkeypatch.setattr(network_monitor, '_run', lambda cmd: (1, 'error'))

        assert not network_monitor._has_default_route()


class TestHasIpv4Addr:

    def test_returns_true_for_normal_ip(self, monkeypatch):
        """Should return True for a non-loopback, non-link-local IPv4 address."""
        monkeypatch.setattr(
            network_monitor, '_run',
            lambda cmd: (0, '    inet 192.168.1.10/24 brd 192.168.1.255 scope global wlan0'),
        )

        assert network_monitor._has_ipv4_addr() is True

    def test_returns_false_for_loopback(self, monkeypatch):
        """Should return False if only loopback address is found."""
        monkeypatch.setattr(
            network_monitor, '_run',
            lambda cmd: (0, '    inet 127.0.0.1/8 scope host lo'),
        )

        assert network_monitor._has_ipv4_addr() is False

    def test_returns_false_for_link_local(self, monkeypatch):
        """Should return False for link-local (169.254.x.x) addresses."""
        monkeypatch.setattr(
            network_monitor, '_run',
            lambda cmd: (0, '    inet 169.254.10.20/16 scope link wlan0'),
        )

        assert network_monitor._has_ipv4_addr() is False

    def test_returns_false_on_command_failure(self, monkeypatch):
        """Should return False when ip addr fails."""
        monkeypatch.setattr(network_monitor, '_run', lambda cmd: (1, ''))

        assert network_monitor._has_ipv4_addr() is False


class TestLocalOk:

    def test_true_when_both_checks_pass(self, monkeypatch):
        """Should return True when both default route and IPv4 address exist."""
        monkeypatch.setattr(network_monitor, '_has_default_route', lambda: True)
        monkeypatch.setattr(network_monitor, '_has_ipv4_addr', lambda iface=None: True)

        assert network_monitor._local_ok(None) is True

    def test_false_when_no_default_route(self, monkeypatch):
        """Should return False without a default route."""
        monkeypatch.setattr(network_monitor, '_has_default_route', lambda: False)
        monkeypatch.setattr(network_monitor, '_has_ipv4_addr', lambda iface=None: True)

        assert network_monitor._local_ok(None) is False

    def test_false_when_no_ipv4(self, monkeypatch):
        """Should return False without an IPv4 address."""
        monkeypatch.setattr(network_monitor, '_has_default_route', lambda: True)
        monkeypatch.setattr(network_monitor, '_has_ipv4_addr', lambda iface=None: False)

        assert network_monitor._local_ok(None) is False



def _make_monitor(*, down_after_failures=3, up_after_successes=1) -> network_monitor.NetworkMonitor:
    """Create a NetworkMonitor with test-friendly config."""
    cfg = network_monitor.Config(
        down_after_failures=down_after_failures,
        up_after_successes=up_after_successes,
        poll_ok_s=0.1,
        poll_min_s=0.05,
        poll_max_s=1.0,
    )
    return network_monitor.NetworkMonitor(cfg)


class TestConsiderHysteresis:

    def test_same_status_resets_candidate(self):
        """Observing the current stable status should reset the candidate."""
        mon = _make_monitor()
        # Initial status is OFFLINE.
        mon._consider(network_monitor.Status.OFFLINE, 'same')

        assert mon._candidate is None
        assert mon._cand_succ == 0

    def test_upgrade_with_single_success(self):
        """With up_after_successes=1, one observation should promote stable."""
        mon = _make_monitor(up_after_successes=1)
        # Initial status is OFFLINE. Observe ONLINE.
        mon._consider(network_monitor.Status.ONLINE, 'ok')

        assert mon.snapshot.status == network_monitor.Status.ONLINE

    def test_downgrade_requires_multiple_failures(self):
        """Downgrade should require down_after_failures consecutive observations."""
        mon = _make_monitor(down_after_failures=3)
        # Move to ONLINE first.
        mon._set_stable(network_monitor.Status.ONLINE, 'ok')

        # 1st and 2nd failure — should stay at ONLINE.
        mon._consider(network_monitor.Status.OFFLINE, 'fail1')
        assert mon.snapshot.status == network_monitor.Status.ONLINE

        mon._consider(network_monitor.Status.OFFLINE, 'fail2')
        assert mon.snapshot.status == network_monitor.Status.ONLINE

        # 3rd failure — should transition to OFFLINE.
        mon._consider(network_monitor.Status.OFFLINE, 'fail3')
        assert mon.snapshot.status == network_monitor.Status.OFFLINE

    def test_candidate_resets_on_different_observed(self):
        """If observed status changes from candidate, counter should reset."""
        mon = _make_monitor(down_after_failures=3)
        mon._set_stable(network_monitor.Status.ONLINE, 'ok')

        mon._consider(network_monitor.Status.OFFLINE, 'fail1')
        mon._consider(network_monitor.Status.OFFLINE, 'fail2')
        # Interrupt with a different status.
        mon._consider(network_monitor.Status.NETWORK_ONLY, 'different')

        # Counter for OFFLINE should have been reset.
        assert mon._candidate == network_monitor.Status.NETWORK_ONLY
        assert mon._cand_fail == 1

    def test_backoff_increases_on_downgrade(self):
        """Backoff should increase when transitioning to a worse state."""
        mon = _make_monitor(down_after_failures=1)
        mon._set_stable(network_monitor.Status.ONLINE, 'ok')
        initial_backoff = mon._backoff_s

        mon._consider(network_monitor.Status.OFFLINE, 'fail')

        assert mon._backoff_s > initial_backoff

    def test_backoff_resets_on_online(self):
        """Reaching ONLINE should reset backoff to poll_ok_s."""
        mon = _make_monitor(up_after_successes=1)
        mon._backoff_s = 20.0

        mon._consider(network_monitor.Status.ONLINE, 'ok')

        assert mon._backoff_s == mon.cfg.poll_ok_s



class TestCheckOnce:

    def test_offline_when_no_local_connectivity(self, monkeypatch):
        """check_once should return OFFLINE when local network is down."""
        monkeypatch.setattr(network_monitor, '_local_ok', lambda iface: False)

        mon = _make_monitor()
        status, detail = mon.check_once()

        assert status == network_monitor.Status.OFFLINE
        assert 'no_ip' in detail

    def test_network_only_when_dns_fails(self, monkeypatch):
        """check_once should return NETWORK_ONLY when DNS resolution fails."""
        monkeypatch.setattr(network_monitor, '_local_ok', lambda iface: True)
        monkeypatch.setattr(
            network_monitor, '_resolve',
            lambda host, port, timeout_s: (_ for _ in ()).throw(OSError('DNS fail')),
        )

        mon = _make_monitor()
        status, detail = mon.check_once()

        assert status == network_monitor.Status.NETWORK_ONLY
        assert 'dns_fail' in detail

    def test_network_only_when_tcp_fails(self, monkeypatch):
        """check_once should return NETWORK_ONLY when TCP connection fails."""
        monkeypatch.setattr(network_monitor, '_local_ok', lambda iface: True)
        monkeypatch.setattr(
            network_monitor, '_resolve',
            lambda host, port, timeout_s: [('AF_INET', 'SOCK_STREAM', 0, '', ('1.2.3.4', 443))],
        )
        monkeypatch.setattr(
            network_monitor, '_tcp_ok',
            lambda addrs, timeout_s: (_ for _ in ()).throw(OSError('TCP fail')),
        )

        mon = _make_monitor()
        status, detail = mon.check_once()

        assert status == network_monitor.Status.NETWORK_ONLY
        assert 'tcp_fail' in detail

    def test_online_when_all_checks_pass(self, monkeypatch):
        """check_once should return ONLINE when DNS+TCP succeed."""
        monkeypatch.setattr(network_monitor, '_local_ok', lambda iface: True)
        monkeypatch.setattr(
            network_monitor, '_resolve',
            lambda host, port, timeout_s: [],
        )
        monkeypatch.setattr(network_monitor, '_tcp_ok', lambda addrs, timeout_s: True)

        mon = _make_monitor()
        status, detail = mon.check_once()

        assert status == network_monitor.Status.ONLINE
        assert detail == 'ok'



class TestNetworkMonitorLifecycle:

    def test_initial_snapshot_is_offline(self):
        """Monitor should start in OFFLINE state."""
        mon = _make_monitor()
        assert mon.snapshot.status == network_monitor.Status.OFFLINE

    def test_register_callback_before_start(self):
        """Callbacks should be registerable before start."""
        mon = _make_monitor()
        mon.register_on_change_callback(lambda snap: None)

    def test_register_callback_after_start_raises(self):
        """Registering a callback after start should raise RuntimeError."""
        mon = _make_monitor()
        mon.start()
        try:
            with pytest.raises(RuntimeError, match='Cannot register'):
                mon.register_on_change_callback(lambda snap: None)
        finally:
            mon.shutdown()

    def test_callbacks_fire_on_state_change(self, monkeypatch):
        """Registered callbacks should be called when stable state changes."""
        received: list[network_monitor.Snapshot] = []

        # Make check_once return ONLINE immediately.
        monkeypatch.setattr(network_monitor, '_local_ok', lambda iface: True)
        monkeypatch.setattr(
            network_monitor, '_resolve',
            lambda host, port, timeout_s: [],
        )
        monkeypatch.setattr(network_monitor, '_tcp_ok', lambda addrs, timeout_s: True)

        cfg = network_monitor.Config(
            up_after_successes=1,
            poll_ok_s=0.05,
            poll_min_s=0.05,
        )
        mon = network_monitor.NetworkMonitor(cfg)
        mon.register_on_change_callback(lambda snap: received.append(snap))
        mon.start()

        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if received:
                    break
                time.sleep(0.05)

            assert len(received) >= 1
            assert received[0].status == network_monitor.Status.ONLINE
        finally:
            mon.shutdown()

    def test_wake_interrupts_sleep(self):
        """wake() should cause the monitor to check sooner."""
        mon = _make_monitor()
        # Just verify it doesn't crash (wake sets the event).
        mon.wake()

    def test_shutdown_stops_thread(self, monkeypatch):
        """shutdown() should stop the monitor thread."""
        monkeypatch.setattr(network_monitor, '_local_ok', lambda iface: False)

        mon = _make_monitor()
        mon.start()
        mon.shutdown()

        assert mon._thread is None or not mon._thread.is_alive()

    def test_start_is_idempotent(self, monkeypatch):
        """Calling start() twice should not create a second thread."""
        monkeypatch.setattr(network_monitor, '_local_ok', lambda iface: False)

        mon = _make_monitor()
        mon.start()
        first_thread = mon._thread
        mon.start()

        assert mon._thread is first_thread
        mon.shutdown()

    def test_callback_exception_does_not_crash_loop(self, monkeypatch):
        """Callback that raises should be caught; loop should continue."""
        callback_invoked = []

        # Make check_once alternate between OFFLINE and ONLINE.
        call_count = [0]

        def _alternating_check(self_mon):
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                return network_monitor.Status.ONLINE, 'ok'
            return network_monitor.Status.OFFLINE, 'no_ip'

        monkeypatch.setattr(
            network_monitor.NetworkMonitor, 'check_once', _alternating_check)

        def _failing_callback(snap):
            callback_invoked.append(snap.status)
            raise RuntimeError('callback error')

        cfg = network_monitor.Config(
            up_after_successes=1,
            down_after_failures=1,
            poll_ok_s=0.05,
            poll_min_s=0.05,
        )
        mon = network_monitor.NetworkMonitor(cfg)
        mon.register_on_change_callback(_failing_callback)
        mon.start()

        try:
            deadline = time.monotonic() + 2.0
            while time.monotonic() < deadline:
                if len(callback_invoked) >= 2:
                    break
                time.sleep(0.05)

            # Callback should have been called at least twice (loop didn't die).
            assert len(callback_invoked) >= 2
        finally:
            mon.shutdown()



class TestRunFunction:

    def test_file_not_found_returns_127(self, monkeypatch):
        """FileNotFoundError from subprocess should return code 127."""
        import subprocess
        monkeypatch.setattr(
            subprocess, 'run',
            lambda cmd, **kw: (_ for _ in ()).throw(
                FileNotFoundError('no such command')),
        )

        rc, out = network_monitor._run(['nonexistent_cmd'])

        assert rc == 127
        assert 'not_found' in out

    def test_os_error_returns_126(self, monkeypatch):
        """OSError from subprocess should return code 126."""
        import subprocess
        monkeypatch.setattr(
            subprocess, 'run',
            lambda cmd, **kw: (_ for _ in ()).throw(
                PermissionError('permission denied')),
        )

        rc, out = network_monitor._run(['some_cmd'])

        assert rc == 126
        assert 'os_error' in out



class TestHasIpv4AddrWithIface:

    def test_passes_iface_to_command(self, monkeypatch):
        """_has_ipv4_addr should include 'dev <iface>' when iface is set."""
        captured_cmds = []

        def _capture_run(cmd):
            captured_cmds.append(cmd)
            return 0, '    inet 10.0.0.1/24 scope global wlan0'

        monkeypatch.setattr(network_monitor, '_run', _capture_run)

        result = network_monitor._has_ipv4_addr(iface='wlan0')

        assert result is True
        assert 'dev' in captured_cmds[0]
        assert 'wlan0' in captured_cmds[0]
