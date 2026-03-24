"""Docker socket helpers for managing the wifi-connect container."""
from typing import Optional

import json
import os
import socket
import threading
import urllib.parse
from collections.abc import Callable

CONTAINER_NAME = 'wifi-connect'
DOCKER_SOCKET = '/var/run/docker.sock'
WIFI_CONNECT_IMAGE = os.environ.get(
    'WIFI_CONNECT_IMAGE',
    'wifi-connect:latest'
)


def _send_request(sock: socket.socket, method: str, path: str,
                  body: dict | None = None):
    """Format and send an HTTP request over a socket."""
    body_str = json.dumps(body) if body else ''
    lines = [
        f'{method} {path} HTTP/1.1',
        'Host: localhost',
        'Content-Type: application/json',
    ]
    if body_str:
        lines.append(f'Content-Length: {len(body_str)}')
    lines.append('')
    lines.append(body_str)
    sock.sendall('\r\n'.join(lines).encode())


def _parse_headers(raw: bytes) -> tuple[int, bool]:
    """Extract content-length and chunked flag from raw header bytes."""
    header_text = raw.decode().lower()
    content_length = -1
    is_chunked = 'transfer-encoding: chunked' in header_text
    for line in raw.decode().split('\r\n'):
        if line.lower().startswith('content-length:'):
            content_length = int(line.split(':')[1].strip())
            break
    return content_length, is_chunked


def _body_complete(body: bytes, content_length: int, is_chunked: bool) -> bool:
    """Check whether we've received the full response body."""
    if content_length >= 0:
        return len(body) >= content_length
    if is_chunked:
        return b'\r\n0\r\n' in body or body.endswith(b'0\r\n\r\n')
    return False


def _recv_response(sock: socket.socket) -> bytes:
    """Read a complete HTTP response from the socket."""
    buf = b''
    content_length = -1
    is_chunked = False
    header_end = -1

    while True:
        try:
            chunk = sock.recv(4096)
        except TimeoutError:
            if header_end >= 0:
                break
            raise
        if not chunk:
            break
        buf += chunk

        if header_end < 0 and b'\r\n\r\n' in buf:
            header_end = buf.index(b'\r\n\r\n')
            content_length, is_chunked = _parse_headers(buf[:header_end])

        if header_end >= 0:
            body = buf[header_end + 4:]
            if _body_complete(body, content_length, is_chunked):
                break
            if content_length < 0 and not is_chunked:
                sock.settimeout(0.5)
                try:
                    extra = sock.recv(4096)
                    if extra:
                        buf += extra
                except TimeoutError:
                    pass
                break

    return buf


def _parse_response(raw: bytes) -> tuple[int, dict | str]:
    """Parse raw HTTP response bytes into (status_code, body)."""
    text = raw.decode('utf-8', errors='replace')
    if '\r\n\r\n' not in text:
        return 500, 'Failed to parse response'
    header, body = text.split('\r\n\r\n', 1)
    status_code = int(header.split('\r\n')[0].split(' ')[1])
    if not body.strip():
        return status_code, {}
    try:
        return status_code, json.loads(body)
    except json.JSONDecodeError:
        return status_code, body


def _docker_request(method: str, path: str, body: dict | None = None,
                    timeout: float = 10.0) -> tuple[int, dict | str]:
    """Make a request to the Docker Engine API via Unix socket."""
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        sock.connect(DOCKER_SOCKET)
        sock.settimeout(timeout)
        _send_request(sock, method, path, body)
        return _parse_response(_recv_response(sock))
    except Exception as e:
        return 500, str(e)
    finally:
        sock.close()


def _remove_container():
    """Force-remove the wifi-connect container."""
    _docker_request('DELETE', f'/containers/{CONTAINER_NAME}?force=true', timeout=30.0)


def is_running() -> bool:
    """Check if wifi-connect container is currently running."""
    status, response = _docker_request('GET', f'/containers/{CONTAINER_NAME}/json')
    if status == 200 and isinstance(response, dict):
        result: bool = response.get('State', {}).get('Running', False)
        return result
    return False


def stop() -> bool:
    """Stop the wifi-connect container if running."""
    status, _ = _docker_request('POST', f'/containers/{CONTAINER_NAME}/stop?t=10')
    return status in (204, 304)


def _create_container(force_mode: bool) -> bool:
    """Create and start the wifi-connect container. Returns True on success."""
    status, _ = _docker_request('GET', f'/containers/{CONTAINER_NAME}/json', timeout=2.0)
    if status == 200:
        _remove_container()

    env_vars = ['DBUS_SYSTEM_BUS_ADDRESS=unix:path=/host/run/dbus/system_bus_socket']
    if force_mode:
        env_vars.insert(0, 'WIFI_CONNECT_FORCE=true')

    config = {
        'Image': WIFI_CONNECT_IMAGE,
        'Env': env_vars,
        'HostConfig': {
            'NetworkMode': 'host',
            'CapAdd': ['NET_ADMIN'],
            'Binds': [
                '/var/run/dbus:/host/run/dbus',
                '/etc/NetworkManager/dnsmasq-shared.d:/etc/NetworkManager/dnsmasq-shared.d',
            ],
        },
    }

    name = urllib.parse.quote(CONTAINER_NAME)
    status, _ = _docker_request('POST', f'/containers/create?name={name}', config)
    if status not in (200, 201):
        return False

    status, _ = _docker_request('POST', f'/containers/{CONTAINER_NAME}/start')
    return status in (204, 304)


def _wait_for_container() -> bool:
    """Block until the container exits. Returns True if exit code was 0."""
    status, response = _docker_request(
        'POST', f'/containers/{CONTAINER_NAME}/wait', timeout=600.0)
    threading.Thread(target=_remove_container, daemon=True).start()
    if status == 200 and isinstance(response, dict):
        result: bool = response.get('StatusCode', 1) == 0
        return result
    return True


def start(force_mode: bool,
          callback: Optional[Callable[[bool], None]]) -> bool:
    """Start the wifi-connect container in a background thread."""

    def run():
        try:
            ok = _create_container(force_mode)
            if not ok:
                if callback:
                    callback(False)
                return
            success = _wait_for_container()
            if callback:
                callback(success)
        except Exception:
            if callback:
                callback(False)

    threading.Thread(target=run, daemon=True).start()
    return True
