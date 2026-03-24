"""Tests for the capture device abstraction (devices/capture_device.py)."""
from typing import Optional

import contextlib

import pytest

from omgrab import testing
from omgrab.devices import capture_device


class TestCaptureDeviceProtocol:
    """Verify CaptureDevice protocol is runtime-checkable.

    Conforming / non-conforming objects are classified correctly.
    """

    def test_protocol_is_runtime_checkable(self):
        """CaptureDevice should be decorated with @runtime_checkable."""
        assert isinstance(capture_device.CaptureDevice, type)
        # runtime_checkable protocols support isinstance checks.
        assert hasattr(capture_device.CaptureDevice, '__protocol_attrs__') or \
            issubclass(type(capture_device.CaptureDevice), type)

    def test_fake_capture_device_is_instance(self):
        """FakeCaptureDevice should satisfy isinstance(obj, CaptureDevice)."""
        fake = testing.FakeCaptureDevice()
        assert isinstance(fake, capture_device.CaptureDevice)

    def test_plain_object_is_not_instance(self):
        """An arbitrary object should not satisfy CaptureDevice."""
        assert not isinstance(object(), capture_device.CaptureDevice)

    def test_partial_implementation_is_not_instance(self):
        """An object missing required methods should not satisfy the protocol."""

        class Incomplete:
            @property
            def connected(self) -> bool:
                return True

            # Missing: ready, device_type, wait_until_ready, __enter__, __exit__, preview

        assert not isinstance(Incomplete(), capture_device.CaptureDevice)



class TestFakeCaptureDeviceDefaults:
    """Verify FakeCaptureDevice default construction matches protocol semantics."""

    def test_connected_defaults_true(self):
        """A freshly-created device should report connected by default."""
        dev = testing.FakeCaptureDevice()
        assert dev.connected is True

    def test_ready_defaults_true(self):
        """A freshly-created device should report ready by default."""
        dev = testing.FakeCaptureDevice()
        assert dev.ready is True

    def test_device_type_defaults_none(self):
        """device_type should be None when no type is specified."""
        dev = testing.FakeCaptureDevice()
        assert dev.device_type is None

    def test_not_opened_initially(self):
        """Device should not be in the opened state before __enter__."""
        dev = testing.FakeCaptureDevice()
        assert dev.opened is False


class TestFakeCaptureDeviceConfigurable:
    """Verify FakeCaptureDevice constructor overrides."""

    def test_connected_override_false(self):
        """connected=False should be respected."""
        dev = testing.FakeCaptureDevice(connected=False)
        assert dev.connected is False

    def test_ready_override_false(self):
        """ready=False should be respected."""
        dev = testing.FakeCaptureDevice(ready=False)
        assert dev.ready is False

    def test_device_type_override(self):
        """A string device_type should be assignable via constructor."""
        dev = testing.FakeCaptureDevice(device_type='oakd_pro_wide')
        assert dev.device_type == 'oakd_pro_wide'


class TestFakeCaptureDeviceContextManager:
    """Verify context manager (__enter__/__exit__) behaviour."""

    def test_enter_sets_opened(self):
        """__enter__ should set opened to True."""
        dev = testing.FakeCaptureDevice()
        dev.__enter__()
        assert dev.opened is True

    def test_exit_clears_opened(self):
        """__exit__ should set opened back to False."""
        dev = testing.FakeCaptureDevice()
        dev.__enter__()
        dev.__exit__(None, None, None)
        assert dev.opened is False

    def test_with_statement_lifecycle(self):
        """Using `with` should open and then close the device."""
        dev = testing.FakeCaptureDevice()

        with dev as d:
            assert d is dev
            assert dev.opened is True

        assert dev.opened is False

    def test_enter_returns_self(self):
        """__enter__ should return the device instance for `as` binding."""
        dev = testing.FakeCaptureDevice()
        result = dev.__enter__()
        assert result is dev
        dev.__exit__(None, None, None)

    def test_exit_does_not_suppress_exceptions(self):
        """__exit__ should return False so exceptions propagate."""
        dev = testing.FakeCaptureDevice()
        dev.__enter__()
        result = dev.__exit__(ValueError, ValueError('oops'), None)
        assert result is False

    def test_exit_restores_state_on_exception(self):
        """Device should be closed even when the with-block raises."""
        dev = testing.FakeCaptureDevice()

        with pytest.raises(RuntimeError), dev:
            raise RuntimeError('boom')

        assert dev.opened is False

    def test_double_enter_raises(self):
        """Entering an already-open device should raise RuntimeError.

        The real OakDCaptureDevice enforces this ('OakD camera already in use').
        The fake should enforce the same invariant so that tests catch
        accidental double-open bugs in code under test.
        """
        dev = testing.FakeCaptureDevice()
        dev.__enter__()

        with pytest.raises(RuntimeError):
            dev.__enter__()

        dev.__exit__(None, None, None)

    def test_multiple_enter_exit_cycles(self):
        """Device should support being opened and closed multiple times."""
        dev = testing.FakeCaptureDevice()

        for _ in range(3):
            with dev:
                assert dev.opened is True
            assert dev.opened is False


class TestFakeCaptureDeviceWaitUntilReady:
    """Verify wait_until_ready() behaviour."""

    def test_returns_true_when_ready(self):
        """Should return True when device is ready."""
        dev = testing.FakeCaptureDevice(ready=True)
        assert dev.wait_until_ready() is True

    def test_returns_false_when_not_ready(self):
        """Should return False when device is not ready."""
        dev = testing.FakeCaptureDevice(ready=False)
        assert dev.wait_until_ready() is False

    def test_accepts_timeout_parameter(self):
        """wait_until_ready should accept an optional timeout without error."""
        dev = testing.FakeCaptureDevice(ready=True)
        assert dev.wait_until_ready(timeout=5.0) is True

    def test_timeout_none_accepted(self):
        """Explicit timeout=None should work (means wait forever)."""
        dev = testing.FakeCaptureDevice(ready=True)
        assert dev.wait_until_ready(timeout=None) is True


class TestFakeCaptureDevicePreview:
    """Verify preview() behaviour on FakeCaptureDevice."""

    def test_preview_raises_preview_unavailable(self):
        """FakeCaptureDevice.preview() should raise PreviewUnavailableError."""
        dev = testing.FakeCaptureDevice()
        with pytest.raises(capture_device.PreviewUnavailableError), dev.preview():
            pass



class TestProtocolShape:
    """Verify that the Protocol defines the expected set of attributes."""

    def test_connected_is_property(self):
        """CaptureDevice should define 'connected' as a property."""
        assert hasattr(capture_device.CaptureDevice, 'connected')

    def test_ready_is_property(self):
        """CaptureDevice should define 'ready' as a property."""
        assert hasattr(capture_device.CaptureDevice, 'ready')

    def test_device_type_is_property(self):
        """CaptureDevice should define 'device_type' as a property."""
        assert hasattr(capture_device.CaptureDevice, 'device_type')

    def test_wait_until_ready_is_callable(self):
        """CaptureDevice should define 'wait_until_ready' as a method."""
        assert callable(getattr(capture_device.CaptureDevice, 'wait_until_ready', None))

    def test_enter_is_callable(self):
        """CaptureDevice should define '__enter__'."""
        assert callable(getattr(capture_device.CaptureDevice, '__enter__', None))

    def test_exit_is_callable(self):
        """CaptureDevice should define '__exit__'."""
        assert callable(getattr(capture_device.CaptureDevice, '__exit__', None))

    def test_preview_is_callable(self):
        """CaptureDevice should define 'preview'."""
        assert callable(getattr(capture_device.CaptureDevice, 'preview', None))



class TestCustomCaptureDeviceCompliance:
    """Verify that a minimal custom implementation satisfies the protocol."""

    def _make_minimal_device(self):
        """Build a minimal object that satisfies CaptureDevice."""

        class MinimalDevice:
            @property
            def label(self) -> str:
                return 'minimal'

            @property
            def connected(self) -> bool:
                return True

            @property
            def ready(self) -> bool:
                return False

            @property
            def device_type(self) -> Optional[str]:
                return None

            def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
                return self.ready

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                return False

            def preview(self):
                return contextlib.nullcontext(self)

        return MinimalDevice()

    def test_minimal_device_satisfies_protocol(self):
        """A minimal implementation should pass isinstance check."""
        dev = self._make_minimal_device()
        assert isinstance(dev, capture_device.CaptureDevice)

    def test_minimal_device_usable_as_context_manager(self):
        """A minimal implementation should work with `with` statement."""
        dev = self._make_minimal_device()
        with dev as d:
            assert d is dev
