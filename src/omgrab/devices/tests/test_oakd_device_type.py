"""Tests for oakd_device_type.py — device type enum, parser, and pipeline detection."""
from unittest import mock

import pytest

from omgrab.devices import oakd_device_type


@pytest.mark.parametrize(
    ('product_name', 'expected'),
    [
        ('OAK-D-PRO-W-97', oakd_device_type.OakDDeviceType.OAKD_PRO_WIDE),
        ('oak-d-pro-w-97', oakd_device_type.OakDDeviceType.OAKD_PRO_WIDE),
        ('OAK-D-LITE', oakd_device_type.OakDDeviceType.OAKD_LITE),
        ('OAK-D-W-123', oakd_device_type.OakDDeviceType.OAKD_WIDE),
    ],
)
def test_parse_device_type(
        product_name: str,
        expected: oakd_device_type.OakDDeviceType):
    assert oakd_device_type.parse_device_type(product_name) == expected


def test_parse_device_type_unknown():
    with pytest.raises(ValueError):
        oakd_device_type.parse_device_type('UNKNOWN-DEVICE')


class TestDetectFromPipeline:
    """Tests for detect_from_pipeline() with mock DepthAI objects."""

    def _make_mock_pipeline(self, product_name: str) -> mock.Mock:
        """Create a mock pipeline that returns the given product name."""
        pipeline = mock.Mock()
        eeprom = mock.Mock()
        eeprom.productName = product_name
        pipeline.getDefaultDevice().readCalibration().getEepromData.return_value = eeprom
        return pipeline

    def test_detects_pro_wide(self):
        pipeline = self._make_mock_pipeline('OAK-D-PRO-W-97')
        result = oakd_device_type.detect_from_pipeline(pipeline)
        assert result == oakd_device_type.OakDDeviceType.OAKD_PRO_WIDE

    def test_detects_wide(self):
        pipeline = self._make_mock_pipeline('OAK-D-W-123')
        result = oakd_device_type.detect_from_pipeline(pipeline)
        assert result == oakd_device_type.OakDDeviceType.OAKD_WIDE

    def test_detects_lite(self):
        pipeline = self._make_mock_pipeline('OAK-D-LITE')
        result = oakd_device_type.detect_from_pipeline(pipeline)
        assert result == oakd_device_type.OakDDeviceType.OAKD_LITE

    def test_returns_none_when_no_device(self):
        pipeline = mock.Mock()
        pipeline.getDefaultDevice.return_value = None
        result = oakd_device_type.detect_from_pipeline(pipeline)
        assert result is None

    def test_returns_none_when_no_product_name(self):
        pipeline = mock.Mock()
        eeprom = mock.Mock(spec=[])  # No productName attribute
        pipeline.getDefaultDevice().readCalibration().getEepromData.return_value = eeprom
        result = oakd_device_type.detect_from_pipeline(pipeline)
        assert result is None

    def test_returns_none_on_exception(self):
        pipeline = mock.Mock()
        pipeline.getDefaultDevice.side_effect = RuntimeError('device error')
        result = oakd_device_type.detect_from_pipeline(pipeline)
        assert result is None

    def test_returns_none_for_unknown_product(self):
        pipeline = self._make_mock_pipeline('TOTALLY-UNKNOWN-DEVICE')
        result = oakd_device_type.detect_from_pipeline(pipeline)
        assert result is None
