"""OAK-D device type detection from EEPROM."""
from typing import Optional

import enum
import logging

logger = logging.getLogger(__name__)


class OakDDeviceType(enum.Enum):
    """OAK-D hardware variant detected from EEPROM."""

    OAKD_PRO_WIDE = 'oakd_pro_wide'
    OAKD_WIDE = 'oakd_wide'
    OAKD_LITE = 'oakd_lite'


def parse_device_type(product_name: str) -> OakDDeviceType:
    """Parse an OAK-D device type from EEPROM product name.

    Args:
        product_name: EEPROM productName string.

    Returns:
        Parsed device type.

    Raises:
        ValueError: If the product name doesn't match known variants.
    """
    name = (product_name or '').upper()
    # Order matters: PRO-W would otherwise match WIDE.
    if 'PRO-W' in name or 'PRO_W' in name or 'PRO W' in name:
        return OakDDeviceType.OAKD_PRO_WIDE
    if 'LITE' in name:
        return OakDDeviceType.OAKD_LITE
    if 'OAK-D-W' in name or 'OAK-D W' in name or 'WIDE' in name:
        return OakDDeviceType.OAKD_WIDE
    raise ValueError(f'Unknown OAK-D product name: {product_name}')


def detect_from_pipeline(pipeline) -> Optional[OakDDeviceType]:
    """Detect OAK-D device type by reading EEPROM from a running pipeline.

    Args:
        pipeline: A running DepthAI pipeline.

    Returns:
        Detected device type, or None if detection failed.
    """
    try:
        device = pipeline.getDefaultDevice()
        if device is None:
            logger.warning('Could not get device from pipeline')
            return None
        calib = device.readCalibration()
        eeprom = calib.getEepromData()
        product_name = getattr(eeprom, 'productName', None)
        if product_name:
            return parse_device_type(product_name)
        logger.warning('Could not read device type from EEPROM')
        return None
    except Exception as e:
        logger.warning('Failed to read device type from EEPROM: %s', e)
        return None
